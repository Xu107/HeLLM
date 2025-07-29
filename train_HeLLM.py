import torch
import torch.nn as nn

from config_HeLLM import parse_args_llama
from utils import *
from transformers import LlamaTokenizer
from llama.HeLLM  import Transformer,ModelArgs
from tqdm import tqdm
import pathlib
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  #0+1

args=parse_args_llama()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
seed_everything(args.seed)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
args.device=device

log_path=os.path.join(args.logging_dir,args.dataset,args.GNNModel)
logger=logging.Logger(log_path,args.target,args.logging)
logger.logging(args)

def main():

    # Step 1:Build Dataset and DataLoader

    dataset = SequentialDataset(args.dataset, args.max_len)
    GNN_Model_Path=f"datasets/sequential/{args.dataset}/{args.GNNModel}"
    user_embed,item_embed=nn.Embedding.from_pretrained(torch.load(GNN_Model_Path+"_user_embeddings.pt"),freeze=True),nn.Embedding.from_pretrained(torch.load(GNN_Model_Path+"_item_embeddings.pt"),freeze=True)
    SASRec_item_embed=nn.Embedding.from_pretrained(torch.load(f"datasets/sequential/{args.dataset}/SASRec_item_embedding.pt"),freeze=True)
    user_embed=user_embed.to(device)
    item_embed=item_embed.to(device)
    SASRec_item_embed=SASRec_item_embed.to(device)
    data_collator = SequentialCollator()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=True, collate_fn=data_collator)

    # Step 2:Build tokenizer and config
    model_args: ModelArgs = ModelArgs(w_lora=args.w_lora,
                                      w_adapter=args.w_adapter,
                                      adapter_len=args.adapter_len,
                                      prefix_adapter=args.prefix_adapter,
                                      # gnn_adapter=args.gnn_adapter,
                                      lora_alpha=args.lora_alpha,
                                      lora_r=args.lora_r,
                                      # input_nodes=args.input_nodes,
                                      # input_sequent=args.input_sequent
                                      )
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name,legacy=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    model_args.vocab_size = tokenizer.vocab_size

    # Step 3: Prompt generation index
    prompter = Prompter(tokenizer,args.task_type,args.prompt_template_name)
    model_args.prompter=prompter
    model_args.task_type=args.task_type
    model_args.output_dim=item_embed.weight.shape[0]

    # Step 4:Load Model

    model_args.user_embed=user_embed
    model_args.item_embed=item_embed

    model_args.multimodal=args.multimodal
    if args.multimodal:
        model_args.image_feat=image_feat
        model_args.text_feat=text_feat
        if args.dataset=="tiktok":
            model_args.audio_feat=audio_feat
    model_args.dataset=args.dataset
    model_args.SASRec_item_embed=SASRec_item_embed

    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    model: Transformer = Transformer(params=model_args)
    torch.set_default_tensor_type(torch.FloatTensor)

    ckpt = f"{args.model_name}/consolidated.00.pth"
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)

    save_path=os.path.join(args.output_dir,args.dataset,args.GNNModel)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)


    # Step 5:Set Optimizer
    param_adapter, param_lora = model.set_trainable_params_new()

    lr_group = {
        'adapter': args.lr,
        'lora': args.lr,
    }

    wd_group = {
        'adapter': args.wd,
        'lora': args.wd,
    }

    optimizer = torch.optim.AdamW(
        [
            {'params': param_adapter, 'lr': lr_group['adapter'], 'weight_decay': wd_group['adapter']},
            {'params': param_lora, 'lr': lr_group['lora'], 'weight_decay': wd_group['lora']},
        ],
        betas=(0.9, 0.95))


    if args.load_modal:
        pass


    trainable_params, all_param = model.print_trainable_params()
    logger.logging(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    # Step 6. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps),desc='train')

    model=model.to(device)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.
        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()

            input_ids=batch['inputs']
            labels=batch['labels']
            inputs_mask=batch['inputs_mask']

            input_ids=input_ids.to(device)
            labels=labels.to(device)
            inputs_mask=inputs_mask.to(device)

            loss = model(input_ids,labels,inputs_mask)
            loss.backward()

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr_group['adapter'], step / len(train_loader) + epoch,
                                     args)
                adjust_learning_rate(optimizer.param_groups[1], lr_group['lora'], step / len(train_loader) + epoch,
                                     args)
            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
            progress_bar.update(1)
            # break

        logger.logging(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")

        torch.save(model.save_trainable_params(), f"{save_path}/{args.target}_adapter_epoch_{epoch}.pth")
        checkpoint = {"optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(checkpoint, f"{save_path}/{args.target}_checkpoint_epoch_{epoch}.pth")

        # Step 7. Val
        val(model, dataset, epoch)
        test(model, dataset, epoch)

    # Step 8. Test
    val(model, dataset, epoch)
    test(model, dataset, epoch)
    # Step 9:Save Model
    torch.save(model.save_trainable_params(), f"{save_path}/{args.target}_adapter_epoch_{epoch}.pth")
    checkpoint = {"optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(checkpoint, f"{save_path}/{args.target}_checkpoint_epoch_{epoch}.pth")

def val(model,dataset,epoch=0):
    model.eval()
    topk = [5, 10, 20]
    results = {'Precision': np.zeros(len(topk)),
               'Recall': np.zeros(len(topk)),
               'MRR': np.zeros(len(topk)),
               'MAP': np.zeros(len(topk)),
               'NDCG': np.zeros(len(topk))}

    valData = dataset.valData
    users = np.arange(dataset.n_user)
    val_steps = dataset.n_user
    val_bar = tqdm(range(val_steps), desc='val')
    for u in users:

        if len(valData[u]) == 0:
            continue

        groundTruth = [[valData[u][1]]]
        inputs = torch.LongTensor(valData[u][0]).cuda().unsqueeze(0)
        inputs_mask = torch.ones(inputs.shape).long().cuda()
        _, ratings = model.forward_inference(inputs, inputs_mask)

        _, ratings_K = torch.topk(ratings, k=topk[-1])
        ratings_K = ratings_K.cpu().numpy()

        r = getLabel(groundTruth, ratings_K)
        for j, k in enumerate(topk):
            pre, rec = RecallPrecision_atK(groundTruth, r, k)
            mrr = MRR_atK(groundTruth, r, k)
            map = MAP_atK(groundTruth, r, k)
            ndcg = NDCG_atK(groundTruth, r, k)
            results['Precision'][j] += pre
            results['Recall'][j] += rec
            results['MRR'][j] += mrr
            results['MAP'][j] += map
            results['NDCG'][j] += ndcg
        val_bar.update(1)

    for key in results.keys():
        results[key] /= float(len(users))
    logger.logging('-' * 60 + "\n")
    logger.logging(f'Valid:{epoch} for User: \n')
    for j, k in enumerate(topk):
        logger.logging(f'Precision@{k}: {results["Precision"][j]} \n '
                       f'Recall@{k}: {results["Recall"][j]} \n '
                       f'MRR@{k}: {results["MRR"][j]} \n '
                       f'MAP@{k}: {results["MAP"][j]} \n '
                       f'NDCG@{k}: {results["NDCG"][j]} \n')

def test(model,dataset,epoch=0):
    model.eval()
    topk = [ 5, 10, 20]
    results = {'Precision': np.zeros(len(topk)),
               'Recall': np.zeros(len(topk)),
               'MRR': np.zeros(len(topk)),
               'MAP': np.zeros(len(topk)),
               'NDCG': np.zeros(len(topk))}

    testData = dataset.testData
    users = np.arange(dataset.n_user)
    test_steps = dataset.n_user
    test_bar = tqdm(range(test_steps),desc='test')
    test_user_num=0
    for u in users:

        if len(testData[u]) == 0:
            continue
        test_user_num += 1

        groundTruth = [[testData[u][1]]]
        inputs = torch.LongTensor(testData[u][0]).cuda().unsqueeze(0)
        inputs_mask = torch.ones(inputs.shape).long().cuda()
        _, ratings = model.forward_inference(inputs, inputs_mask)

        _, ratings_K = torch.topk(ratings, k=topk[-1])
        ratings_K = ratings_K.cpu().numpy()

        r = getLabel(groundTruth, ratings_K)
        for j, k in enumerate(topk):
            pre, rec = RecallPrecision_atK(groundTruth, r, k)
            mrr = MRR_atK(groundTruth, r, k)
            map = MAP_atK(groundTruth, r, k)
            ndcg = NDCG_atK(groundTruth, r, k)
            results['Precision'][j] += pre
            results['Recall'][j] += rec
            results['MRR'][j] += mrr
            results['MAP'][j] += map
            results['NDCG'][j] += ndcg
        test_bar.update(1)
        # break

    for key in results.keys():
        results[key] /= float(test_user_num)

    logger.logging('-'*60+"\n")
    logger.logging(f'Test:{epoch} for User: \n')
    for j, k in enumerate(topk):
        logger.logging(f'Precision@{k}: {results["Precision"][j]} \n '
              f'Recall@{k}: {results["Recall"][j]} \n '
              f'MRR@{k}: {results["MRR"][j]} \n '
              f'MAP@{k}: {results["MAP"][j]} \n '
              f'NDCG@{k}: {results["NDCG"][j]} \n')

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()