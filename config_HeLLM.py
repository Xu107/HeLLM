from os import path
import argparse

module_path = path.dirname(path.abspath(__file__))

def parse_args_llama():
    parser = argparse.ArgumentParser(description="GraphLLM")
    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--model_name", type=str, default='Llama-2-7b')
    parser.add_argument("--lr", type=float, default=0.0003)#5e-5,2e-4
    parser.add_argument("--wd", type=float, default=0.1)
    #Model
    parser.add_argument("--adapter_len", type=int, default=10)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--w_lora", type=bool, default=True)
    parser.add_argument("--w_adapter", type=bool, default=True)
    parser.add_argument("--gnn_adapter", type=bool, default=True)
    parser.add_argument("--prefix_adapter", type=bool, default=True)
    parser.add_argument("--input_nodes", type=bool, default=True)
    parser.add_argument("--input_sequent", type=bool, default=False)#SASRec
    parser.add_argument("--multimodal", type=bool, default=False)
    # Model Training1
    parser.add_argument("--batch_size", type=int, default=7)
    parser.add_argument("--grad_steps", type=int, default=1)#2
    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=20)#15
    parser.add_argument("--warmup_epochs", type=float, default=1)#1
    parser.add_argument("--prompt_template_name", type=str, default='alpaca')
    parser.add_argument("--task_type", type=str, default='sequential')
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument("--dataset", type=str, default='Sports_and_Outdoors',help="[Toys_and_Games,Beauty,Sports_and_Outdoors]")
    parser.add_argument("--GNNModel", type=str, default='MMHCL',
                        help="GNN Model :[LightGCN,MMHCL]")
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--logging", type=bool, default=True)
    parser.add_argument("--logging_dir", type=str, default='log')
    parser.add_argument("--target", type=str, default='Xu8_HeLLM')
    parser.add_argument("--load_modal", type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    args = parser.parse_args()
    return args