from dataclasses import dataclass
import math
from typing import Tuple,Optional
import torch
from torch import nn
import torch.nn.functional as F
from .utils import *

# Llama2-7B config
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    # Start of modifications to the original LLaMA-2-7B model
    adapter_len: int = 10
    adapter_layer: int = 0
    w_adapter: bool = True#True
    prefix_adapter:bool=True
    w_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.05
    target_modules: Tuple[str] = ('down_proj', 'up_proj', 'gate_proj')     # Option

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, lora_r, lora_alpha, lora_dropout=0.05,
    ):
        super().__init__()

        if lora_r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {lora_r} must be less or equal than {min(in_features, out_features)}"
            )
        self.lora_r = lora_r
        self.lora_down = nn.Linear(in_features, lora_r, bias=False)
        self.dropout = nn.Dropout(lora_dropout)
        self.lora_up = nn.Linear(lora_r, out_features, bias=False)
        self.scale = 1. * lora_alpha / lora_r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        x = x.to(self.lora_up.weight.dtype)
        result = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        result = result.to(previous_dtype)
        return result

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, args: ModelArgs):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.params = args
        self.w1 = nn.Linear(dim, hidden_dim, bias=False,)    # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False,)    # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False,)    # up_proj
    #Start of modifications to the original LLaMA-2-7B model
        if self.params.w_lora:
            if 'up_proj' in args.target_modules:
                self.lora_w3 = LoraInjectedLinear(self.w3.in_features, self.w3.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'down_proj' in args.target_modules:
                self.lora_w2 = LoraInjectedLinear(self.w2.in_features, self.w2.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'gate_proj' in args.target_modules:
                self.lora_w1 = LoraInjectedLinear(self.w1.in_features, self.w1.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    #End of modifications to the original LLaMA-2-7B model
    def forward(self, x):
        previous_dtype = x.dtype
        x = x.to(self.w1.weight.dtype)

        up_x = self.w3(x)
        gate_x = self.w1(x)
        # Start of modifications to the original LLaMA-2-7B model
        if self.params.w_lora:
            if 'up_proj' in self.params.target_modules:
                up_x = up_x + self.lora_w3(x)
            if 'gate_proj' in self.params.target_modules:
                gate_x = gate_x + self.lora_w1(x)
        #End of modifications to the original LLaMA-2-7B model
        down_input = F.silu(gate_x) * up_x
        out = self.w2(down_input)
        # Start of modifications to the original LLaMA-2-7B model
        if self.params.w_lora:
            if 'down_proj' in self.params.target_modules:
                out = out + self.lora_w2(down_input)
        # End of modifications to the original LLaMA-2-7B model
        return out

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, args=args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None,
            freqs_cis_prefix=None
    ):#add adapter,freqs_cis_prefix
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, freqs_cis_prefix)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False,)
        # Start of modifications to the original LLaMA-2-7B model
        self.w_lora = args.w_lora
        self.target_modules = args.target_modules
        if self.w_lora:
            if 'q_proj' in args.target_modules:
                self.lora_wq = LoraInjectedLinear(self.wq.in_features, self.wq.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'k_proj' in args.target_modules:
                self.lora_wk = LoraInjectedLinear(self.wk.in_features, self.wk.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'v_proj' in args.target_modules:
                self.lora_wv = LoraInjectedLinear(self.wv.in_features, self.wv.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'o_proj' in args.target_modules:
                self.lora_wo = LoraInjectedLinear(self.wo.in_features, self.wo.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None
    # End of modifications to the original LLaMA-2-7B model
    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None

    def forward(
            self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor], adapter=None, freqs_cis_prefix=None
    ):
        previous_dtype = x.dtype
        x = x.to(self.wq.weight.dtype)


        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Start of modifications to the original LLaMA-2-7B model
        if self.w_lora:
            if 'q_proj' in self.target_modules:
                xq = xq + self.lora_wq(x)
            if 'k_proj' in self.target_modules:
                xk = xk + self.lora_wk(x)
            if 'v_proj' in self.target_modules:
                xv = xv + self.lora_wv(x)
        # End of modifications to the original LLaMA-2-7B model
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)#
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # Start of modifications to the original LLaMA-2-7B model
        if adapter is not None:#prefix key value
            adapter_key, adapter_value = adapter
            adapter_len = adapter_key.shape[1]
            adapter_k = self.wk(adapter_key)
            adapter_k = adapter_k.view(bsz, adapter_len, self.n_heads, self.head_dim)
            adapter_v = self.wv(adapter_value)
            adapter_v = adapter_v.view(bsz, adapter_len, self.n_heads, self.head_dim)
            # adapter_k = apply_rotary_emb_single(adapter_k, freqs_cis=freqs_cis_prefix)#k应用了旋转位置编码
            adapter_k = adapter_k.transpose(1, 2)
            adapter_v = adapter_v.transpose(1, 2)
        # End of modifications to the original LLaMA-2-7B model
        keys = xk
        values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # Start of modifications to the original LLaMA-2-7B model
        if self.cache_enabled:
            if self.cache_k is None:
                assert start_pos == 0
                self.cache_k, self.cache_v = keys, values
            else:
                assert self.cache_k.size(2) >= start_pos
                self.cache_k = torch.cat([self.cache_k[:, :, :start_pos], keys], dim=2)
                self.cache_v = torch.cat([self.cache_v[:, :, :start_pos], values], dim=2)
                keys, values = self.cache_k, self.cache_v
        if adapter is not None:
            keys = torch.cat([adapter_k, keys], dim=2)
            values = torch.cat([adapter_v, values], dim=2)
        # End of modifications to the original LLaMA-2-7B model
        output = self._forward_scaled_dot_product_attention(xq, keys, values, attention_mask=mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        if self.w_lora and 'o_proj' in self.target_modules:
            return self.wo(output) + self.lora_wo(output)
        else:
            return self.wo(output)

    def _forward_scaled_dot_product_attention(self, q, k, v, attention_mask=None):
        if False and hasattr(F, "scaled_dot_product_attention"):
           return F.scaled_dot_product_attention(q, k, v, attention_mask if attention_mask is not None else None)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = torch.matmul(attn_weights, v)
        return attn_weights

class Transformer(nn.Module):
    def __init__(self,params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim,
        )
        # Start of modifications to the original LLaMA-2-7B model
        self.w_adapter = params.w_adapter
        self.prefix_adapter_use=params.prefix_adapter
        self.adapter_len = params.adapter_len
        self.criterion = torch.nn.CrossEntropyLoss()
        # End of modifications to the original LLaMA-2-7B model
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output_dim=params.output_dim
        self.score = nn.Linear(params.dim, self.output_dim, bias=False,)
        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
        # Start of modifications to the original LLaMA-2-7B model

        self.task_type=params.task_type
        self.instruct_ids,self.instruct_mask,self.response_ids,self.response_mask=params.prompter.generate_prompt()
        self.user_embedding=params.user_embed
        self.item_embedding=params.item_embed
        self.item_embedding_sequential=params.SASRec_item_embed
        self.gnn_input_user_proj=nn.Linear(self.user_embedding.weight.shape[-1],params.dim, bias=False)

        self.gnn_input_item_proj= nn.Sequential(
            nn.Linear(64*2, params.dim),
            nn.GELU(),
            nn.Linear(params.dim, params.dim)
        )

        self.gnn_nodes_norm=True
        self.gnn_structure_norm=True

        self.gnn_input_user_norm=RMSNorm(self.user_embedding.weight.shape[-1], eps=params.norm_eps)
        self.gnn_input_item_norm=RMSNorm(self.item_embedding.weight.shape[-1], eps=params.norm_eps)
        self.gnn_input_item_norm_post=RMSNorm(params.dim, eps=params.norm_eps)


        if self.w_adapter:
            if self.prefix_adapter_use:
                self.prefix_adapter = PrefixEncoder(params)
            self.gnn_bimap=nn.Linear(self.user_embedding.weight.shape[-1],64)

            self.gnn_global_key = nn.Sequential(
                        nn.Linear(4096, 4096),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(4096, self.params.dim),
                        nn.GELU(),
                        nn.Dropout(0.3)
                        )
            self.gnn_global_value = nn.Sequential(
                        nn.Linear(4096, 4096),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(4096, self.params.dim),
                        nn.GELU(),
                        nn.Dropout(0.3)
                        )
            self.gnn_global_norm=RMSNorm(4096,eps=params.norm_eps)
            self.gnn_global_key_norm=RMSNorm(4096,eps=params.norm_eps)
            self.gnn_global_value_norm=RMSNorm(4096,eps=params.norm_eps)
            self.gnn_global_key_norm_pre=RMSNorm(4096,eps=params.norm_eps)
            self.gnn_global_value_norm_pre=RMSNorm(4096,eps=params.norm_eps)

            self.gnn_global_features=nn.Parameter(torch.cat((self.user_embedding.weight,self.item_embedding.weight),dim=0),requires_grad=False)

        # End of modifications to the original LLaMA-2-7B model
    def forward(self, input_ids, labels, attention_mask=None): #node_ids, attention_mask=None):
        _bsz, seqlen = input_ids.shape
        input_ids_sequential=input_ids+1
        past_key_values_length = self.adapter_len

        instruct_embeds=self.tok_embeddings(self.instruct_ids).expand(_bsz,-1,-1)
        response_embeds=self.tok_embeddings(self.response_ids).expand(_bsz,-1,-1)

        instruct_mask = self.instruct_mask.expand(_bsz, -1)
        response_mask = self.response_mask.expand(_bsz, -1)

        if self.task_type == 'general':
            gnn_input_user=self.user_embedding(input_ids)
            gnn_input_user_norm=self.gnn_input_user_norm(gnn_input_user)

        else:
            gnn_input_item = self.item_embedding(input_ids)
            seq_input_item=self.item_embedding_sequential(input_ids_sequential)

            inputs = self.gnn_input_item_proj(torch.cat((gnn_input_item,seq_input_item),dim=-1))

        inputs_embeds = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, attention_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs_embeds.size()[0] and attention_mask.size()[1] == inputs_embeds.size()[1]
        seqlen=attention_mask.size()[1]

        freqs_cis = self.freqs_cis.to(inputs_embeds.device)[past_key_values_length:]
        freqs_cis_prefix = self.freqs_cis.to(inputs_embeds.device)[:past_key_values_length]

        #For decoder_layer e.g.:[0,0,0,1,1,1,1] position_id:[0,0,0,0,1,2,3]
        # position_id = torch.arange(seqlen).repeat(_bsz, 1).to(inputs_embeds.device)
        # position_id = position_id - ((attention_mask == 0).sum(dim=-1)).unsqueeze(-1)
        # position_id[position_id < 0] = 0

        #For casual position attention_mask apply rotary_emb e.g.:[0,0,1,1,1,0,0,1,1]->[0,0,0,1,2,0,0,3,4]
        position_id=(attention_mask.cumsum(dim=-1)-1)*attention_mask
        freqs_cis = freqs_cis[position_id]

        if past_key_values_length > 0:
            prefix_attention_mask = torch.ones(
                (_bsz, past_key_values_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(
                (_bsz, seqlen), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (_bsz, seqlen), inputs_embeds, past_key_values_length)

        start_pos, adapter = 0, None

        if self.w_adapter:
            adapter = self.adapter_forward(gnn_input_item,batch_size=_bsz)#, node_ids=node_ids)

        h = self.transformer_forward(input_embeds=inputs_embeds, freqs_cis=freqs_cis,
                                     attention_mask=attention_mask, adapter=adapter,
                                     freqs_cis_prefix=freqs_cis_prefix if self.w_adapter else None
                                     )

        h = self.norm(h)
        previous_dtype = h.dtype
        h = h.to(self.score.weight.dtype)
        output = self.score(h[:,-1])
        c_loss = self.criterion(output.view(-1,self.output_dim), labels.view(-1))

        return c_loss

    def transformer_forward(self, input_embeds, freqs_cis, attention_mask, adapter, freqs_cis_prefix=None,
                            start_pos=0):
        h = input_embeds

        if adapter is None:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, attention_mask)
            return h

        adapter_index = 0
        adapter_key, adapter_value = adapter[0], adapter[1]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, attention_mask,
                      (adapter_key[:, adapter_index].bfloat16(), adapter_value[:, adapter_index].bfloat16()),
                      freqs_cis_prefix)

            adapter_index = adapter_index + 1

        return h

    @torch.inference_mode()
    def forward_inference(self, input_ids, attention_mask=None): #node_ids, attention_mask=None):
        _bsz, seqlen = input_ids.shape
        input_ids_sequential=input_ids+1
        past_key_values_length = self.adapter_len

        instruct_embeds=self.tok_embeddings(self.instruct_ids).expand(_bsz,-1,-1)
        response_embeds=self.tok_embeddings(self.response_ids).expand(_bsz,-1,-1)

        instruct_mask = self.instruct_mask.expand(_bsz, -1)
        response_mask = self.response_mask.expand(_bsz, -1)

        if self.task_type == 'general':
            gnn_input_user=self.user_embedding(input_ids)
            gnn_input_user_norm=self.gnn_input_user_norm(gnn_input_user)

        else:
            gnn_input_item = self.item_embedding(input_ids)
            seq_input_item=self.item_embedding_sequential(input_ids_sequential)

            inputs = self.gnn_input_item_proj(torch.cat((gnn_input_item,seq_input_item),dim=-1))

        inputs_embeds = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, attention_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs_embeds.size()[0] and attention_mask.size()[1] == inputs_embeds.size()[1]
        seqlen=attention_mask.size()[1]
        # inputs_embeds = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis.to(inputs_embeds.device)[past_key_values_length:]
        freqs_cis_prefix = self.freqs_cis.to(inputs_embeds.device)[:past_key_values_length]
        #For decoder_layer e.g.:[0,0,0,1,1,1,1] position_id:[0,0,0,0,1,2,3]
        # position_id = torch.arange(seqlen).repeat(_bsz, 1).to(inputs_embeds.device)
        # position_id = position_id - ((attention_mask == 0).sum(dim=-1)).unsqueeze(-1)
        # position_id[position_id < 0] = 0

        #For casual position attention_mask apply rotary_emb e.g.:[0,0,1,1,1,0,0,1,1]->[0,0,0,1,2,0,0,3,4]
        position_id=(attention_mask.cumsum(dim=-1)-1)*attention_mask
        freqs_cis = freqs_cis[position_id]
        if past_key_values_length > 0:
            prefix_attention_mask = torch.ones(
                (_bsz, past_key_values_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(
                (_bsz, seqlen), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (_bsz, seqlen), inputs_embeds, past_key_values_length)
        start_pos, adapter = 0, None
        if self.w_adapter:
            adapter = self.adapter_forward(gnn_input_item,batch_size=_bsz)#, node_ids=node_ids)
        h = self.transformer_forward(input_embeds=inputs_embeds, freqs_cis=freqs_cis,
                                     attention_mask=attention_mask, adapter=adapter,
                                     freqs_cis_prefix=freqs_cis_prefix if self.w_adapter else None
                                     )
        h = self.norm(h)
        previous_dtype = h.dtype
        h = h.to(self.score.weight.dtype)
        output = self.score(h[:,-1])

        return h,output.view(-1,self.output_dim)

    def adapter_forward(self, inputs,batch_size):#, node_ids):

        if self.prefix_adapter_use:
            p_adapter_key, p_adapter_value = self.prefix_adapter()
            p_adapter_key = p_adapter_key.repeat(batch_size, 1, 1, 1)
            p_adapter_value = p_adapter_value.repeat(batch_size, 1, 1, 1)

        gnn_global=inputs
        gnn_global=gnn_global.transpose(1,2)@gnn_global
        gnn_global=torch.flatten(gnn_global,start_dim=1,end_dim=2)

        gnn_global_key=self.gnn_global_key(gnn_global)
        gnn_global_value=self.gnn_global_value(gnn_global)

        gnn_global_key = gnn_global_key.unsqueeze(1).unsqueeze(1).repeat(1, self.params.n_layers, self.adapter_len, 1)
        gnn_global_value = gnn_global_value.unsqueeze(1).unsqueeze(1).repeat(1, self.params.n_layers, self.adapter_len,
                                                                             1)
        gnn_global_key=self.gnn_global_key_norm_pre(gnn_global_key)
        gnn_global_value=self.gnn_global_value_norm_pre(gnn_global_value)

        # print(p_adapter_value.shape,gnn_global_key.shape)
        if self.prefix_adapter_use:
            adapter_key = p_adapter_key + gnn_global_key
            adapter_value = p_adapter_value + gnn_global_value
        else:
            adapter_key=gnn_global_key
            adapter_value=gnn_global_value

        adapter_key=self.gnn_global_key_norm(adapter_key)
        adapter_value=self.gnn_global_value_norm(adapter_value)
        adapter = (adapter_key, adapter_value)
        return adapter

    def set_trainable_params_new(self):
        param_adapter, param_lora  = [],  []
        adapter = ["graph_adapter", "prefix_adapter", "up_projection", "down_projection","gnn"]

        for name, param in self.named_parameters():
            if any(n in name for n in adapter):
                param.requires_grad = True
                param.data = param.data.float()
                param_adapter.append(param)
            elif "lora" in name:
                param.requires_grad = True
                param.data = param.data.float()
                param_lora.append(param)
            else:
                param.requires_grad = False

        return param_adapter, param_lora

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param

    def print_trainable_names(self):
        for name,param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def save_trainable_params(self):
        save = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                save[name]=param
        return save

    def enable_cache(self):
        for layer in self.layers:
            layer.attention.enable_cache()

    def disable_cache(self):
        for layer in self.layers:
            layer.attention.disable_cache()

class PrefixEncoder(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super().__init__()
        self.dim = params.dim
        self.adapter_len, self.adapter_layer = params.adapter_len, params.n_layers
        self.prefix_keys = nn.Parameter(torch.randn(1, self.adapter_layer, self.adapter_len, self.dim), requires_grad=True)
        self.prefix_values = nn.Parameter(torch.randn(1, self.adapter_layer, self.adapter_len, self.dim), requires_grad=True)

    def forward(self):
        return self.prefix_keys, self.prefix_values