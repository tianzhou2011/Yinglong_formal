import math, random
import numpy as np
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from lit_gpt.config import Config
from xformers.ops import SwiGLU
import torch.nn.functional as F
from .fused_rotary_embedding import apply_rotary_emb_func
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
PretokenCache = torch.Tensor
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
from einops import rearrange


class quantitleLoss(torch.nn.Module):
    def __init__(self,  
                 qSize = 99,
                 patch_size = 16,
                 *args,**kwargs) -> None:
        
        super().__init__()
        self.qSize = qSize
        self.patch_size = patch_size
        
       
        q = np.array([i+1 for i in range(self.qSize)])
        q = q / (self.qSize + 1)
        q = q.reshape((1,1,-1))
        
        q_variance = q*(1-q)
        
        self.register_buffer('q', torch.tensor(q))
        self.register_buffer('q_variance', torch.tensor(q_variance))

        
    def forward(self, input: torch.Tensor, target: torch.Tensor,rel_loss = False) -> torch.Tensor:
        
        target = target.unsqueeze(-1)
        input = input[:,:target.shape[1],:,:]
        
        posPart = input - target
        negPart = -posPart
        
        raw_loss = torch.maximum(self.q * negPart, (1-self.q) * posPart)         

        target_absmean = torch.mean(target.abs(),dim = (1,2),keepdims = True)
        raw_loss = raw_loss  / torch.sqrt(self.q_variance)  / (target_absmean + 1e-4)
        
        return torch.mean(raw_loss)
        

def haarMatrix_unnormalized(n):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])
    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h


def haarMatrix(n,normalized = 'ortho'):
    h = haarMatrix_unnormalized(n)
    scaler = np.diag(1/np.sqrt(np.diag(h@h.transpose())))
    if normalized == 'ortho':
        return scaler @ h
    elif normalized == 'forward':
        return scaler @ h/ np.sqrt(n)
        
    else:
        return scaler @ h *  np.sqrt(n)
 
    
    
    
    
class Tokenizer(torch.nn.Module):
    def __init__(self, config: None, *args,**kwargs) -> None:
        super().__init__()
        
        self.config = config
        self.tokenizer = nn.Linear(config.patch_size,self.config.n_embd)
         
        self.patch_size = config.patch_size
        self.mask0 = nn.Linear(1,config.n_embd)
        
        self.register_buffer('mask_token', torch.zeros(1000))
        if self.config.haar_trans:
            self.register_buffer('haar_transform',torch.Tensor(haarMatrix(self.config.patch_size,normalized = self.config.haar_trans_norm)))

        
    def forward(self,x, 
                future_token = 0, 
                prev_token = 0, 
                factor = 0.2,
                sequential = False,
                *args, **kwargs):
        
        
        b = x.shape[0]
        
        x_raw = rearrange(x, "b (l c) -> b l c", c = self.patch_size)
        x_raw_0 = rearrange(x, "b (l c) -> b l c", c = self.patch_size).detach().clone()
        
        if future_token == 0:
            if not sequential:
                masks = torch.randperm(x_raw.shape[1])
                unmasks,masks = masks[:int(x_raw.shape[1]*factor)],masks[int(x_raw.shape[1]*factor):]
            else:
                masks = [_ for _ in range(x_raw.shape[1])]
                factor = np.random.rand()*0.6 + 0.2
                unmasks,masks = masks[:int(x_raw.shape[1]*factor)],masks[int(x_raw.shape[1]*factor):]
            
            
            
            x_raw_remains = x_raw[:,unmasks,:]
            
            mean = x_raw_remains.mean(dim = (-2,-1),keepdims = True)
            std = x_raw_remains.std(dim = (-2,-1),keepdims = True)
            x_raw = (x_raw - mean)/ (std + 1e-4)
          
            
            if self.config.haar_trans:
                x_featured = torch.einsum('blc,ac->bla',x_raw,self.haar_transform)
                x_featured = self.tokenizer(x_featured)
            else:
                x_featured = self.tokenizer(x_raw)
                
 
            x_featured[:,masks,:] = self.mask0(self.mask_token[0].unsqueeze(0))
            
            
   
        else:
   

            factor = 1
            more_rows = future_token // self.patch_size + 1
            prev_more_rows = prev_token // self.patch_size + 1
        
            mean = x_raw[:,prev_more_rows:-more_rows,:].mean(dim = (-2,-1),keepdims = True)
            std = x_raw[:,prev_more_rows:-more_rows,:].std(dim = (-2,-1),keepdims = True)
            x_raw = (x_raw - mean)/ (std + 1e-4)
            
            
            if self.config.haar_trans:
                x_featured = torch.einsum('blc,ac->bla',x_raw,self.haar_transform)
                x_featured = self.tokenizer(x_featured)
            else:
                x_featured = self.tokenizer(x_raw)
    
    
            masks = [jj for jj in range(x_featured.shape[1])]
            masks = masks[-more_rows:]
    
            x_featured[:,-more_rows:] = self.mask0(self.mask_token[:len(masks)].unsqueeze(-1)).repeat(x_featured.shape[0],1,1)
            x_featured[:,:prev_more_rows] = self.mask0(self.mask_token[:prev_more_rows].unsqueeze(-1)).repeat(x_featured.shape[0],1,1)


        return x_featured, x_raw_0, masks, mean, std, x_raw
        
    
class GPT(nn.Module):
    def __init__(self, config: None, *args,**kwargs) -> None:
        
        super().__init__()
        
        self.config = config
        self.patch_size = config.patch_size
        self.unet = config.unet
        
        
        if config.stats_encoding:
            self.stat_tokens = 1
        else:
            self.stat_tokens = 0
                
        #patch to token
        self.tokenizer = Tokenizer(config)

        #decoder head
        self.lm_head =  nn.Linear(config.n_embd, 99*self.patch_size)
      
        #loss
        self.quantitleLoss = quantitleLoss(99,patch_size = self.patch_size)
        
        # main body
        if self.unet:
            assert config.n_layer%2 == 0
            self.unet_projection = nn.ModuleList(nn.Sequential(nn.Linear(config.n_embd*2,config.n_embd),
                                                            config.norm_class(config.n_embd, eps=config.norm_eps),
                                                            )
                                            for  _ in range(config.n_layer//2)
                                           )
            self.unet_merge = nn.ModuleList(nn.Sequential(nn.Linear(config.n_embd*2,config.n_embd),
                                                            config.norm_class(config.n_embd, eps=config.norm_eps),
                                                            )
                                            for  _ in range(config.n_layer//2)
                                           )
         
        self.transformer = nn.ModuleDict(dict(h = nn.ModuleList(Block(config) 
                                          for _ in range(config.n_layer))
                                             )
                                        )
        
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None    
        self.kv_caches: List[KVCache] = []


    def _init_weights(self, module: nn.Module, n_layer) -> None:
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)     
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, BidirectedlSelfAttention))): 
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)
        
    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self, idx: torch.Tensor, 
        max_seq_length: Optional[int] = None, 
        input_pos: Optional[torch.Tensor] = None,
        next_token: torch.Tensor = None,
        future_token: int = 0,
        prev_token: int = 0,
        val: bool = False,
        print_intermediate: bool = False,
        cot_rounds: int = -1,
        sequential: bool = False,
        *args,**kwargs,
    ) -> torch.Tensor:
        
        # add dummy time point for inference stage.
        if future_token > 0:
            more_rows = future_token // self.patch_size + 1
            idx = torch.cat((idx,torch.zeros(idx.shape[0],more_rows*self.patch_size).to(idx.device)),dim = -1).bfloat16()
            
        B, T = idx.size()

        use_kv_cache = input_pos is not None
        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size    
        if use_kv_cache:  
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)
        cos, sin = self.rope_cache
        if use_kv_cache:         
            if self.stat_tokens:
                if len(input_pos) == 1:
                    idx = idx[:,input_pos]
                    input_pos = input_pos.add_(1)
                else:
                    input_pos = torch.arange(0, input_pos[-1]+2, device=idx.device)
                    
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
                mask = self.mask_cache.index_select(2, input_pos)
                mask = mask[:, :, :, :max_seq_length]
         
            else:
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
                idx = idx[:,input_pos]
        else:
            cos = cos[:max(T,1024) + self.stat_tokens]
            sin = sin[:max(T,1024) +  self.stat_tokens]
            mask = None


                
        # we don't consider using kv-cache.
        if use_kv_cache:
            pass
        else:
            x,x_raw,masks,mean,std,x_0 =  self.tokenizer(idx, 
                                                         future_token =future_token,
                                                         prev_token = prev_token,
                                                         sequential = sequential,
                                                        )
            
               


        if self.unet:
            skips = []
              
        for block_idx in range(len( self.transformer.h)):

            block =  self.transformer.h[block_idx]
            if self.unet and block_idx >=len(self.transformer.h) //2:               
                x = self.unet_projection[block_idx - len(self.transformer.h) //2](torch.cat((skips.pop(),x),dim = -1))
            x, *_ = block(x, (cos, sin), max_seq_length)

            if self.unet and block_idx <len(self.transformer.h) //2:
                skips.append(x)
                x_delay = torch.cat((x[:,0,:].unsqueeze(1),x[:,:-1,:]),dim = 1)
                x = self.unet_merge[block_idx](torch.cat((x_delay,x),dim = -1))
                    
        res = self.lm_head(x)     

        
        # reshape the results for 99 quantitles.   
        res = rearrange(res,'b c (l1 l2) -> b c l1 l2', l2 = 99)
        

        if self.config.haar_trans_inv:
            res = torch.einsum('bcal,ad->bcdl',res,self.tokenizer.haar_transform)
            if self.config.haar_trans_norm == "backward":
                res = res / np.sqrt(res.shape[-2])
            elif self.config.haar_trans_norm == "forward":
                res = res * np.sqrt(res.shape[-2])

                
        res = res * (std.unsqueeze(-1) + 1e-4) + mean.unsqueeze(-1)
        
            
        if future_token == 0:
            return res[:,masks,:,:], x_raw[:,masks,:]
        else:
            return res[:,masks,:,:]


    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size +  self.stat_tokens,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            base = self.config.rope_base,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size+self.stat_tokens, self.config.block_size+self.stat_tokens), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = BidirectedlSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class BidirectedlSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)  
        v = v.reshape(B,  T, -1, self.config.head_size)  

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)
        
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

            

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False
        )
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # print('    print(seq_idx.shape,theta.shape,sin.shape,cos.shape,idx_theta.shape)',seq_idx.shape,theta.shape,sin.shape,cos.shape,idx_theta.shape)
    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
