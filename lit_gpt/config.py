
from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple


@dataclass
class Config:
    org: str = ""
    name: str = ""
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm","BatchNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1
    rope_base: int = 100000
    group: str= f"120M-aug-m-wql-md-1-nd-ou"
    patch_size:int =  1
    rollback_win:int = 256
    scaling:bool =  True
    quantitle:bool = True
    sum_divided:bool =  False
    unet:bool =  True
    ou:bool =  True
    stats_encoding:bool=  True


    num_of_devices:int = 8
    num_of_nodes:int = 1
    global_batch_size:int = 512
    micro_batch_size:int = 64

    max_step:int = 715256 * 2
    warmup_steps:int = 2000
    learning_rate:float = 5e-4
    weight_decay:float = 1e-1
    betas:tuple[int] = (0.9,0.95)
    grad_clip:float = 1.0
    min_lr:float = 1e-5


    log_step_interval:int = 10
    eval_iters:int = 100
    save_step_interval:int = 5000
    eval_step_interval:int = 1000


    seed0:int = 3407

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd
        self.rope_n_elem = int(self.rotary_percentage * self.head_size)
        self.rope_condense_ratio = self.condense_ratio
        
        
    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        

        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm
            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm
            return FusedRMSNorm
        elif self._norm_class == 'BatchNorm':
            from lit_gpt.inverted_batchnorm import iBatchNorm
            return iBatchNorm
        
        return getattr(torch.nn, self._norm_class)
 