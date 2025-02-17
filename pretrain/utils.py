# -*- coding: utf-8 -*-

import glob
from pathlib import Path
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.utils import step_csv_logger
from pytorch_lightning.loggers import WandbLogger
import random
from dataclasses import make_dataclass, asdict
from transformers import PretrainedConfig
import ruamel.yaml 
    
    
    
def get_config(args = None,train_data_config = None,val_data_config = None):

 
    curr_folder = Path(__file__).parents[1]
        
    if isinstance(args,str):
        filename = args
    else:
        filename = args.config_name

    with open(    curr_folder /f'configs/{filename}', 'r') as stream:
        data_loaded = ruamel.yaml .safe_load(stream)

    Config = make_dataclass(
        'data', ((k, type(v)) for k, v in data_loaded.items())
        )(**data_loaded)

    assert Config.n_embd % Config.n_head == 0
    assert Config.n_head % Config.n_query_groups == 0

    Config.head_size = Config.n_embd //  Config.n_head 
    Config.rope_n_elem = int(Config.rotary_percentage * Config.head_size)
    Config.rope_condense_ratio = Config.condense_ratio

    Config.out_dir = Path(f"{curr_folder}/out") / Config.name
    Config.batch_size = Config.global_batch_size// (Config.num_of_devices *  Config.num_of_nodes)
    Config.gradient_accumulation_steps = Config.batch_size  // Config.micro_batch_size

    assert Config.gradient_accumulation_steps > 0

    Config.warmup_iters = Config.warmup_steps  * Config.gradient_accumulation_steps

    Config.max_iters = Config.max_step * Config.gradient_accumulation_steps

    Config.log_iter_interval = Config.log_step_interval * Config.gradient_accumulation_steps

    
    if isinstance(args,str):
        Config.seed0 = 3407
    else:
        Config.seed0 = args.seed

    
    if Config._norm_class == "RMSNorm":
        from lit_gpt.rmsnorm import RMSNorm

        Config.norm_class =  RMSNorm
    elif Config._norm_class == "FusedRMSNorm":
        from lit_gpt.rmsnorm import FusedRMSNorm
        Config.norm_class =  FusedRMSNorm

    elif Config._norm_class == 'BatchNorm':
        from lit_gpt.inverted_batchnorm import iBatchNorm
        Config.norm_class =  iBatchNorm

    if Config._mlp_class == "GptNeoxMLP":
        from lit_gpt.model import GptNeoxMLP

        Config.mlp_class =  GptNeoxMLP
    elif Config._mlp_class == "LLaMAMLP":
        from lit_gpt.model import LLaMAMLP
        Config.mlp_class =  LLaMAMLP

        
    from ast import literal_eval
    
    Config.betas = literal_eval(Config.betas)
    
    
    
    Config.train_data_config = train_data_config
    Config.val_data_config = val_data_config

    logger = step_csv_logger(f"{curr_folder}/out", Config.name, flush_logs_every_n_steps=Config.log_iter_interval)
    
    wandb_config2log = asdict(Config)
    wandb_config2log['train_data_config'] = train_data_config
    wandb_config2log['val_data_config'] = val_data_config
    
    
    wandb_logger = WandbLogger(group =Config.group,name = Config.name, config = wandb_config2log)

  
    return Config, [logger,wandb_logger]





class HFConfig(PretrainedConfig):

    def __init__(self,args = 'test-tiny.yaml'):
        super().__init__()
        print('args\n',args,'args\n')
        curr_folder = Path(__file__).parents[1]

        if isinstance(args,str):
            filename = args
        else:
            filename = args.config_name

        with open(    curr_folder /f'configs/{filename}', 'r') as stream:
            data_loaded = ruamel.yaml .safe_load(stream)

        self.Config = make_dataclass(
            'data', ((k, type(v)) for k, v in data_loaded.items())
            )(**data_loaded)
        assert self.Config.n_embd % self.Config.n_head == 0
        assert self.Config.n_head % self.Config.n_query_groups == 0
        
        self.n_embd = self.Config.n_embd
        self.n_head = self.Config.n_head
        
        self.n_query_groups = self.Config.n_query_groups
        
        

        self.Config.head_size = self.Config.n_embd //  self.Config.n_head 
        self.head_size = self.Config.head_size
        
        self.Config.rope_n_elem = int(self.Config.rotary_percentage * self.Config.head_size)
        self.rope_n_elem = self.Config.rope_n_elem
        self.rotary_percentage = self.Config.rotary_percentage
        
        
        self.Config.rope_condense_ratio = self.Config.condense_ratio
        self.condense_ratio = self.Config.condense_ratio
        self.rope_condense_ratio = self.Config.rope_condense_ratio

        if isinstance(args,str):
            self.Config.seed0 = 3407
        else:
            self.Config.seed0 = args.seed



        self.norm_class = self.Config._norm_class
        self.mlp_class = self.Config._mlp_class
        
        self.padded_vocab_size = self.Config.padded_vocab_size
        self.block_size  = self.Config.block_size 
        self.n_layer = self.Config.n_layer
        self.n_embd = self.Config.n_embd
        self.intermediate_size = self.Config.intermediate_size
        self.n_query_groups = self.Config.n_query_groups
        self.parallel_residual = self.Config.parallel_residual
        self.bias = self.Config.bias
        self.norm_eps = self.Config.norm_eps
        self.shared_attention_norm = self.Config.shared_attention_norm
        self.rope_base = self.Config.rope_base
        self.model_type = 'casual'
        
        del self.Config
        




def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, split="train",config= None,n_chunks = 1, num_workers = 1
) -> DataLoader:
    datasets = []
    data_config = config.train_data_config if split == "train" else config.val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))

        random.seed(config.seed0)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            n_chunks=n_chunks,
            block_size=block_size,
            shuffle=shuffle,
            seed=config.seed0+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=config.seed0, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers = num_workers)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    config = None,
    *args,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:

    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        split="train",
        config = config,
        n_chunks = 2,
        num_workers = 8,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=config.micro_batch_size_val,
            block_size=config.block_size_val,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=True,
            split="validation",
            config = config,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader







class consineScheduler():
    def __init__(self,config):
        self.warmup_iters = config.warmup_iters
        self.lr_decay_iters = config.max_iters
        self.min_lr = config.min_lr
        self.learning_rate = config.learning_rate
        
    def __call__(self,iteration):
        
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / self.warmup_iters
        
        if iteration > self.lr_decay_iters:
            return self.min_lr
        

        decay_ratio = (iteration - self.warmup_iters ) / (self.lr_decay_iters - self.warmup_iters)
        
        assert 0 <= decay_ratio <= 1
        
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  
        
        return self.min_lr  + coeff * (self.learning_rate- self.min_lr)
        

     
        
def augmentation(train_data: torch.Tensor, model = None) -> torch.Tensor:
    
    
    # mixup
    augmentOrder = torch.randperm(train_data.shape[0])
    augmentStrength = torch.rand((train_data.shape[0],1))*0.2
    augmentStrength = augmentStrength.to(train_data.device)
    train_data_new = train_data * (1- augmentStrength) + augmentStrength * train_data[augmentOrder,:]
    train_data_new = train_data_new[:,:model.config.block_size - model.config.patch_size]

    # scaling and crop
    if np.random.randint(4, size=1)>=0:

        scalingfactor = np.random.randint(4, size=1) + 1
        if scalingfactor >1:
            train_data_new = torch.nn.functional.interpolate(train_data_new.unsqueeze(1), size=None, scale_factor=scalingfactor[0], mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False).squeeze(1)

    if train_data_new.shape[-1] > model.config.block_size:
        start_point = np.random.choice(train_data_new.shape[-1] - model.config.block_size)
        train_data_new = train_data_new[:,start_point:start_point + model.config.block_size - model.config.patch_size]
        
    
    # trends inject
    trends_injection = torch.arange(-1, 1.02, 1/train_data_new.shape[-1]).to(train_data.device).bfloat16().contiguous()
    trends_injection = trends_injection[:train_data_new.shape[-1]].repeat(train_data_new.shape[0],1)
    starts = torch.rand(train_data_new.shape[0],1).to(train_data.device).bfloat16().contiguous()*2 - 1
    ends = torch.rand(train_data_new.shape[0],1).to(train_data.device).bfloat16().contiguous()*2 - 1
    strengths =  torch.rand(train_data_new.shape[0],1).to(train_data.device).bfloat16().contiguous()*0.9
    trends_injection = trends_injection *(ends - starts)/2 + starts    
    train_data_new = train_data_new * (1-strengths) + trends_injection * strengths

    return train_data_new


   