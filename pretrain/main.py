import glob
import math
import sys
import time
from pathlib import Path
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import get_default_supported_precision, num_parameters
import numpy as np
from utils import get_config, create_dataloaders, augmentation, consineScheduler
import argparse
    


    
    
    

    
    
def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
        
    model = state["model"]
    optimizer = state["optimizer"]
    Config = model.config

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check


    with torch.device("meta"):
        meta_model = GPT(model.config)
        estimated_flops = estimate_flops(meta_model) * Config.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (Config.micro_batch_size, model.config.block_size))

        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()


    initial_iter = state["iter_num"]
    curr_iter = 0


    get_lr = consineScheduler(Config)

    for  train_data in train_dataloader:
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                if curr_iter % 100 == 0 and fabric.global_rank == 0:
                    print(f'has passed {curr_iter} trained iterations.')
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= Config.max_iters :
            break

        lr = get_lr(state["iter_num"]) if Config.decay_lr  else Config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        train_data_new = augmentation(train_data,model)


        is_accumulating = (state["iter_num"] + 1) % Config.gradient_accumulation_steps  != 0
        loss = 0
            
            
        with fabric.no_backward_sync(model, enabled=is_accumulating):


            logits1,targets = model(train_data_new)
            loss += model.quantitleLoss(logits1, targets.float())
            fabric.backward(loss / Config.gradient_accumulation_steps)


        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=Config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
            
        state["iter_num"] += 1
        
        total_lengths += train_data.size(1)
        
        t1 = time.perf_counter()
        
        if state['iter_num'] % 100 == 0:
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (Config.max_iters  - state['iter_num']) / 3600:.2f} hours. " 
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (Config.max_iters  - state['iter_num']) / 3600 / 24:.2f} days. "
            )

        monitor.on_train_batch_end(
            state["iter_num"] * Config.micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )




        if val_dataloader is not None and not is_accumulating and state["step_count"] % Config.eval_step_interval == 0:

            t0 = time.perf_counter()

            val_loss = validate(fabric, model, val_dataloader,step = state["step_count"])
            
            
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * Config.micro_batch_size * fabric.world_size}, state["step_count"])
            
            fabric.barrier()
        if not is_accumulating and state["step_count"] % Config.save_step_interval  == 0:
            checkpoint_path = Config.out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
            
            
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, step = None) -> torch.Tensor:
    fabric.print("Validating ...")
       
    model.eval()
    loss_func = torch.nn.L1Loss()
    losses = torch.zeros(Config.eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= Config.eval_iters:
            break

        input_ids = val_data[:, 0 : model.config.block_size_val - model.config.patch_size].contiguous()        

        max_0 = torch.max(input_ids,dim = -1,keepdims = True).values
        min_0 = torch.min(input_ids,dim = -1,keepdims = True).values
        
        input_ids = (input_ids - min_0) / (max_0 - min_0 + 1e-4) - 1/2 
        
        logits,targets = model(input_ids)
        loss = loss_func(logits[...,49], targets.float())*2
        losses[k] = loss.item()
        
    out = losses[:k+1].mean()
    model.train()
    return out
        
        
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--config_name', type=str,  default='test0.yaml')
    parser.add_argument('--train_data_dir', type=Path, default='data_processed')
    parser.add_argument('--val_data_dir', type=Path,  default='data_processed')
    parser.add_argument('--resume', default=False, action="store_true")
    parser.add_argument('--seed', type=int, default=3407)
    
    
    args = parser.parse_args()
    
    # (dataset_name, sampling_eate)     
    train_data_config = [
    ('dataset_1',0.15),         
    ('dataset_2',0.01),
    ('dataset_3',0.14), 
    ('dataset_4',0.6),
    #...
    ('dataset_n',0.1),
    ]


    val_data_config = [
        ("vali_data",1.0),
    ]

    
    Config,loggers = get_config(args,train_data_config,val_data_config)

    devices = Config.num_of_devices
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    precision = get_default_supported_precision(training=True, tpu=False)
    tpu = False
    resume = args.resume

    if devices > 1:
        if tpu:
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=loggers)



    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=Config.log_iter_interval)

    if fabric.global_rank == 0:
        Config.out_dir.mkdir(parents=True, exist_ok=True)



    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=Config.micro_batch_size,
        block_size=Config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        config = Config
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(Config.seed0)  
    fabric.print(f"Loading model with {Config.__dict__}")

    t0 = time.perf_counter()

    with fabric.init_module(empty_init=False):
        model = GPT(config = Config)
        model.apply(partial(model._init_weights ,n_layer=Config.n_layer))
        model.config.use_cache = True


    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")



    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay, betas=Config.betas, foreach=False,fused = True)


    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}


    if resume is True:
        resume = sorted(Config.out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()

    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)


    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

