import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict

from .utils import all_reduce_mean, is_main_process, save_sanity_png, slide_reconstruct, barrier
from .data import QuantileScaler

def train_one_epoch(loader, model, criterion, opt, device, use_amp=True, grad_clip=1.0, epoch=1, sampler=None):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type=='cuda')
    total, n = 0.0, 0

    for batch in loader:
        he = batch['he'].to(device, non_blocking=True)
        tgt_log = batch['tgt_log'].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
            out = model(he, return_presence=getattr(model, "has_presence_head", False))
            if isinstance(out, tuple):
                pred_log, presence_logits = out
            else:
                pred_log, presence_logits = out, None
            loss = criterion(pred_log, tgt_log, presence_logits=presence_logits)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()

        bs = he.size(0)
        total += float(loss) * bs
        n += bs

    # average across processes
    t = torch.tensor([total, n], device=device, dtype=torch.float32)
    t = all_reduce_mean(t)
    total_m, n_m = t[0].item(), t[1].item()
    return total_m / max(1.0, n_m)


@torch.no_grad()
def validate_one_epoch(loader, model, criterion, device, use_amp=True, sampler=None):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        he = batch['he'].to(device, non_blocking=True)
        tgt_log = batch['tgt_log'].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
            out = model(he, return_presence=getattr(model, "has_presence_head", False))
            if isinstance(out, tuple):
                pred_log, presence_logits = out
            else:
                pred_log, presence_logits = out, None
            loss = criterion(pred_log, tgt_log, presence_logits=presence_logits)
        bs = he.size(0)
        total += float(loss) * bs
        n += bs

    t = torch.tensor([total, n], device=device, dtype=torch.float32)
    t = all_reduce_mean(t)
    total_m, n_m = t[0].item(), t[1].item()
    return total_m / max(1.0, n_m)

class HexifTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, 
                 train_loader, val_loader, val_ds,
                 device, scaler: QuantileScaler,
                 output_dir: Path,
                 use_amp: bool = True,
                 save_every: int = 10,
                 sanity_every: int = 1):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_ds = val_ds
        self.device = device
        self.scaler = scaler
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.save_every = save_every
        self.sanity_every = sanity_every
        
        # Cache for sanity check
        self.he0, self.or0, self.val_name0 = None, None, None
        if is_main_process() and len(val_ds.basenames) > 0:
            self.he0, self.or0 = val_ds._load_pair(0)
            self.val_name0 = val_ds.basenames[0]

    def train(self, epochs: int):
        best_val = 1e9
        history = []
        has_valid_model = False
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            trn = train_one_epoch(self.train_loader, self.model, self.criterion, self.optimizer, 
                                  self.device, use_amp=self.use_amp, grad_clip=1.0, 
                                  epoch=epoch, sampler=self.train_loader.sampler)
                                  
            val = validate_one_epoch(self.val_loader, self.model, self.criterion, 
                                     self.device, use_amp=self.use_amp, sampler=self.val_loader.sampler)
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val)
            else:
                self.scheduler.step()
                
            dt = time.time() - t0
            
            if is_main_process():
                is_valid = (not np.isnan(val) and not np.isinf(val) and val > 0 and 
                           not np.isnan(trn) and not np.isinf(trn))
                           
                logging.info(f"Epoch {epoch:03d}/{epochs} | train {trn:.4f} | val {val:.4f} | LR {current_lr:.2e} | {dt:.1f}s")
                
                history.append({"epoch": epoch, "train": trn, "val": val, "time_sec": dt})
                
                # CSV logging
                mode = 'a' if epoch > 1 else 'w'
                with open(self.output_dir / "metrics.csv", mode) as f:
                    if epoch == 1:
                        f.write("epoch,split,loss,time_sec\n")
                    f.write(f"{epoch},train,{trn},{dt}\n")
                    f.write(f"{epoch},val,{val},{dt}\n")
                
                if is_valid and val < best_val:
                    if epoch >= 2 or not has_valid_model:
                        best_val = val
                        has_valid_model = True
                        state = self._get_state(epoch, history, best_val=best_val)
                        torch.save(state, self.output_dir / "best_model.pth")
                        logging.info(f"  → New best saved (val={best_val:.4f})")
                        
                if epoch % self.save_every == 0:
                    state = self._get_state(epoch, history)
                    torch.save(state, self.output_dir / f"checkpoint_epoch_{epoch}.pth")
                    
                # Sanity check
                if epoch % self.sanity_every == 0 and self.he0 is not None:
                    self._run_sanity_check(epoch)
                    
        # Final save
        if is_main_process():
            final = self._get_state(epochs, history, best_val=best_val)
            torch.save(final, self.output_dir / "final_model.pth")
            logging.info(f"Training complete. Best val: {best_val:.4f}")

    def _get_state(self, epoch, history, best_val=None):
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model.state_dict()
        state = {
            "epoch": epoch,
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "history": history,
            "scaler": self.scaler.to_dict()
        }
        if best_val is not None:
            state["best_val"] = best_val
        return state

    def _run_sanity_check(self, epoch):
        self.model.eval()
        with torch.no_grad():
            # Handle DDP model unwrapping
            model_module = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            
            pred_log = slide_reconstruct(model_module, self.he0, self.val_ds.tf_eval,
                                         ps=self.val_ds.ps, stride=160, device=self.device)
            tgt_log = self.val_ds._scale_to_log(self.or0).transpose(2,0,1)
            
        save_sanity_png(self.output_dir, self.he0, pred_log, tgt_log, epoch, tag=self.val_name0)
