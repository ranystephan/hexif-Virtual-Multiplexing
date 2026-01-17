import os
import logging
import torch
import torch.distributed
import numpy as np
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt

def ddp_setup(backend: str = "nccl"):
    """Initialize DDP from torchrun env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # single-process fallback
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        torch.distributed.init_process_group(backend=backend, timeout=timedelta(seconds=7200))
        
    return rank, world_size, local_rank

def is_main_process():
    return (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def all_reduce_mean(t: torch.Tensor):
    if not torch.distributed.is_initialized():
        return t
    rt = t.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

def set_seed(seed: int = 42, rank: int = 0):
    import random
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(output_dir: Path, rank: int, log_name: str = "train.log"):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / (log_name if rank == 0 else f"{log_name}_rank{rank}")
    
    # Clear existing handlers
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
        
    level = logging.INFO if rank == 0 else logging.WARNING
    
    handlers = [logging.FileHandler(log_file)]
    if rank == 0:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format=f"%(asctime)s | %(levelname)s | r{rank} | %(message)s",
        handlers=handlers,
    )
    if rank == 0:
        logging.info(f"Logs → {log_file}")

def save_sanity_png(outdir: Path, he_img: np.ndarray, pred_log: np.ndarray, tgt_log: np.ndarray,
                    epoch: int, tag: str = ""):
    cmaps = [
        "viridis", "magma", "plasma", "cividis", "inferno",
        "Greens", "Blues", "Reds", "Purples", "Oranges",
        "Greys", "twilight", "turbo", "coolwarm", "seismic",
        "RdYlBu", "RdYlGn", "Spectral", "jet", "rainbow"
    ]

    # Handle case where prediction has fewer channels than 20
    C = pred_log.shape[0]
    pick = [0,1,2,3,4,5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 
    pick = [p for p in pick if p < C]
    if len(pick) == 0:
        pick = list(range(min(6, C)))

    def inv(x_log):
        lin = np.expm1(x_log)
        return np.clip(lin, 0, 1)

    H, W, _ = he_img.shape
    fig, axes = plt.subplots(nrows=len(pick), ncols=3, figsize=(9, 3*len(pick)))
    
    # Handle single channel case or small number of channels for axes indexing
    if len(pick) == 1:
        axes = np.array([axes])

    for i, c in enumerate(pick):
        axes[i,0].imshow(he_img)
        axes[i,0].set_title("H&E")
        axes[i,0].axis('off')
        
        if tgt_log is not None:
            axes[i,1].imshow(inv(tgt_log[c]), cmap=cmaps[i%len(cmaps)])
            axes[i,1].set_title(f"GT ch{c}")
        else:
            axes[i,1].text(0.5, 0.5, "No GT", ha='center')
            
        axes[i,1].axis('off')
        
        axes[i,2].imshow(inv(pred_log[c]), cmap=cmaps[i%len(cmaps)])
        axes[i,2].set_title(f"Pred ch{c}")
        axes[i,2].axis('off')
        
    fig.suptitle(f"Epoch {epoch} {tag}", y=0.995)
    outpath = outdir / f"sanity_epoch_{epoch:03d}{'_'+tag if tag else ''}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved sanity check → {outpath}")

@torch.no_grad()
def slide_reconstruct(model: torch.nn.Module, he_img: np.ndarray, tf_eval,
                      ps: int, stride: int = 16, device: torch.device = torch.device("cpu")):
    H, W, _ = he_img.shape
    out_accum = None
    weight = None
    
    # Simple sliding window
    for y in range(0, max(1, H - ps) + 1, stride):
        for x in range(0, max(1, W - ps) + 1, stride):
            he_crop = (he_img[y:y+ps, x:x+ps, :]*255).astype(np.uint8)
            # tf_eval expects PIL or uint8 tensor
            he_t = tf_eval(he_crop).unsqueeze(0).to(device)
            
            pred_log = model(he_t).detach().cpu().numpy()[0]  # (C,ps,ps)
            
            if out_accum is None:
                C = pred_log.shape[0]
                out_accum = np.zeros((C, H, W), dtype=np.float32)
                weight = np.zeros((1, H, W), dtype=np.float32)
                
            y2 = min(H, y + ps)
            x2 = min(W, x + ps)
            ph = y2 - y
            pw = x2 - x
            
            out_accum[:, y:y2, x:x2] += pred_log[:, :ph, :pw]
            weight[:, y:y2, x:x2] += 1.0
            
    out_log = out_accum / np.clip(weight, 1e-6, None)
    return out_log
