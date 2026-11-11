"""Exercise the current FreqFormer trainer with temporary synthetic data.

Run from the repository root with torchrun --standalone --nproc_per_node=1
scripts/validate_environment.py (or use 2 processes on a two-GPU machine).
This checks execution, not model accuracy or paper reproduction.
"""

import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from datasets.CD_dataset import CDDataset
from models.trainer_ddp import CDTrainer, cleanup_ddp, setup_ddp


def main():
    setup_ddp()
    trainer = None
    try:
        rank = dist.get_rank()
        if rank != int(os.environ['LOCAL_RANK']):
            raise RuntimeError('This validation supports single-node runs only.')
        torch.set_num_threads(2)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        with tempfile.TemporaryDirectory(prefix='facd-validation-') as work:
            root = Path(work)
            for folder in ('A', 'B', 'label', 'list', 'checkpoint', 'vis'):
                (root / folder).mkdir()
            rng = np.random.default_rng(42 + rank)
            for i in range(2):
                first = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
                second = first.copy()
                second[64:160, 80:176] = 255 - second[64:160, 80:176]
                label = np.zeros((256, 256), dtype=np.uint8)
                label[64:160, 80:176] = 255
                for folder, array in [('A', first), ('B', second), ('label', label)]:
                    Image.fromarray(array).save(root / folder / f'{i}.png')
            (root / 'list/train.txt').write_text('0.png\n1.png\n')
            dataset = CDDataset(str(root), img_size=256, split='train',
                                is_train=True, label_transform='norm')
            loader = DataLoader(dataset, batch_size=1, num_workers=0)
            args = SimpleNamespace(
                n_class=2, net_G='FreqFormer', embed_dim=64,
                optimizer='adamw', lr=1e-4, lr_policy='linear',
                checkpoint_dir=str(root / 'checkpoint'), vis_dir=str(root / 'vis'),
                batch_size=1, max_epochs=1, shuffle_AB='False',
                multi_scale_train='True', multi_scale_infer='False',
                multi_pred_weights=[0.2, 0.2, 0.2, 0.4, 1.0], loss='ce', pretrain=None,
            )
            trainer = CDTrainer(args, {'train': loader, 'val': loader}, rank)
            trainer.net_G.train()
            losses = []
            parameter = next(trainer.net_G.parameters())
            original = parameter.detach().clone()
            for batch in loader:
                trainer.optimizer_G.zero_grad(set_to_none=True)
                trainer._forward_pass(batch)
                expected = [(1, 2, s, s) for s in (8, 16, 32, 64, 256)]
                assert [tuple(p.shape) for p in trainer.G_pred] == expected
                assert all(torch.isfinite(p).all() for p in trainer.G_pred)
                trainer._backward_G()
                assert torch.isfinite(trainer.G_loss)
                grads = [p.grad for p in trainer.net_G.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                trainer.optimizer_G.step()
                losses.append(trainer.G_loss.item())
            assert not torch.equal(parameter.detach(), original), 'Optimizer did not update weights'
            # DDP must leave model parameters identical across ranks.
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            assert torch.equal(reference, parameter.detach())
            trainer.net_G.eval()
            with torch.no_grad():
                trainer._forward_pass(batch)
                assert torch.isfinite(trainer.G_final_pred).all()
                metric = trainer._update_metric()
                assert np.isfinite(metric)
            torch.cuda.synchronize()
            print(json.dumps({
                'status': 'PASS', 'rank': rank, 'world_size': dist.get_world_size(),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(rank), 'losses': losses,
                'output_shapes': expected, 'eval_metric': float(metric),
                'peak_allocated_MiB': round(torch.cuda.max_memory_allocated() / 2**20),
            }), flush=True)
    finally:
        if trainer is not None and dist.get_rank() == 0:
            trainer.writer.close()
        cleanup_ddp()


if __name__ == '__main__':
    main()
