from argparse import ArgumentParser
import torch 
## 4.10.2025 调 cuDNN 设置
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import os
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
###
import utils
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from models.trainer import *
from models.trainer_ddp import *
import data_config
from datasets.CD_dataset import CDDataset

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


# def train(args):
#     dataloaders = utils.get_loaders(args)
#     model = CDTrainer(args=args, dataloaders=dataloaders)
#     model.train_models()


# def test(args):
#     from models.evaluator import CDEvaluator
#     dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
#                                   batch_size=args.batch_size, is_train=False,
#                                   split='test')
#     model = CDEvaluator(args=args, dataloader=dataloader)

#     model.eval_models()


if __name__ == '__main__':
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    # parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='ChangeFormer', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.2, 0.2, 0.2, 0.4, 1.0])

    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    # utils.get_device(args)
    # print(args.gpu_ids)
    # if rank == 0:
    #     args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    #     os.makedirs(args.checkpoint_dir, exist_ok=True)
    #     args.vis_dir = os.path.join(args.vis_root, args.project_name)
    #     os.makedirs(args.vis_dir, exist_ok=True)
    #     print("Arguments parsed and directories created.")
    # 1. 为 args 添加路径属性，这一步需要在所有进程中都执行
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    args.vis_dir = os.path.join(args.vis_root, args.project_name)

# 2. 创建文件夹的操作，只在主进程 (rank 0) 中执行，避免多进程冲突
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)
        print("Arguments parsed and directories created.")
    # #  checkpoints dir
    # args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    # os.makedirs(args.checkpoint_dir, exist_ok=True)
    # #  visualize dir
    # args.vis_dir = os.path.join(args.vis_root, args.project_name)
    # os.makedirs(args.vis_dir, exist_ok=True)
    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = args.split_val
    
    training_set = CDDataset(root_dir=root_dir, split=split,
                             img_size=args.img_size, is_train=True,
                             label_transform=label_transform)
    val_set = CDDataset(root_dir=root_dir, split=split_val,
                        img_size=args.img_size, is_train=False,
                        label_transform=label_transform)
    
    train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    
    # train(args)

    # test(args)
    dataloaders = {
        'train': DataLoader(training_set,
                            batch_size=args.batch_size,
                            shuffle=False,  # shuffle 必须为 False
                            sampler=train_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True),
        'val': DataLoader(val_set,
                          batch_size=args.batch_size,
                          shuffle=False, # shuffle 必须为 False
                          sampler=val_sampler,
                          num_workers=args.num_workers,
                          pin_memory=True)
    }
    trainer = CDTrainer(args, dataloaders, rank)
    trainer.train_models()
    
    if rank == 0:
        print("Training finished. Starting evaluation on test set...")
        from models.evaluator import CDEvaluator
        
        # 复用你原来的 get_loader 函数来加载 test set
        # 注意：测试时不需要 DDP Sampler，因为只在一个进程上跑
        test_dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                     batch_size=args.batch_size, is_train=False,
                                     split='test')
        evaluator = CDEvaluator(args=args, dataloader=test_dataloader)
        evaluator.eval_models()

    
    cleanup_ddp()
