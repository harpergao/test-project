# trainer_ddp.py

import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import FocalLoss, mIoULoss, mmIoULoss, DiceLoss

from misc.logger_tool import Logger, Timer

from utils import de_norm

from tqdm import tqdm

# DDP --- ADDED: Import necessary libraries for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ----------------- [TENSORBOARD ADDED 1/4] 导入库 -----------------
from torch.utils.tensorboard import SummaryWriter

# multi-scale-train --- ADDED: 余弦退火平滑权重
from models.help_funcs import LossWeightAnnealing, WaveletLoss

class CDTrainer():

    # DDP --- MODIFIED: Add 'rank' as an argument for process identification
    def __init__(self, args, dataloaders, rank):
        self.args = args
        self.dataloaders = dataloaders
        self.rank = rank  # DDP --- ADDED: Store the rank of the current process

        self.n_class = args.n_class
        
        # DDP --- MODIFIED: The device is now specific to the process's rank
        self.device = torch.device(f"cuda:{self.rank}")
        
        # DDP --- MODIFIED: Define model and move it to the correct device *before* wrapping with DDP
        model = define_G(args=args).to(self.rank)
        self.net_G = DDP(model, device_ids=[self.rank], find_unused_parameters=True)

        print(f"Process on rank {self.rank} is using device {self.device}")
        ## test ##
        # if torch.cuda.is_available():
        #     print("CUDA is available. Using GPU:", args.gpu_ids)
        # else:
        #     print("CUDA is not available. Using CPU.")
        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
                                     weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                    betas=(0.9, 0.999), weight_decay=0.01)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # DDP --- MODIFIED: Only the main process (rank 0) should handle files and logging
        if self.rank == 0:
            logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
            self.logger = Logger(logger_path)
            self.logger.write_dict_str(args.__dict__)
            # ----------------- [TENSORBOARD ADDED 2/4] 初始 -----------------
            # 在 checkpoint 目录下创建一个 'runs' 子目录存放 tensorboard 日志
            tb_log_dir = os.path.join(args.checkpoint_dir, 'runs')
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"TensorBoard initialized at: {tb_log_dir}")
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        # DDP --- MODIFIED: The number of steps per epoch is now per process
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.loss1 = None
        self.dice_loss = None
        self.wave_loss = None
        # self.part_loss = [0.0] * 5  # for 4 multi-scale loss recording and 1 cos value.

        self.shuffle_AB = args.shuffle_AB

        # define the loss functions
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        
        self.initial_weights = tuple(args.multi_pred_weights)
        # 权重余弦退火平滑
        # self.final_weights = [0.0, 0.0, 0.0, 0.0, 1.0]
        # self.weight_scheduler = LossWeightAnnealing(
        #     initial_weights=self.initial_weights,
        #     final_weights=self.final_weights,
        #     total_epochs=self.max_num_epochs,
        #     warmup_epochs=50,       # 50轮预热
        #     decay_end_epoch=150     # 在第150轮衰减到0
        # )
        self.weights = self.initial_weights
        
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        # ... (loss function definitions remain the same, just ensure they are moved to the correct device if needed)
        # For example, if a loss function has internal parameters, it needs .to(self.device)
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'fl':
            # print('\n Calculating alpha in Focal-Loss (FL) ...')
            # alpha           = get_alpha(dataloaders['train']) # calculare class occurences
            # print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
            self._pxl_loss  = FocalLoss(alpha=0.75, gamma=2, ignore_index=255)#.to(self.device)
        elif args.loss == "miou":
            # ...
            print('\n Calculating Class occurances in training set...')
            alpha   = np.asarray(get_alpha(dataloaders['train'])) # calculare class occurences
            alpha   = alpha/np.sum(alpha)
            # weights = torch.tensor([1.0, 1.0]).cuda()
            # weights = 1-torch.from_numpy(alpha).cuda()
            weights = 1-torch.from_numpy(alpha).to(self.device)
            print(f"Weights = {weights}")
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()
        elif args.loss == "mmiou":
             self._pxl_loss = mmIoULoss(n_classes=args.n_class).to(self.device)
        # ... (rest of the loss functions)
        else:
            raise NotImplemented(args.loss)
        
        self.waveLoss = WaveletLoss().to(self.device)
        self.dice_loss = DiceLoss().to(self.device)
        
        # DDP --- MODIFIED: Only main process handles file I/O
        if self.rank == 0:
            self.VAL_ACC = np.array([], np.float32)
            if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
                self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
            self.TRAIN_ACC = np.array([], np.float32)
            if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
                self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

            # check and create model dir
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print(f"\n[Rank {self.rank}] Checking for checkpoint...")
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            # DDP --- MODIFIED: Load checkpoint onto the correct device for each process
            # map_location is crucial to prevent process 0 from loading GPU 1's data
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device, weights_only=False)
            
            # DDP --- MODIFIED: Load state dict into the .module attribute
            # The actual model is wrapped inside the DDP object
            self.net_G.module.load_state_dict(checkpoint['model_G_state_dict'], strict=False)

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            # update some other states (all processes need this info)
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            if self.rank == 0:
                self.logger.write('loading last checkpoint...\n')
                self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                      (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
                self.logger.write('\n')
        
        elif self.args.pretrain is not None:
             if self.rank == 0:
                print("Initializing backbone weights from: " + self.args.pretrain)
             self.net_G.module.load_state_dict(torch.load(self.args.pretrain, map_location=self.device), strict=False)
        else:
            if self.rank == 0:
                print('training from scratch...')
        print(f"\n[Rank {self.rank}] Finished checkpoint check.")
    

    def _save_checkpoint(self, ckpt_name):
        # DDP --- MODIFIED: Only rank 0 should save the checkpoint
        if self.rank == 0:
            print(f"Saving checkpoint: {ckpt_name}")
            torch.save({
                'epoch_id': self.epoch_id,
                'best_val_acc': self.best_val_acc,
                'best_epoch_id': self.best_epoch_id,
                # DDP --- MODIFIED: Save the .module's state_dict
                'model_G_state_dict': self.net_G.module.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            }, os.path.join(self.checkpoint_dir, ckpt_name))


    def _collect_running_batch_states(self):
        # (Metric calculation here is per-process. For exact global metrics,
        # you'd need to sync results across GPUs, but for logging this is often sufficient)
        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        # DDP --- MODIFIED: Only rank 0 should log to console/file and save images
        if self.rank == 0:
            imps, est = self._timer_update()
            if np.mod(self.batch_id, 100) == 1:
                message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                          (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                         imps*self.batch_size*int(os.environ['WORLD_SIZE']), est, # DDP: imps reflects total imps
                         self.G_loss.item(), running_acc)
                self.logger.write(message)
                
                # ----------------- [TENSORBOARD ADDED 3a/4] 记录 Loss -----------------
                # 只记录训练阶段的 Loss，或者你也可以分开记录 Val Loss
                tag_prefix = 'Train' if self.is_training else 'Val'
                self.writer.add_scalar(f'Loss/{tag_prefix}_Total', self.G_loss.item(), self.epoch_id)
                self.writer.add_scalar(f'Metric/{tag_prefix}_Running_MF1', running_acc, self.epoch_id)
            if np.mod(self.batch_id, 500) == 1:
                vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
                vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
                vis_pred = utils.make_numpy_grid(self._visualize_pred())
                vis_gt = utils.make_numpy_grid(self.batch['L'])
                vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
                vis = np.clip(vis, a_min=0.0, a_max=1.0)
                file_name = os.path.join(
                    self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                                  str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
                plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        # (Similar to batch states, these scores are per-process.
        cm_local = self.running_metric.sum        
        # local_pixels = cm_local.sum()
        # print(f"[Rank {self.rank}] Local pixels sum: {local_pixels}")
        cm_tensor = torch.from_numpy(cm_local).long().to(self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
        self.running_metric.sum = cm_tensor.cpu().numpy()
        # For precise evaluation, you'd need to gather results before calculating.)
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']

        # DDP --- MODIFIED: Only rank 0 should log
        if self.rank == 0:
            # global_pixels = self.running_metric.sum.sum()
            # print(f"[Rank 0] Global pixels sum (After Reduce): {global_pixels}")
            self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
                  (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
            message = ''
            for k, v in scores.items():
                message += '%s: %.5f ' % (k, v)
            self.logger.write(message+'\n')
            self.logger.write('\n')
            
            # ----------------- [TENSORBOARD ADDED 4/4] 记录 Epoch 指标 -----------------
            tag_prefix = 'Train' if self.is_training else 'Val'
            # 记录 mF1
            self.writer.add_scalar(f'Epoch_Metric/{tag_prefix}_mF1', self.epoch_acc, self.epoch_id)
            # 记录 F1_1
            if 'F1_1' in scores:
                self.writer.add_scalar(f'Epoch_Metric/{tag_prefix}_F1_1', scores['F1_1'], self.epoch_id)
    def _update_training_acc_curve(self):
        # DDP --- MODIFIED: Only rank 0 should save numpy arrays
        if self.rank == 0:
            self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
            np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # DDP --- MODIFIED: Only rank 0 should save numpy arrays
        if self.rank == 0:
            self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
            np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    # All other methods like _forward_pass, _backward_G remain largely the same,
    # as DDP handles the gradient synchronization automatically.
    # ... (No changes needed for _forward_pass, _backward_G, _visualize_pred, _timer_update etc.)
    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est
    def _visualize_pred(self):
        pred = torch.argmax(self.G_final_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_final_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score
    def _update_checkpoints(self):
        if self.rank == 0:
            self._save_checkpoint(ckpt_name='last_ckpt.pt')
            self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
                  % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

            if self.epoch_acc > self.best_val_acc:
                self.best_val_acc = self.epoch_acc
                self.best_epoch_id = self.epoch_id
                self._save_checkpoint(ckpt_name='best_ckpt.pt')
                self.logger.write('*' * 10 + 'Best model updated!\n')
                self.logger.write('\n')
    def _clear_cache(self):
        self.running_metric.clear()
    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)
        if self.multi_scale_infer == "True":
            self.G_final_pred = torch.zeros(self.G_pred[-1].size()).to(self.device)
            for pred in self.G_pred:
                if pred.size(2) != self.G_pred[-1].size(2):
                    self.G_final_pred = self.G_final_pred + F.interpolate(pred, size=self.G_pred[-1].size(2), mode="nearest")
                else:
                    self.G_final_pred = self.G_final_pred + pred
            self.G_final_pred = self.G_final_pred/len(self.G_pred)
        else:
            self.G_final_pred = self.G_pred[-1]
            
    def _backward_G(self):
        # 1. 获取 GT 并转为 float，维度通常为 [B, H, W]
        gt = self.batch['L'].to(self.device).float()

        if self.multi_scale_train == "True":
            i = 0
            temp_loss = 0.0
            # self.weights = self.weight_scheduler.get_weights(self.epoch_id)

            for pred in self.G_pred:
                if pred.size(2) != gt.size(2):
                    temp_loss1 = self.weights[i] * self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2), mode="nearest"))
                    # temp_loss1 = self.weights[i] * (
                    #     self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2), mode="nearest")) +
                    #     0.1 * self.dice_loss(pred, F.interpolate(gt, size=pred.size(2), mode="nearest"))
                    # )
                else:
                    temp_loss1 = self.weights[i] * self._pxl_loss(pred, gt) 
                    # temp_loss1 = self.weights[i] * (
                    #     self._pxl_loss(pred, gt) +
                    #     0.1 * self.dice_loss(pred, gt)
                    # )

                # self.part_loss[i] = temp_loss1.item()
                temp_loss += temp_loss1
                i += 1

            # self.part_loss[4] = self.weights[0] * 5.0  # record the cos weight
            self.loss1 = temp_loss

        else:
            self.loss1 = self._pxl_loss(self.G_pred[-1], gt)
            # self.loss1 = (
            #     self._pxl_loss(self.G_pred[-1], gt)
            #     + 0.1 * self.dice_loss(self.G_pred[-1], gt)
            # )

        self.wave_loss = self.waveLoss(self.G_pred[-1], gt)
        self.G_loss = self.loss1 + 0.1 *self.wave_loss

        self.G_loss.backward()

    def train_models(self):
        self._load_checkpoint()
        # DDP --- ADDED: Make sure all processes have loaded the model before starting training
        dist.barrier()

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            # DDP --- ADDED: Set the epoch for the sampler, crucial for proper shuffling
            self.dataloaders['train'].sampler.set_epoch(self.epoch_id)
            
            # ... (training loop remains the same)
            self._clear_cache()
            self.is_training = True
            self.net_G.train()
            total = len(self.dataloaders['train'])
            if self.rank == 0:
                self.logger.write('lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])
            
            # Use tqdm only on the main process to avoid multiple progress bars
            data_iter = self.dataloaders['train']
            if self.rank == 0:
                data_iter = tqdm(data_iter, total=total)

            for self.batch_id, batch in enumerate(data_iter, 0):
                self._forward_pass(batch)
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
            
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            # Evaluation should also be done in a distributed manner
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()
            
            if self.rank == 0:
                self.logger.write('Begin evaluation...\n')
            
            # self.dataloaders['val'].sampler.set_epoch(self.epoch_id) # Not strictly necessary for validation
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            self._update_val_acc_curve()
            self._update_checkpoints()
            
            # ----------------- [TENSORBOARD ADDED 4/4] 显式关闭 -----------------
        if self.rank == 0:
            self.writer.close()
            


# DDP --- ADDED: Setup function for DDP environment
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# DDP --- ADDED: Cleanup function
def cleanup_ddp():
    dist.destroy_process_group()

# DDP --- ADDED: Main execution block to launch the training
if __name__ == '__main__':
    # Assume 'args' and 'dataloaders' are created here based on your original main script.
    # For example:
    # from options.train_options import TrainOptions
    # from data.cd_dataset import CDDataset
    # from torch.utils.data import DataLoader
    
    # 1. Initialize DDP
    setup_ddp()
    rank = int(os.environ["RANK"])

    # 2. Parse arguments
    # args = TrainOptions().parse()

    # 3. Create datasets and distributed samplers
    # train_dataset = CDDataset(root=args.root, split='train')
    # val_dataset = CDDataset(root=args.root, split='val')
    
    # train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=int(os.environ['WORLD_SIZE']))
    # val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=int(os.environ['WORLD_SIZE']))
    
    # dataloaders = {
    #     'train': DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=True),
    #     'val': DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)
    # }

    # 4. Create trainer and start training
    # This is a placeholder for your actual dataloader creation logic
    # You MUST replace this with your code that prepares `args` and `dataloaders`
    
    # args = ... 
    # dataloaders = ... 
    
    # trainer = CDTrainer(args, dataloaders, rank)
    # trainer.train_models()

    # 5. Cleanup
    
    cleanup_ddp()