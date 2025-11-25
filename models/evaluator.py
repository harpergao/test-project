import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
# for softmax
import torch.nn.functional as F  
# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        # self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        # self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                #    else "cpu")
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_G = define_G(args=args, gpu_ids=[])      
        self.net_G.to(self.device)
        
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device, weights_only=False)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'], strict=False)

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()
# ==============================================================
    #  修改点：新增 TTA 预测核心函数
    # ==============================================================
    def _predict_with_tta(self, img1, img2):
        """
        执行 TTA 策略：原图 + 水平翻 + 垂直翻 + 旋转90 (共4图投票)
        """
        self.net_G.eval()
        preds = []

        # --- 1. 原图预测 ---
        out = self.net_G(img1, img2)[-1] # 注意保留你的 [-1] 取最后一个输出
        prob = F.softmax(out, dim=1)     # 转成概率分布
        preds.append(prob)

        # --- 2. 水平翻转 (Horizontal Flip) ---
        # 变换输入
        img1_h = torch.flip(img1, dims=[3]) 
        img2_h = torch.flip(img2, dims=[3])
        
        out_h = self.net_G(img1_h, img2_h)[-1]
        prob_h = F.softmax(out_h, dim=1)
        
        # 变换回原方向
        preds.append(torch.flip(prob_h, dims=[3]))

        # --- 3. 垂直翻转 (Vertical Flip) ---
        img1_v = torch.flip(img1, dims=[2])
        img2_v = torch.flip(img2, dims=[2])
        
        out_v = self.net_G(img1_v, img2_v)[-1]
        prob_v = F.softmax(out_v, dim=1)
        
        preds.append(torch.flip(prob_v, dims=[2]))

        # --- 4. 旋转 90度 (Rotate 90) ---
        # dim=[2,3] 是 H,W 维度
        img1_r = torch.rot90(img1, k=1, dims=[2, 3])
        img2_r = torch.rot90(img2, k=1, dims=[2, 3])
        
        out_r = self.net_G(img1_r, img2_r)[-1]
        prob_r = F.softmax(out_r, dim=1)
        
        # 逆时针转回去 (k=-1)
        preds.append(torch.rot90(prob_r, k=-1, dims=[2, 3]))

        # --- 5. 求平均 ---
        # stack 起来取 mean
        final_prob = torch.stack(preds).mean(dim=0)
        return final_prob

    # ==============================================================
    #  修改点：替换原有的 _forward_pass
    # ==============================================================
    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        
        # 原代码: self.G_pred = self.net_G(img_in1, img_in2)[-1]
        
        # 新代码: 使用 TTA
        self.G_pred = self._predict_with_tta(img_in1, img_in2)
    # def _forward_pass(self, batch):
    #     self.batch = batch
    #     img_in1 = batch['A'].to(self.device)
    #     img_in2 = batch['B'].to(self.device)
    #     self.G_pred = self.net_G(img_in1, img_in2)[-1]

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
