import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

#Focal Loss
def get_alpha(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha

# for FocalLoss
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=1, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
	
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
            alpha = 1/alpha # inverse of class frequency
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
	
        # to resolve error in idx in scatter_
        idx[idx==225]=0
        
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


#miou loss
from torch.autograd import Variable
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

#Minimax iou
class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        #minimum iou of two classes
        min_iou = torch.min(iou)

        #loss
        loss = -min_iou-torch.mean(iou)
        return loss


class ImageRestorationLoss(nn.Module):
    """
    复现 "Harmony in diversity" (CCNet) 论文中的图像恢复损失 (L_rst)。
    该损失计算恢复图像与原始输入图像之间的L1距离。
    """
    def __init__(self):
        super(ImageRestorationLoss, self).__init__()
        # 使用L1 Loss，计算像素级的平均绝对误差
        self.l1_loss = nn.L1Loss()

    def forward(self, restored_image, original_image):
        """
        前向传播。
        :param restored_image: 解码器恢复出的图像张量。
        :param original_image: 原始的输入图像张量。
        :return: L1损失值。
        """
        return self.l1_loss(restored_image, original_image)

def _channel_wise_correlation_matrix(features):
    """
    计算一个特征张量在通道维度上的相关系数矩阵。
    :param features: 输入特征张量，形状为 (B, C, H, W)。
    :return: 相关系数矩阵，形状为 (B, C, C)。
    """
    B, C, H, W = features.shape
    # 将空间维度展平
    features_flat = features.view(B, C, -1) # (B, C, H*W)
    
    # 计算每个通道的均值
    mean = torch.mean(features_flat, dim=2, keepdim=True) # (B, C, 1)
    
    # 中心化
    features_centered = features_flat - mean # (B, C, H*W)
    
    # 计算协方差矩阵
    # (X^T * X) / (n-1)
    # 这里我们用 (X * X^T) / (n-1) 来得到通道间的协方差
    # (B, C, H*W) * (B, H*W, C) -> (B, C, C)
    covariance_matrix = torch.bmm(features_centered, features_centered.transpose(1, 2)) / (H * W - 1)
    
    # 计算每个通道的标准差
    std_dev = torch.std(features_flat, dim=2, keepdim=True) # (B, C, 1)
    
    # 计算标准差的外积，用于归一化协方差矩阵
    # (B, C, 1) * (B, 1, C) -> (B, C, C)
    std_dev_prod = torch.bmm(std_dev, std_dev.transpose(1, 2))
    
    # 防止除以零
    std_dev_prod[std_dev_prod == 0] = 1e-8
    
    # 计算相关系数矩阵
    correlation_matrix = covariance_matrix / std_dev_prod
    
    return correlation_matrix

class FeatureSeparationLoss(nn.Module):
    """
    修正后的特征分离损失。
    确保输入的内容特征和风格特征有相同的通道数 C。
    """
    def __init__(self):
        super(FeatureSeparationLoss, self).__init__()

    def forward(self, content_features, style_features):
        """
        Args:
            content_features (torch.Tensor): 内容特征图, 形状 (B, C, H, W)
            style_features (torch.Tensor): 风格特征向量, 形状 (B, C, 1, 1)
        """
        # 验证通道数是否一致，这是先决条件
        if content_features.shape[1] != style_features.shape[1]:
            raise ValueError(f"内容特征通道数 ({content_features.shape[1]}) 与 "
                             f"风格特征通道数 ({style_features.shape[1]}) 必须一致!")

        B, C, H, W = content_features.shape
        
        # 将风格特征广播到与内容特征相同的空间维度
        style_features_expanded = style_features.expand_as(content_features)
        
        # 展平空间维度，得到 (B, C, H*W)
        content_flat = content_features.view(B, C, -1)
        style_flat = style_features_expanded.view(B, C, -1)
        
        # 计算内容和风格特征之间的相关性矩阵
        # torch.bmm 是批处理矩阵乘法
        # (B, C, H*W) @ (B, H*W, C) -> 得到一个形状为 (B, C, C) 的相关性矩阵
        correlation_matrix = torch.bmm(content_flat, style_flat.transpose(1, 2))
        
        # 计算相关性矩阵的弗罗贝尼乌斯范数的平方。
        # 这个值越小，代表C个内容通道和C个风格通道之间的相关性越低。
        # 我们对范数取平方是为了让惩罚更强，且与L2损失形式上更一致。
        frobenius_norm_sq = torch.sum(torch.pow(correlation_matrix, 2))
        
        # 根据批次大小和维度进行归一化，防止损失值过大
        loss = frobenius_norm_sq / (B * C * C)
        
        return loss
    
    
def sliced_wasserstein_distance(p1, p2, num_projections=50):
    """
    计算两组高维特征点之间的切片瓦瑟斯坦距离(Sliced Wasserstein Distance)。
    :param p1: 第一组特征点，形状为 (n, d)，n为点的数量，d为特征维度。
    :param p2: 第二组特征点，形状为 (m, d)，m为点的数量，d为特征维度。
    :param num_projections: 随机投影的数量。
    :return: 计算出的SWD标量值。
    """
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return torch.tensor(0.0, device=p1.device)
        
    feature_dim = p1.shape[1]
    
    # 1. 生成随机投影方向
    projections = torch.randn(feature_dim, num_projections, device=p1.device)
    projections = F.normalize(projections, dim=0) # (d, num_projections)

    # 2. 将特征点投影到一维
    projected_p1 = torch.matmul(p1, projections) # (n, num_projections)
    projected_p2 = torch.matmul(p2, projections) # (m, num_projections)

    # 3. 对投影后的一维分布进行排序
    sorted_p1, _ = torch.sort(projected_p1, dim=0)
    sorted_p2, _ = torch.sort(projected_p2, dim=0)
    
    # 4. 计算排序后的一维分布之间的L1距离 (这是1D瓦瑟斯坦距离的直接计算)
    # 为保证长度一致，我们将较短的分布插值到较长的长度
    if sorted_p1.shape[0] != sorted_p2.shape[0]:
        max_len = max(sorted_p1.shape[0], sorted_p2.shape[0])
        sorted_p1 = F.interpolate(sorted_p1.unsqueeze(0).unsqueeze(0), size=(max_len, num_projections), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        sorted_p2 = F.interpolate(sorted_p2.unsqueeze(0).unsqueeze(0), size=(max_len, num_projections), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    wasserstein_distance = torch.mean(torch.abs(sorted_p1 - sorted_p2))
    
    return wasserstein_distance


class ContentSimilarityLoss(nn.Module):
    """
    复现 "Harmony in diversity" (CCNet) 论文中的内容相似度损失 (L_sim)。
    该损失计算两个时相在未变化区域的内容特征之间的切片瓦瑟斯坦距离。
    """
    def __init__(self, num_projections=50):
        super(ContentSimilarityLoss, self).__init__()
        self.num_projections = num_projections

    def forward(self, content_features1, content_features2, gt_change_map):
        """
        前向传播。
        :param content_features1: T1时刻的内容特征图 (B, C, H, W)。
        :param content_features2: T2时刻的内容特征图 (B, C, H, W)。
        :param gt_change_map: 真实变化图标签 (B, 1, H_orig, W_orig)，1为变化，0为不变。
        :return: 内容相似度损失值。
        """
        B, C, H, W = content_features1.shape
        
        # 1. 将GT标签图下采样到与特征图相同的空间尺寸
        # 使用'nearest'模式以避免在标签中产生模糊的中间值
        gt_mask = F.interpolate(gt_change_map.float(), size=(H, W), mode='nearest')

        # 2. 创建"未变化区域"的掩码 (unchanged_mask)
        # gt_mask中0代表不变，所以我们直接使用 (1.0 - gt_mask)
        unchanged_mask = 1.0 - gt_mask # (B, 1, H, W)
        
        # 将特征图和掩码展平以便处理
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        p1_flat = content_features1.view(B, C, -1).permute(0, 2, 1)
        p2_flat = content_features2.view(B, C, -1).permute(0, 2, 1)
        mask_flat = unchanged_mask.view(B, -1) # (B, H*W)
        
        total_loss = 0.0
        # 3. 逐个样本计算损失 (因为每个样本的未变化像素数量不同)
        for i in range(B):
            # 找到当前样本中所有未变化像素的索引
            unchanged_indices = torch.where(mask_flat[i] > 0.5)[0]
            
            if len(unchanged_indices) == 0:
                continue # 如果没有未变化像素，则跳过
            
            # 提取未变化区域的特征向量
            unchanged_p1 = p1_flat[i, unchanged_indices, :]
            unchanged_p2 = p2_flat[i, unchanged_indices, :]
            
            # 4. 计算这两组特征之间的切片瓦瑟斯坦距离
            loss_sample = sliced_wasserstein_distance(unchanged_p1, unchanged_p2, self.num_projections)
            total_loss += loss_sample
            
        # 返回批次的平均损失
        return total_loss / B
