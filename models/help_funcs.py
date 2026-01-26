import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import numpy as np

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            #(1, 0): 表示在最后一个维度的左边（开头）填充1个元素，右边（末尾）填充0个元素；value = True: 填充的值为 True
            mask = F.pad(mask.flatten(1), (1, 0), value = True) 
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale #einsum 会将两个输入中相同且不在输出中出现的字母维度（这里是 d）进行相乘并求和
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:    # attn取layers[0], ff取layers[1],即attn = Residual(PreNorm(Attention(...)))，ff = Residual(PreNorm(FeedForward(...)))
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x

from scipy.io import savemat
def save_to_mat(x1, x2, fx1, fx2, cp, file_name):
    #Save to mat files
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()
        
        fx1_0_np = fx1[0].detach().cpu().numpy()
        fx2_0_np = fx2[0].detach().cpu().numpy()
        fx1_1_np = fx1[1].detach().cpu().numpy()
        fx2_1_np = fx2[1].detach().cpu().numpy()
        fx1_2_np = fx1[2].detach().cpu().numpy()
        fx2_2_np = fx2[2].detach().cpu().numpy()
        fx1_3_np = fx1[3].detach().cpu().numpy()
        fx2_3_np = fx2[3].detach().cpu().numpy()
        fx1_4_np = fx1[4].detach().cpu().numpy()
        fx2_4_np = fx2[4].detach().cpu().numpy()
        
        cp_np = cp[-1].detach().cpu().numpy()

        mdic = {'x1': x1_np, 'x2': x2_np, 
                'fx1_0': fx1_0_np, 'fx1_1': fx1_1_np, 'fx1_2': fx1_2_np, 'fx1_3': fx1_3_np, 'fx1_4': fx1_4_np,
                'fx2_0': fx2_0_np, 'fx2_1': fx2_1_np, 'fx2_2': fx2_2_np, 'fx2_3': fx2_3_np, 'fx2_4': fx2_4_np,
                "final_pred": cp_np}
                
        savemat("/media/lidan/ssd2/ChangeFormer/vis/mat/"+file_name+".mat", mdic)


class SpatiotemporalAttentionFull(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """
        super(SpatiotemporalAttentionFull, self).__init__()
        assert dimension in [2,]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
                         )
                         
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0),
            )
        self.energy_time_1_sf = nn.Softmax(dim=-1)
        self.energy_time_2_sf = nn.Softmax(dim=-1)
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)
        
    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w) 
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        
        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = self.energy_time_1_sf(energy_time_1) 
        energy_time_2s = self.energy_time_2_sf(energy_time_2) 
        energy_space_2s = self.energy_space_2s_sf(energy_space_1) 
        energy_space_1s = self.energy_space_1s_sf(energy_space_2) 

        # energy_time_2s*g_x11*energy_space_2s = C2*S(C1) × C1*H1W1 × S(H1W1)*H2W2 = (C2*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()#C2*H2W2
        # energy_time_1s*g_x12*energy_space_1s = C1*S(C2) × C2*H2W2 × S(H2W2)*H1W1 = (C1*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)

class LossWeightAnnealing:
    def __init__(self, initial_weights, final_weights, total_epochs, warmup_epochs=50, decay_end_epoch=150):
        """
        初始化权重衰减器
        :param initial_weights: 列表, 初始权重 (e.g., [0.2, 0.2, 0.4, 1.0])
        :param final_weights: 列表, 最终权重 (e.g., [0.0, 0.0, 0.0, 1.0])
        :param total_epochs: 总训练轮数 (e.g., 200)
        :param warmup_epochs: 预热轮数，期间权重保持不变 (e.g., 50)
        :param decay_end_epoch: 衰减结束轮数 (e.g., 150)
        """
        self.initial_weights = np.array(initial_weights)
        self.final_weights = np.array(final_weights)
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.decay_start_epoch = warmup_epochs
        self.decay_end_epoch = decay_end_epoch
        self.decay_epochs = self.decay_end_epoch - self.decay_start_epoch
        
        # 确保最后一个权重（主损失）不被衰减
        # if self.initial_weights[-1] == self.final_weights[-1]:
        #     print(f"主损失权重将保持在 {self.initial_weights[-1]}")
        
        # print(f"权重衰减策略: [0-{self.warmup_epochs}] 保持初始权重。")
        # print(f"[{self.warmup_epochs}-{self.decay_end_epoch}] 余弦衰减至最终权重。")
        # print(f"[{self.decay_end_epoch}-{self.total_epochs}] 保持最终权重。")


    def get_weights(self, current_epoch):
        """
        根据当前 epoch 获取动态权重
        :param current_epoch: 当前的 epoch (从 0 开始)
        """
        if current_epoch < self.decay_start_epoch:
            # 1. 预热阶段
            return self.initial_weights.tolist()
        
        elif current_epoch >= self.decay_end_epoch:
            # 3. 精调阶段
            return self.final_weights.tolist()
            
        else:
            # 2. 衰减阶段 (使用余弦衰减)
            # 计算衰减进度 (0.0 -> 1.0)
            progress = (current_epoch - self.decay_start_epoch) / self.decay_epochs
            
            # 余弦插值: 0.5 * (1 + cos(pi * progress)) 会从 1.0 平滑降到 0.0
            # 我们需要的是从 initial_weight 降到 final_weight
            cosine_scale = 0.5 * (1 + np.cos(np.pi * progress)) # (1.0 -> 0.0)
            
            # 权重 = 最终 + (初始 - 最终) * 缩放
            # (当 scale=1, 权重=初始; 当 scale=0, 权重=最终)
            current_weights = self.final_weights + (self.initial_weights - self.final_weights) * cosine_scale
            
            return current_weights.tolist()

# 有时侯需要多次使用 HINet 模块
class HINet_Module(nn.Module):
    """
    半实例归一化 (Half-Instance Normalization) 模块。
    """
    def __init__(self, in_channels):
        # 假设输入通道为 C，模块内部扩展通道为 2C
        super().__init__()
        self.conv_expand = nn.Conv2d(in_channels, 2*in_channels, kernel_size=1)
        self.instance_norm = nn.InstanceNorm2d(in_channels)
        self.conv_3x3 = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_residual = nn.Conv2d(2*in_channels, in_channels, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # F_mid 尺寸: (B, 2C, H, W)
        F_mid = self.conv_expand(x)
        F1, F2 = torch.chunk(F_mid, 2, dim=1)
        # F1 (控制 "style"): 通过实例归一化
        F1_norm = self.instance_norm(F1)
        # F2 (保留 "content"): 保持不变
        F2_idnet = F2
        F_r = torch.cat([F1_norm, F2_idnet], dim=1)
        
        path1 = self.leaky_relu(F_r)
        path1 = self.conv_3x3(path1)
        path1 = self.leaky_relu(path1)
        
        path2 = self.conv_residual(F_mid)
        # 主路径 残差路径两个路径相加
        F_out = path1 + path2
        
        return F_out

class PFA_Module(nn.Module):
    def __init__(self, in_channels, reduction_channels=None):
        super(PFA_Module, self).__init__()
        

        if reduction_channels is None:
            reduction_channels = in_channels // 2 if in_channels > 4 else in_channels
            
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2, reduction_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(reduction_channels, reduction_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(reduction_channels, 2, kernel_size=3, padding=1, bias=True)
        )

        self.offset_predictor[-1].bias.data.zero_()

    def forward(self, f_t1_norm, f_t2_norm):
        B, C, H, W = f_t2_norm.shape
        

        f_combined = torch.cat([f_t1_norm, f_t2_norm], dim=1)
        offsets = self.offset_predictor(f_combined) 
        
        offsets = torch.tanh(offsets) * 0.5

        x_coords = torch.linspace(-1.0, 1.0, W, device=f_t2_norm.device, dtype=f_t2_norm.dtype)
        y_coords = torch.linspace(-1.0, 1.0, H, device=f_t2_norm.device, dtype=f_t2_norm.dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # base_grid 形状 (H, W, 2)，顺序为 (x, y)
        base_grid = torch.stack([x_grid, y_grid], dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)
        
        # (B, 2, H, W) -> (B, H, W, 2)
        offsets_permuted = offsets.permute(0, 2, 3, 1)
        
        final_grid = base_grid + offsets_permuted
         
        f_t2_aligned = F.grid_sample(
            f_t2_norm, 
            final_grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )
        
        return f_t2_aligned
    
class HINET(nn.Module):
    def __init_(self, in_channels):
        super(HINET, self).__init__()
        self.hi_T1 = HINet_Module(in_channels)
        self.hi_T2 = HINet_Module(in_channels)
        
        # PFA 模块
        # self.pfa = PFA_Module(in_channels)
    
    def forward(self, f_T1, f_T2):
        # 半实例归一化
        f_T1_norm = self.hi_T1(f_T1)
        f_T2_norm = self.hi_T2(f_T2)
        
        # PFA 对齐 (可选)
        # f_T2_aligned = self.pfa(f_T1_norm, f_T2_norm)
        
        return f_T1_norm, f_T2_norm

# from "Pan-Sharpening With Wavelet-Enhanced High-Frequency Information"
class DWT(nn.Module):
    """
    基于 winet.py  的离散小波变换（Haar）实现。
    输入: (B, C, H, W)
    输出: (B, C, H/2, W/2), (B, C, 3, H/2, W/2)
          (LL, [LH, HL, HH])
    """
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False # DWT 滤波器是固定的

    def forward(self, x):
        return self.dwt_init(x)

    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01 + x02
        x2 = x01 - x02
        
        x11 = x1[:, :, :, 0::2]
        x12 = x1[:, :, :, 1::2]
        x21 = x2[:, :, :, 0::2]
        x22 = x2[:, :, :, 1::2]
        
        x_LL = x11 + x12
        x_LH = x11 - x12
        x_HL = x21 + x22
        x_HH = x21 - x22
        
        # 将 (LH, HL, HH) 组合到一个张量中
        x_HFs = torch.cat((x_LH.unsqueeze(2), x_HL.unsqueeze(2), x_HH.unsqueeze(2)), dim=2)
        
        return x_LL, x_HFs

class IWT(nn.Module):
    """
    基于 winet.py  的逆离散小波变换（Haar）实现。
    输入: (B, C, H/2, W/2), (B, C, 3, H/2, W/2)
          (LL, [LH, HL, HH])
    输出: (B, C, H, W)
    """
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_HFs):
        # 从 HFs 张量中分离 LH, HL, HH
        x_LH = x_HFs[:, :, 0, :, :]
        x_HL = x_HFs[:, :, 1, :, :]
        x_HH = x_HFs[:, :, 2, :, :]
        return self.iwt_init(x_LL, x_LH, x_HL, x_HH)

    def iwt_init(self, x_LL, x_LH, x_HL, x_HH):
        x11 = (x_LL + x_LH) / 2
        x12 = (x_LL - x_LH) / 2
        x21 = (x_HL + x_HH) / 2
        x22 = (x_HL - x_HH) / 2
        
        x1 = torch.zeros_like(x_LL).repeat(1, 1, 1, 2)
        x1[:, :, :, 0::2] = x11
        x1[:, :, :, 1::2] = x12
        
        x2 = torch.zeros_like(x_LL).repeat(1, 1, 1, 2)
        x2[:, :, :, 0::2] = x21
        x2[:, :, :, 1::2] = x22
        
        x = torch.zeros_like(x_LL).repeat(1, 1, 2, 2)
        x[:, :, 0::2, :] = (x1 + x2)
        x[:, :, 1::2, :] = (x1 - x2)
        
        return x


# 实现 FreqFusion  中的空间可变滤波
class _SpatiallyVariantConvolution(nn.Module):
    """
    用于应用空间可变（即每像素不同）的 2D 卷积核。
    """
    def __init__(self, kernel_size):
        super(_SpatiallyVariantConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        # nn.Unfold 用于有效地从输入特征图中提取所有 KxK 的局部块
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=self.padding)

    def forward(self, features, kernels):
        """
        Args:
            features (torch.Tensor): 输入特征图 (B, C, H, W)
            kernels (torch.Tensor):  空间可变的卷积核 (B, K*K, H, W)

        Returns:
            torch.Tensor: 滤波后的特征图 (B, C, H, W)
        """
        B, C, H, W = features.shape
        K = self.kernel_size
        
        # 1. 提取局部块
        # features_unfolded: (B, C*K*K, H*W)
        features_unfolded = self.unfold(features)
        
        # 2. 重新塑形以便进行批处理乘法
        # (B, C, K*K, H*W)
        features_unfolded = features_unfolded.view(B, C, K*K, H*W)
        
        # (B, 1, K*K, H*W)
        kernels = kernels.view(B, 1, K*K, H*W) 

        # 3. 应用空间可变核
        # (B, C, K*K, H*W) * (B, 1, K*K, H*W) -> (B, C, K*K, H*W)
        weighted_patches = features_unfolded * kernels
        
        # 4. 求和得到卷积结果 (对 K*K 维度求和)
        # (B, C, H, W)
        filtered_features = torch.sum(weighted_patches, dim=2).view(B, C, H, W)
        
        return filtered_features


class _AdaptiveFilterPredictor(nn.Module):
    """
    FreqFusion  中描述的 "小型 CNN"  的一个实现，
    用于预测空间可变的核权重。
    """
    def __init__(self, in_channels, kernel_size, mid_channels_ratio=0.5):
        super(_AdaptiveFilterPredictor, self).__init__()
        
        mid_channels = int(in_channels * mid_channels_ratio)
        if mid_channels < 1:
            mid_channels = 1
        # 建议(ChatGPT)：LFDF/HFDF 中的 predictor 最好加上 dilation（模仿 FreqFusion）. 建议加Conv(dilation=2)
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, kernel_size * kernel_size, kernel_size=1, padding=0)
        )
        
        # 初始化最后一个卷积层，使其在训练开始时接近零
        self.predictor[-1].weight.data.zero_()
        self.predictor[-1].bias.data.zero_()

    def forward(self, x):
        # 输出 (B, K*K, H, W)
        return self.predictor(x)


# LFDF 和 HFDF
class LFDF(nn.Module):
    """
    低频差异滤波器 (LFDF)。
    借鉴 FreqFusion  (ALPF) 的思想 ：预测空间可变的低通（平滑）核，
    """
    def __init__(self, in_channels, kernel_size=5): # K=5 
        super(LFDF, self).__init__()
        self.kernel_size = kernel_size
        
        # 预测器，用于生成 (K*K) 个权重
        self.predictor = _AdaptiveFilterPredictor(in_channels, kernel_size)
        
        # 应用空间可变滤波的执行器
        self.apply_filter = _SpatiallyVariantConvolution(kernel_size)

    def forward(self, diff_ll):
        B, C, H, W = diff_ll.shape
        KK = self.kernel_size * self.kernel_size
        
        # 1. 预测空间可变的核 (B, K*K, H, W)
        # 建议 1：加入 Hamming window 正则化。 可加：kernels = kernels * Hamming(K)，kernels = kernels / sum
        kernels = self.predictor(diff_ll)
        
        # 2. 归一化核，使其成为低通滤波器 (Low-Pass Filter)
        kernels = F.softmax(kernels.view(B, KK, H*W), dim=1) # 在 K*K 维度上 Softmax
        kernels = kernels.view(B, KK, H, W)
        
        # 3. 应用自适应低通滤波器
        filtered_ll = self.apply_filter(diff_ll, kernels)
        return filtered_ll


class HFDF(nn.Module):
    """
    高频差异滤波器 (HFDF)。
    """
    def __init__(self, in_channels, kernel_size=3): # K=3
        super(HFDF, self).__init__()
        # 建议(ChatGPT) 对 HFDF 进行 multi-kernel size 融合,高频差异原本就复杂，推荐K=3 + K=5 两个 HPF,Concat → 1x1 Conv
        self.kernel_size = kernel_size
        
        # 预测器，用于生成 (K*K) 个 LPF 权重
        self.predictor = _AdaptiveFilterPredictor(in_channels, kernel_size)
        
        # 应用空间可变滤波的执行器
        self.apply_filter = _SpatiallyVariantConvolution(kernel_size)
        
        # 创建一个恒等的核 (Identity Kernel) (中心的 1)
        identity_kernel = torch.zeros(kernel_size * kernel_size)
        center_idx = (kernel_size * kernel_size) // 2
        identity_kernel[center_idx] = 1.0
        
        # 注册为 buffer (非可训练参数)
        # (1, K*K, 1, 1) 以便广播
        self.register_buffer('identity_kernel', identity_kernel.view(1, -1, 1, 1))

    def forward(self, diff_hf):
        B, C, H, W = diff_hf.shape
        KK = self.kernel_size * self.kernel_size
        
        # 1. (B, K*K, H, W)
        lpf_kernels = self.predictor(diff_hf)
        
        # 2. 归一化核，使其成为低通滤波器 (LPF)
        lpf_kernels = F.softmax(lpf_kernels.view(B, KK, H*W), dim=1) # 在 K*K 维度上 Softmax
        lpf_kernels = lpf_kernels.view(B, KK, H, W)
        
        # 3. 通过滤波器反转创建高通滤波器 (HPF)
        #    HPF = Identity - LPF (FreqFusion  的核心逻辑 )
        #    self.identity_kernel (1, K*K, 1, 1) 会被广播到 (B, K*K, H, W)
        hpf_kernels = self.identity_kernel - lpf_kernels
        
        # 4. 应用自适应高通滤波器
        filtered_hf = self.apply_filter(diff_hf, hpf_kernels)
        return filtered_hf
    
# 建议5：SE 注意力模块（用于 IDWT 前增强）
class SEBlock(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w
    
# 核心模块：FAFF (频率域自适应滤波)

class FAFF(nn.Module):
    """
    频率域自适应滤波模块 (FAFF)
    1. 使用 DWT (来自 WINet [1, 1]) 将对齐的 T1 和 T2 特征分解 。
    2. 计算子带差异图 (Diff_LL, Diff_HF)。
    3. 应用 LFDF (类 ALPF ) 抑制 Diff_LL 中的伪变化 。
    4. 应用 HFDF (类 AHPF ) 增强 Diff_HF 中的真实边缘 。
    5. 使用 IDWT (来自 WINet [1, 1]) 重构“干净”的特征差异图 D_freq_clean。
    """
    def __init__(self, in_channels):
        super(FAFF, self).__init__()
        
        # 1. 初始化 DWT 和 IDWT 模块
        self.dwt = DWT()
        self.idwt = IWT()
        
        self.in_channels = in_channels
        # 3 个高频子带 (LH, HL, HH) 拼接
        self.hf_channels = in_channels * 3 
        
        # 2. 初始化自适应滤波器
        # 低频滤波器 (K=5) 
        self.lfdf_filter = LFDF(in_channels=in_channels, kernel_size=5)
        
        # 高频滤波器 (K=3) 
        self.hfdf_filter = HFDF(in_channels=self.hf_channels, kernel_size=3)
        # self.hfdf_filter = HFDF(in_channels=in_channels, kernel_size=3)
        
        # self.hi = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, F_T1_norm, F_T2_aligned):
        # 建议 6：在整个 FAFF 前加一个 deformable alignment（非常有效）
        # T2_feature_aligned = DeformAlign(F_T2, F_T1)
        
        # 1. DWT 分解
        # LL: (B, C, H/2, W/2)
        # HFs: (B, C, 3, H/2, W/2)
        LL_T1, HFs_T1 = self.dwt(F_T1_norm)
        LL_T2, HFs_T2 = self.dwt(F_T2_aligned)

        # 2. 计算子带差异
        Diff_LL = LL_T2 - LL_T1
        Diff_HFs = HFs_T2 - HFs_T1
        # Diff = F_T2_aligned - F_T1_norm
        
        # 重塑 HF 子带以便于滤波 (B, C*3, H/2, W/2)
        B, C, _, H_d, W_d = Diff_HFs.shape
        Diff_HF_reshaped = Diff_HFs.view(B, self.hf_channels, H_d, W_d)
        
        # 3. 应用自适应低频滤波
        Filtered_Diff_LL = self.lfdf_filter(Diff_LL)
        
        # 4. 应用自适应高频滤波
        Filtered_Diff_HF_reshaped = self.hfdf_filter(Diff_HF_reshaped)
        
        # 5. 恢复 HF 子带形状
        # (B, C*3, H/2, W/2) -> (B, C, 3, H/2, W/2)
        Filtered_Diff_HFs = Filtered_Diff_HF_reshaped.view(B, C, 3, H_d, W_d)
        
        # 6. IDWT 重构 (来自 WINet [1, 1])
        # 建议5：在 IDWT 前加入轻量级注意力增强，Filtered_Diff_LL → SE/CBAM，Filtered_Diff_HF → SE/CBAM
        D_freq_clean = self.idwt(Filtered_Diff_LL, Filtered_Diff_HFs)
        # D_freq_clean = self.hi(torch.cat([Filtered_Diff_LL, Filtered_Diff_HF_reshaped], dim=1))
        return D_freq_clean
    
# # 提供给 gate 使用的空间差分平滑层   
# def Spatial_Smoothing_Module(in_channels):
#     return nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
# -----------------------------------------------------------------------------
# 不确定的实现细节 (Uncertain Implementation Details):
# 1. 空间注意力的卷积核大小 (Spatial Attention Kernel Size):
#    -  (SAFFM 论文) 在其公式(2)和图2中指定了一个 3x3 卷积核 。
#    - (winet.py 分析) 和  (CBAM 论文) 使用了一个 7x7 的卷积核。
#    - 使用 7x7 卷积核，因为它在空间注意力模块中是一个更广泛的标准（如 CBAM  和 `winet.py` 所示），但这与 论文中的 3x3 规定不同，因此标记为不确定。
#
# 2. “频率注意力分支”的架构:
#    - 报告第 5.2 节提到，借鉴  的“MLP 结构”（即通道注意力）。
#    - 然而，报告第 5.1 节的核心逻辑是：“只有当一个区域在 D_spatial... 和 D_freq_clean 中... *同时*表现出强烈信号时”。
#    - 决定: 为了实现这种空间上的“逻辑与”门控，两个分支都必须生成*空间*注意力图（B, 1, H, W）。因此，我们将为 "freq_attn_branch" 实现一个与 "spatial_attn_branch" 相同的空间注意力模块，而不是一个通道注意力（MLP）模块。
# -----------------------------------------------------------------------------


class _SpatialAttentionModule(nn.Module):
    """
    生成空间注意力图。
    """
    def __init__(self, kernel_size=7):
        super(_SpatialAttentionModule, self).__init__()
        
        # (winet.py) 和  (CBAM) 使用 7x7。
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels=2, 
            out_channels=1, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        输入 x: 特征图 (B, C, H, W)
        输出: 空间注意力图 (B, 1, H, W)
        """
        # 沿通道维度的平均池化和最大池化 
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接 
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 卷积和 Sigmoid  (公式 2)
        attn_map = self.sigmoid(self.conv(x_cat))
        
        return attn_map


class CAGM(nn.Module):
    """
    变化注意力指导模块 (CAGM)
    """
    def __init__(self, in_channel, out_channels, kernel_size=7):
        super(CAGM, self).__init__()
        
        # 改动1：添加一个轻量级的空间平滑层
        self.spatial_smooth = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,  
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )
        
        # 1. 空间注意力分支 (来自 D_spatial)
        self.spatial_attn_branch = _SpatialAttentionModule(kernel_size)
        
        # 2. 频率注意力分支 (来自 D_freq_clean)
        self.freq_attn_branch = _SpatialAttentionModule(kernel_size)

    def forward(self, D_spatial, D_freq_clean):
        # 1. 生成空间注意力图 A_spatial
        
        # 风险 1：D_spatial 噪声太大可能干扰 gat. 可以加轻量平滑D_spatial_smooth = Conv3x3(D_spatial)或D_spatial → BatchNorm → ReLU → Conv → Gate
        D_spatial = self.spatial_smooth(D_spatial)
        A_spatial = self.spatial_attn_branch(D_spatial)
        
        # 2. 生成频率注意力图 A_freq
        
        A_freq = self.freq_attn_branch(D_freq_clean)
        
        # 3. 生成变化指导图 (Change Guidance Map)
        # 改动2：Soft Gate
        Gate = torch.sigmoid(A_spatial + A_freq)
        
        # Change_Guidance_Map = A_spatial * A_freq # 逐元素相乘 (Hadamard product) 
        
        # 4. 指导解码器进行多尺度特征融合
        # 改动3：双流融合
        Fused_Feature = D_freq_clean * Gate + D_spatial * (1 - Gate)
        # 改动4：残差连接，增强频率特征
        # Fused_Feature = Fused_Feature + 0.1 *D_freq_clean  
        
        return Fused_Feature
    
class BoundaryRefinementHead(nn.Module):
    """
    边界精炼头 (BRH)。
    """
    def __init__(self, in_channels, reduction_channels=None, max_offset_scale=0.05):

        super(BoundaryRefinementHead, self).__init__()
        self.max_offset_scale = max_offset_scale
        # [不确定点 2]: 偏移量预测器的确切架构未指定。
        # 使用一个轻量级的 CNN，类似于第 3 部分的 PFA 模块
        if reduction_channels is None:
            reduction_channels = in_channels // 2 if in_channels > 4 else in_channels
            
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, 2, kernel_size=3, padding=1) # 输出 (dx, dy)
        )
        
        # 初始化最终偏移量预测层的偏置为 0，
        # 使其在训练开始时接近于恒等变换 (identity mapping)。
        self.offset_predictor[-1].weight.data.zero_()
        self.offset_predictor[-1].bias.data.zero_()

    def forward(self, F_decoder_final):
        
        B, C, H, W = F_decoder_final.shape
        
        # 1. 预测边界偏移场 (Delta_boundary) [1, 1]
        # offsets 形状: (B, 2, H, W)
        offsets = self.offset_predictor(F_decoder_final)
        # 改动1:限制偏移范围，防止不稳定
        offsets = torch.tanh(offsets) * self.max_offset_scale
        # 2. 创建用于 F.grid_sample (LIF) 的基础网格 
        x_coords = torch.linspace(-1.0, 1.0, W, device=F_decoder_final.device, dtype=F_decoder_final.dtype)
        y_coords = torch.linspace(-1.0, 1.0, H, device=F_decoder_final.device, dtype=F_decoder_final.dtype)
        
        # indexing='ij' 使得 y_grid.shape = (H, W), x_grid.shape = (H, W)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # base_grid 形状 (H, W, 2)，顺序为 (x, y)
        base_grid = torch.stack([x_grid, y_grid], dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)
        
        # 3. 将预测的偏移量应用于网格
        # offsets (B, 2, H, W) -> offsets_permuted (B, H, W, 2)
        offsets_permuted = offsets.permute(0, 2, 3, 1)
        
        # 偏移量尺度
        final_grid = base_grid + offsets_permuted
        
        # 4. 使用 LIF (F.grid_sample) 重采样特征图 
        F_refined = F.grid_sample(
            F_decoder_final, 
            final_grid, 
            mode='bilinear',          
            padding_mode='border',    
            align_corners=False      
        )
      
        # 改动2：残差连接，增强特征稳定性
        F_refined = F_refined + 0.1 * F_decoder_final
        
        return F_refined
    
class WaveletLoss(nn.Module):

    def __init__(self):
        super(WaveletLoss, self).__init__()
        self.dwt = DWT()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_prob, target_mask):
        # 1. 提取变化通道
        if pred_prob.shape[1] == 2:
            pred_change = pred_prob[:, 1:2, :, :]
        else:
            pred_change = pred_prob
            
        if target_mask.shape[1] == 2:
            target_change = target_mask[:, 1:2, :, :]
        else:
            # 如果真值只有1个通道，直接使用，不需要切片
            target_change = target_mask   

        # 2. 【关键修复】清理 target_mask 中的 NaN，防止传播
        if torch.isnan(target_change).any():
             target_change = torch.nan_to_num(target_change, nan=0.0)

        # 3. DWT 变换
        _, pred_hfs = self.dwt(pred_change)
        _, target_hfs = self.dwt(target_change)
        
        # 4. 【关键修复】防止空张量导致的 NaN (0/0)
        if pred_hfs.numel() == 0:
            return torch.tensor(0.0, device=pred_prob.device, requires_grad=True)

        loss = self.l1_loss(pred_hfs, target_hfs)
        
        return loss
