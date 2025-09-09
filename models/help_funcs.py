import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


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


class StyleEncoder(nn.Module):
    """
    复现 "Harmony in diversity" (CCNet) 论文中 Fig. 5(c) 的风格编码器。
    该编码器将一张图像编码为一个代表其全局风格的特征向量。
    """
    def __init__(self, in_channels=3, style_dim=256):
        """
        初始化函数。
        :param in_channels: 输入图像的通道数,默认为3 (RGB)。
        :param style_dim: 输出风格向量的维度。论文中提到与最深层内容特征C5的通道数一致,
                          这里可以设置为一个超参数,例如256或512。
        """
        super(StyleEncoder, self).__init__()

        # 论文中没有指定通道数，我们使用一个合理的递增序列
        channels = [64, 128, 256, style_dim]

        # 卷积块 1: Conv 7x7 -> ReLU -> Max-pool
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 尺寸减半
        )

        # 卷积块 2: Conv 3x3 -> ReLU -> Max-pool
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 尺寸减半
        )
        
        # 卷积块 3: Conv 3x3 -> ReLU -> Max-pool
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 尺寸减半
        )

        # 卷积块 4: Conv 3x3 -> ReLU
        self.block4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # 输出层: Avg-pool -> FC layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 使用一个1x1卷积来实现全连接层的功能，更灵活
        self.fc_layer = nn.Conv2d(channels[3], style_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播过程。
        :param x: 输入的图像张量，形状为 (B, C, H, W)。
        :return: 风格特征向量，形状为 (B, style_dim, 1, 1)。
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avg_pool(x)
        style_vector = self.fc_layer(x)
        
        return style_vector

def adain(content_feat, style_feat, epsilon=1e-5):
    """
    自适应实例归一化 (AdaIN) 实现。
    Args:
        content_feat (torch.Tensor): 内容特征图 (B, C, H, W)。
        style_feat (torch.Tensor): 风格向量 (B, C*2)，其中前一半是均值，后一半是标准差。
                                     或者直接是 (B, style_dim)，然后通过MLP生成均值和标准差。
                                     这里我们假设style_feat可以直接映射。
    Returns:
        torch.Tensor: 施加风格后的内容特征图。
    """
    # 假设style_feat已经通过一个MLP生成了均值和标准差
    # 这里为了简化，我们假设style_feat可以直接reshape
    # 一个更鲁棒的实现是使用一个小的MLP网络将style_vector映射到gamma和beta
    style_mean, style_std = style_feat.chunk(2, 1)


    content_mean = torch.mean(content_feat, dim=[2,3], keepdim=True)
    content_std = torch.std(content_feat, dim=[2,3], keepdim=True)
    
    normalized_feat = (content_feat - content_mean) / (content_std + epsilon)
    return normalized_feat * style_std.unsqueeze(-1).unsqueeze(-1) + style_mean.unsqueeze(-1).unsqueeze(-1)

class RestorationDecodeBlock(nn.Module):
    """
    重建解码器中的单个块，参考论文图5(d)。
    """
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 用于将style_vector映射到AdaIN所需的gamma和beta
        self.style_mlp = nn.Linear(style_dim, out_channels * 2)

    def forward(self, x, style_vector):
        
        fused_feat = self.conv(x)        
        # 生成AdaIN参数
        style_params = self.style_mlp(style_vector)
        styled_feat = adain(fused_feat, style_params)
        output = self.relu(styled_feat)
        return output
    
class RestorationDecoder(nn.Module):
    """
    完整的重建解码器，参考论文图6。
    """
    def __init__(self, content_channels, style_dim=256, final_out_channels=3):
        super().__init__()
        # content_channels: 一个列表，包含从深到浅各层内容特征的通道数，例如 
        # 假设c5是初始输入，其通道数为content_channels
        self.init_conv = nn.Conv2d(content_channels[0], content_channels[0], kernel_size=3, padding=1)
        
        self.decoder_blocks = nn.ModuleList()
        # skip_channels: [c4, c3, c2, c1] 的通道数
        skip_channels = content_channels[1:] 
        # up_channels: [d4, d3, d2] 的通道数，d4是init_conv的输出
        up_channels = content_channels[:-1]
        
        for i in range(len(content_channels) - 1):
            # in_channels = up_channels[i] (来自上采样) + skip_channels[i] (来自跳跃连接)
            # out_channels = skip_channels[i]
            self.decoder_blocks.append(
                RestorationDecodeBlock(content_channels[i] + content_channels[i+1], content_channels[i+1], style_dim)
            )

        # 最终输出层，将特征图恢复到原始图像通道数
        self.final_conv = nn.Conv2d(content_channels[-1], final_out_channels, kernel_size=7, padding=3)
        self.tanh = nn.Tanh() # 将输出像素值归一化到 [-1, 1]

    def forward(self, content_features, style_vector):
        """
        Args:
            content_features (list): 从深到浅的多尺度内容特征图 [c5, c4, c3, c2, c1]。
            style_vector (torch.Tensor): 风格向量 (B, style_dim)。
        Returns:
            torch.Tensor: 重建的图像 (B, 3, H, W)。
        """
        # 最深层的特征作为解码器初始输入
        x = self.init_conv(content_features[0])
        
        # 级联解码
        for i, block in enumerate(self.decoder_blocks):
            # content_features[i+1] 是 c4, c3, c2, c1
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            skip_feat = content_features[i+1]
            x = torch.cat([x, skip_feat], dim=1)
            x = block(x, style_vector.view(style_vector.size(0), -1))

        # 最后一层上采样到原始尺寸
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        restored_image = self.final_conv(x)
        restored_image = self.tanh(restored_image)
        
        return restored_image