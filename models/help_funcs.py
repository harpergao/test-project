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



