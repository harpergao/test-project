import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.ChangeFormer import ChangeFormerV1, ChangeFormerV2, ChangeFormerV3, ChangeFormerV4, ChangeFormerV5, ChangeFormerV6
from models.SiamUnet_diff import SiamUnet_diff
from models.SiamUnet_conc import SiamUnet_conc
from models.Unet import Unet
from models.DTCDSCN import CDNet34

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'          # 国内镜像1
# os.environ['HF_ENDPOINT'] = 'https://modelscope.cn/'          # 国内镜像2
import timm

from fpn import FPN
###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)

    elif args.net_G == 'base_efficientnet_b4':
        net = EfficientNet(input_nc=3, output_nc=2, backbone='efficientnet_b4')
        
    elif args.net_G == 'ChangeFormerV1':
        net = ChangeFormerV1() #ChangeFormer with Transformer Encoder and Convolutional Decoder
    
    elif args.net_G == 'ChangeFormerV2':
        net = ChangeFormerV2() #ChangeFormer with Transformer Encoder and Convolutional Decoder

    elif args.net_G == 'ChangeFormerV3':
        net = ChangeFormerV3() #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)

    elif args.net_G == 'ChangeFormerV4':
        net = ChangeFormerV4() #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    
    elif args.net_G == 'ChangeFormerV5':
        net = ChangeFormerV5(embed_dim=args.embed_dim) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)

    elif args.net_G == 'ChangeFormerV6':
        net = ChangeFormerV6(embed_dim=args.embed_dim) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    
    elif args.net_G == "SiamUnet_diff":
        #Implementation of ``Fully convolutional siamese networks for change detection''
        #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = SiamUnet_diff(input_nbr=3, label_nbr=2)

    elif args.net_G == "SiamUnet_conc":
        #Implementation of ``Fully convolutional siamese networks for change detection''
        #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = SiamUnet_conc(input_nbr=3, label_nbr=2)

    elif args.net_G == "Unet":
        #Usually abbreviated as FC-EF = Image Level Concatenation
        #Implementation of ``Fully convolutional siamese networks for change detection''
        #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = Unet(input_nbr=3, label_nbr=2)
    
    elif args.net_G == "DTCDSCN":
        #The implementation of the paper"Building Change Detection for Remote Sensing Images Using a Dual Task Constrained Deep Siamese Convolutional Network Model "
        #Code copied from: https://github.com/fitzpchao/DTCDSCN
        net = CDNet34(in_channels=3)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)    #1/2， in=3, out=64
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  #1/4， in=64, out=64

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)    #if stages_num < 5: 1/4, in=512, out=512
                                        
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)   #stages_num < 5: 1/4, in=512, out=512
        return x


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        
        # this zone will be rewritten in the future
        
        x = torch.abs(x1 - x2)
        
        # ######
        
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        outputs = []
        outputs.append(x)
        return outputs

def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out)) * 0.1
        out = self.bn2(self.conv2(out))
        out += residual
        return out
    
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )
    
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )
class EfficientNet(nn.Module):
    def __init__(self, input_nc, output_nc=2,
                 backbone='efficientnet_b4',
                 output_sigmoid=False, if_upsample_2x=True):
        super(EfficientNet, self).__init__()
        if backbone == 'efficientnet_b4':
            self.efficientnet = timm.create_model(
                                'efficientnet_b4', pretrained=True, features_only=True,
                            )
        else:
            raise NotImplementedError
        
        feature_channels = self.efficientnet.feature_info.channels()
        fpn_in_channels = feature_channels[1:]
        fpn_out_channels = 256
        
        self.fpn = FPN(
            in_channels=fpn_in_channels, # 刚刚动态获取的通道数列表
            out_channels=fpn_out_channels,       # FPN 输出的所有特征图的统一通道数，256 是一个常用值
            num_outs=len(fpn_in_channels) # FPN 输出的特征图数量，通常和输入数量一致
            )
        
        self.convd2x    = nn.ConvTranspose2d(fpn_out_channels*4, fpn_out_channels, kernel_size=4, stride=2, padding=1)
        self.dense_2x   = nn.Sequential( ResidualBlock(fpn_out_channels))
        self.convd1x    = nn.ConvTranspose2d(fpn_out_channels, fpn_out_channels//2, kernel_size=4, stride=2, padding=1)
        self.dense_1x   = nn.Sequential( ResidualBlock(fpn_out_channels//2))
        self.change_probability = nn.Conv2d(fpn_out_channels//2, output_nc, kernel_size=3, stride=1, padding=1)
        self.decoder_layer1 = nn.Sequential(
            ResidualBlock(fpn_out_channels * 2),
            ResidualBlock(fpn_out_channels * 2),
            ResidualBlock(fpn_out_channels * 2),
            ResidualBlock(fpn_out_channels * 2),
        )
        self.decoder_layer2 = nn.Sequential(
            ResidualBlock(fpn_out_channels),
            ResidualBlock(fpn_out_channels),
            ResidualBlock(fpn_out_channels),
        )
        self.diff_c3   = conv_diff(in_channels=2*fpn_out_channels, out_channels=fpn_out_channels)
        self.diff_c2   = conv_diff(in_channels=2*fpn_out_channels, out_channels=fpn_out_channels)
        self.diff_c1   = conv_diff(in_channels=2*fpn_out_channels, out_channels=fpn_out_channels)
        self.diff_c0   = conv_diff(in_channels=2*fpn_out_channels, out_channels=fpn_out_channels)
        
        self.pre3 = make_prediction(fpn_out_channels, output_nc)
        self.pre2 = make_prediction(fpn_out_channels, output_nc)
        self.pre1 = make_prediction(fpn_out_channels, output_nc)
        self.pre0 = make_prediction(fpn_out_channels, output_nc)
        
        self.bottleneck_conv = nn.Conv2d(in_channels=fpn_out_channels * 8, out_channels=fpn_out_channels, kernel_size=1, bias=False)
        # self.relu = nn.ReLU()
        # self.upsamplex2 = nn.Upsample(scale_factor=2)
        # self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        # self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        # self.if_upsample_2x = if_upsample_2x

        # self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        # self.output_sigmoid = output_sigmoid
        # self.sigmoid = nn.Sigmoid()

    def change_feature(self, x, y):
        i = 2
        for index in range(0, len(x), i):
            x[index], y[index] = y[index], x[index]
        return x, y
    def euclidean_distance(self, img1, img2):
        diff_squared = (img1 - img2) ** 2
        distances = torch.sqrt(torch.sum(diff_squared, dim=1)).unsqueeze(1)
        max_distance = torch.max(distances)
        normalized_distances = torch.sigmoid(distances / max_distance)
        return normalized_distances
    
    def forward(self, x1, x2):
        x1 = self.efficientnet(x1)[1:]
        x2 = self.efficientnet(x2)[1:]
        # diff = []
        # for f1, f2 in zip(x1, x2):
        #     diff.append(torch.abs(f1 - f2))
        # x = self.fpn(diff)
        # p2, p3, p4, p5 = x
        x1, x2 = self.change_feature(x1, x2)
        x1 = self.fpn(x1)
        x2 = self.fpn(x2)
        x1, x2 = self.change_feature(list(x1), list(x2))
        
        outputs = []
        
        # --- Decoder Level 3 (Lowest Resolution) ---
        x12_3 = torch.cat([x1[3], x2[3]], dim=1)
        x12_3 = self.euclidean_distance(x1[3], x2[3]) * self.decoder_layer1[3](x12_3)
        x12_3 = self.diff_c3(x12_3)
        x12_30 = self.pre3(x12_3) # Prediction head for this level
        outputs.append(x12_30)
        # Prepare for final fusion by upsampling to the highest resolution
        x12_31 = F.interpolate(x12_3, size=x1[0].size()[2:], mode='bilinear', align_corners=False)

        # --- Decoder Level 2 ---
        x12_2 = torch.cat([x1[2], x2[2]], dim=1)
        x12_2 = self.diff_c2(x12_2) + F.interpolate(x12_3, scale_factor=2, mode='bilinear', align_corners=False)
        x12_2 = self.euclidean_distance(x1[2], x2[2]) * self.decoder_layer2[2](x12_2)

        x12_20 = self.pre2(x12_2) # Prediction head for this level
        outputs.append(x12_20)
        # Prepare for final fusion by upsampling to the highest resolution
        x12_21 = F.interpolate(x12_2, size=x1[0].size()[2:], mode='bilinear', align_corners=False)

        # --- Decoder Level 1 ---
        x12_1 = torch.cat([x1[1], x2[1]], dim=1)
        x12_1 = self.diff_c1(x12_1) + F.interpolate(x12_2, scale_factor=2, mode='bilinear', align_corners=False)
        x12_1 = self.euclidean_distance(x1[1], x2[1]) * self.decoder_layer2[1](x12_1)
 
        x12_10 = self.pre1(x12_1) # Prediction head for this level
        outputs.append(x12_10)
        # Prepare for final fusion by upsampling to the highest resolution
        x12_11 = F.interpolate(x12_1, size=x1[0].size()[2:], mode='bilinear', align_corners=False)
        
        # --- Decoder Level 0 (Highest Resolution) ---
        x12_0 = torch.cat([x1[0], x2[0]], dim=1)
        x12_0 = self.diff_c0(x12_0) + F.interpolate(x12_1, scale_factor=2, mode='bilinear', align_corners=False)
        x12_0 = self.euclidean_distance(x1[0], x2[0]) * self.decoder_layer2[0](x12_0)

        x12_00 = self.pre0(x12_0) # Prediction head for this level
        outputs.append(x12_00)
        
        # x12_1 = torch.cat([x1[1], x2[1]], dim=1)
        # x12_1 = self.decoder_layer1[1](x12_1) + F.interpolate(x12_2, scale_factor=2, mode='bilinear', align_corners=False)
        # x12_1 = self.euclidean_distance(x1[1], x2[1])*self.decoder_layer2[1](x12_1)
        # x12_10 = self.pre5(x12_1)
        # outputs.append(x12_10)
        
        x_fuse = torch.cat([x12_0, x12_11, x12_21, x12_31], dim=1)
        # x_fuse = self.bottleneck_conv(x_fuse)
        x = self.convd2x(x_fuse)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)
        #Final prediction
        cp = self.change_probability(x)
        outputs.append(cp)
        
        
        # outputs = []
        # p3_upsampled = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear')
        # p4_upsampled = F.interpolate(p4, size=p2.shape[-2:], mode='bilinear')
        # p5_upsampled = F.interpolate(p5, size=p2.shape[-2:], mode='bilinear')        
        
        # fused_features = torch.cat([p2, p3_upsampled, p4_upsampled, p5_upsampled], dim=1)
        # x = self.convd2x(fused_features)
        # #Residual block
        # x = self.dense_2x(x)
        # #Upsampling x2 (x1 scale)
        # x = self.convd1x(x)
        # #Residual block
        # x = self.dense_1x(x)
        # #Final prediction
        # cp = self.change_probability(x)
        # outputs.append(cp)
        return outputs