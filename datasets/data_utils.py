import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


class CDDataAugmentation:
    import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageFilter

# Helper functions 保持不变，直接复制你的即可
def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)
    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    # 加个保护，防止 box 越界报错
    try:
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    except:
        # 极端情况下如果算错坐标，返回原图resize或黑图，保证不崩
        return image.resize((cropsize, cropsize)) 
    return Image.fromarray(cont)

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)
    w_space = w - cropsize
    h_space = h - cropsize
    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0
    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0
    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)

def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)


# --- 主类 ---

class CDDataAugmentation:
    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False, # 你原本的自定义缩放裁剪
            with_random_blur=False,
            with_random_noise=False,       # [新增] 建议开启
            with_random_cutout=False,      # [新增] 建议开启
            random_color_tf=False
    ):
        self.img_size = img_size
        self.img_size_dynamic = (img_size is None)
        
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.with_random_noise = with_random_noise
        self.with_random_cutout = with_random_cutout
        self.random_color_tf = random_color_tf

    def transform(self, imgs, labels, to_tensor=True):
        # 1. PIL Conversion
        imgs = [TF.to_pil_image(img) for img in imgs]
        labels = [TF.to_pil_image(img) for img in labels]
        
        # 2. Initial Resize
        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BICUBIC) for img in imgs]
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=Image.NEAREST) for img in labels]

        # ---------------- 几何变换 (同步) ----------------
        
        # Flip & Rot
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        # Random Resized Crop (官方版)
        # 修复 Bug: 只有 50% 概率触发，而不是 > 0 (100%)
        # 注意：scale_random_crop 和 random_crop 最好互斥，或者概率叠加。这里我让它们独立判定。
        if self.with_random_crop and random.random() > 0.5:
            # scale 设为 0.5~1.0 增加难度
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.5, 1.0), ratio=(0.8, 1.2))
            
            imgs = [TF.resized_crop(img, i, j, h, w, size=(self.img_size, self.img_size), interpolation=Image.BICUBIC) for img in imgs]
            labels = [TF.resized_crop(img, i, j, h, w, size=(self.img_size, self.img_size), interpolation=Image.NEAREST) for img in labels]

        # Scale Random Crop (你的自定义版 - 支持 Zoom Out)
        # 修复 Bug: 增加概率判定 > 0.5
        elif self.with_scale_random_crop and random.random() > 0.5: 
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
            
            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            
            imgsize = imgs[0].size
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            
            # 注意：Label 填充 255 (通常是 ignore index)，这很好
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0) for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255) for img in labels]

        # ---------------- 像素变换 (仅图像) ----------------

        # Blur
        # 修复 Bug: 增加概率判定 > 0.8 (20%概率模糊)
        if self.with_random_blur and random.random() > 0.8:
            radius = random.uniform(0.1, 1.5)
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius)) for img in imgs]

        # Color Jitter
        if self.random_color_tf and random.random() > 0.2:
            # 稍微加强一点点对比度
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            # Independent Jitter: 每张图随机变，增强鲁棒性
            imgs = [jitter(img) for img in imgs]

        # [新增] Gaussian Noise (使用 float 计算防止溢出白点)
        if self.with_random_noise and random.random() > 0.8:
            def add_noise(pil_img):
                img_np = np.array(pil_img).astype(np.float32)
                noise = np.random.normal(0, 15, img_np.shape) # 噪声强度 15
                img_np = img_np + noise
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                return Image.fromarray(img_np)
            imgs = [add_noise(img) for img in imgs]

        # [新增] Cutout (Transformer 必备)
        if self.with_random_cutout and random.random() > 0.6: # 40% 概率触发
            def cutout(pil_img):
                img_np = np.array(pil_img)
                h, w = img_np.shape[:2]
                ratio = random.uniform(0.1, 0.2) # 遮挡 10%-20%
                ch, cw = int(h*ratio), int(w*ratio)
                y = random.randint(0, h - ch)
                x = random.randint(0, w - cw)
                # 填中性灰 127，比纯黑 0 对模型更友好
                img_np[y:y+ch, x:x+cw] = 127 
                return Image.fromarray(img_np)
            imgs = [cutout(img) for img in imgs]

        # ---------------- Tensor & Norm ----------------
        if to_tensor:
            imgs = [TF.to_tensor(img) for img in imgs]
            # Label 保持 uint8 (long)
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0) for img in labels]
            
            # 这里做了归一化，所以外部函数不要再做了
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for img in imgs]

        return imgs, labels
    # def __init__(
    #         self,
    #         img_size,
    #         with_random_hflip=False,
    #         with_random_vflip=False,
    #         with_random_rot=False,
    #         with_random_crop=False,
    #         with_scale_random_crop=False,
    #         with_random_blur=False,
    #         random_color_tf=False
    # ):
    #     self.img_size = img_size
    #     if self.img_size is None:
    #         self.img_size_dynamic = True
    #     else:
    #         self.img_size_dynamic = False
    #     self.with_random_hflip = with_random_hflip
    #     self.with_random_vflip = with_random_vflip
    #     self.with_random_rot = with_random_rot
    #     self.with_random_crop = with_random_crop
    #     self.with_scale_random_crop = with_scale_random_crop
    #     self.with_random_blur = with_random_blur
    #     self.random_color_tf=random_color_tf
    # def transform(self, imgs, labels, to_tensor=True):
    #     """
    #     :param imgs: [ndarray,]
    #     :param labels: [ndarray,]
    #     :return: [ndarray,],[ndarray,]
    #     """
    #     # resize image and covert to tensor
    #     imgs = [TF.to_pil_image(img) for img in imgs]
    #     if self.img_size is None:
    #         self.img_size = None

    #     if not self.img_size_dynamic:
    #         if imgs[0].size != (self.img_size, self.img_size):
    #             imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
    #                     for img in imgs]
    #     else:
    #         self.img_size = imgs[0].size[0]

    #     labels = [TF.to_pil_image(img) for img in labels]
    #     if len(labels) != 0:
    #         if labels[0].size != (self.img_size, self.img_size):
    #             labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
    #                     for img in labels]

    #     random_base = 0.5
    #     if self.with_random_hflip and random.random() > 0.5:
    #         imgs = [TF.hflip(img) for img in imgs]
    #         labels = [TF.hflip(img) for img in labels]

    #     if self.with_random_vflip and random.random() > 0.5:
    #         imgs = [TF.vflip(img) for img in imgs]
    #         labels = [TF.vflip(img) for img in labels]

    #     if self.with_random_rot and random.random() > random_base:
    #         angles = [90, 180, 270]
    #         index = random.randint(0, 2)
    #         angle = angles[index]
    #         imgs = [TF.rotate(img, angle) for img in imgs]
    #         labels = [TF.rotate(img, angle) for img in labels]

    #     if self.with_random_crop and random.random() > 0:
    #         i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
    #             get_params(img=imgs[0], scale=(0.8, 1.2), ratio=(1, 1))

    #         imgs = [TF.resized_crop(img, i, j, h, w,
    #                                 size=(self.img_size, self.img_size),
    #                                 interpolation=Image.CUBIC)
    #                 for img in imgs]

    #         labels = [TF.resized_crop(img, i, j, h, w,
    #                                   size=(self.img_size, self.img_size),
    #                                   interpolation=Image.NEAREST)
    #                   for img in labels]

    #     if self.with_scale_random_crop:
    #         # rescale
    #         scale_range = [1, 1.2]
    #         target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    #         imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
    #         labels = [pil_rescale(img, target_scale, order=0) for img in labels]
    #         # crop
    #         imgsize = imgs[0].size  # h, w
    #         box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
    #         imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
    #                 for img in imgs]
    #         labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
    #                 for img in labels]

    #     if self.with_random_blur and random.random() > 0:
    #         radius = random.random()
    #         imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
    #                 for img in imgs]

    #     if self.random_color_tf:
    #         color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
    #         imgs_tf = []
    #         for img in imgs:
    #             tf = transforms.ColorJitter(
    #                         color_jitter.brightness, 
    #                         color_jitter.contrast, 
    #                         color_jitter.saturation,
    #                         color_jitter.hue)
    #             imgs_tf.append(tf(img))
    #         imgs = imgs_tf
            
    #     if to_tensor:
    #         # to tensor
    #         imgs = [TF.to_tensor(img) for img in imgs]
    #         labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
    #                   for img in labels]
            
    #         imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #                 for img in imgs]

    #     return imgs, labels



def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
