from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# class MobileV3Unet(nn.Module):
#     def __init__(self, num_classes: int = 3, pretrain_backbone: str = "MobileNet_V3_Large_Weights.IMAGENET1K_V2"):
#         super(MobileV3Unet, self).__init__()
#         backbone = mobilenet_v3_large(weights=pretrain_backbone)

#         backbone = backbone.features

#         stage_indices = [1, 3, 6, 12, 15]
#         self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
#         return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
#         self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

#         c = self.stage_out_channels[4] + self.stage_out_channels[3]
#         self.up1 = Up(c, self.stage_out_channels[3])
#         c = self.stage_out_channels[3] + self.stage_out_channels[2]
#         self.up2 = Up(c, self.stage_out_channels[2])
#         c = self.stage_out_channels[2] + self.stage_out_channels[1]
#         self.up3 = Up(c, self.stage_out_channels[1])
#         c = self.stage_out_channels[1] + self.stage_out_channels[0]
#         self.up4 = Up(c, self.stage_out_channels[0])
#         self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)
#         self.activation = nn.Hardsigmoid()

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         input_shape = x.shape[-2:]
#         backbone_out = self.backbone(x)
#         x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
#         x = self.up2(x, backbone_out['stage2'])
#         x = self.up3(x, backbone_out['stage1'])
#         x = self.up4(x, backbone_out['stage0'])
#         x = self.conv(x)
#         # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

#         return self.activation(x)

class MobileV3Unet(nn.Module):
    def __init__(self, num_classes: int = 3, pretrain_backbone: str = "MobileNet_V3_Large_Weights.IMAGENET1K_V2"):
        super(MobileV3Unet, self).__init__()
        backbone = mobilenet_v3_large(weights=pretrain_backbone)

        backbone = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.up5 = Up(self.stage_out_channels[0]+3, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)
        self.activation = nn.Hardsigmoid()

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = img.shape[-2:]
        backbone_out = self.backbone(img)
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.up5(x, img)
        x = self.conv(x)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return self.activation(x)
    
import torch.nn as nn
import torch.nn.functional as F
import torch

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class ConvNext(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, blocks, block_type="Block"):
		super(ConvNext, self).__init__()

		layers = [eval(block_type)(in_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(in_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class WMConvNext(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=4, channels=128):
        super(WMConvNext, self).__init__()

        self.down_layer = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(channels),
        )
        self.mid_layer = ConvNext(channels, blocks=blocks)
        self.up_layer = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.mix_layer = nn.Conv2d(int(channels) + 3, 3, kernel_size=7, padding=3, stride=1)
        self.activation = nn.Hardsigmoid()
        
    def forward(self, image):
        x = self.down_layer(image)
        x = self.mid_layer(x)
        x = self.up_layer(x)
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        return self.activation(output)
    
import torch.nn as nn
import torch.nn.functional as F
import torch

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class LNConv(nn.Module):
	def __init__(self, channels_in, channels_out, kernel=3, stride=1, padding = 0):
		super(LNConv, self).__init__()

		self.layers = nn.Sequential(
			LayerNorm(channels_in, eps=1e-6, data_format="channels_first"),
			nn.Conv2d(channels_in, channels_out, kernel, stride, padding=padding),
		)

	def forward(self, x):
		return self.layers(x)

class ConvNext(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, blocks, block_type="Block"):
		super(ConvNext, self).__init__()

		layers = [eval(block_type)(in_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(in_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class ConvUNext(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=2, channels=128, down_ratio = 2):
        super(ConvUNext, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1, stride=2),
                                   LayerNorm(128),
                                   ConvNext(channels, blocks=blocks))
        self.down2 = nn.Sequential(LNConv(channels, channels*2, kernel=4, stride=down_ratio*2),
                                   ConvNext(channels*2, blocks=blocks))
        self.down3 = nn.Sequential(LNConv(channels*2, channels*4, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*4, blocks=blocks))
        self.down4 = nn.Sequential(LNConv(channels*4, channels*8, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*8, blocks=blocks))
        
        self.up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=down_ratio)
        self.up2 = nn.ConvTranspose2d(channels*4, channels, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up3 = nn.ConvTranspose2d(channels*8, channels*2, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up4 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=down_ratio*2, stride=down_ratio*2)
        
        self.mix_layer = nn.Conv2d(int(channels) + 3, 3, kernel_size=7, padding=3, stride=1)
        self.activation = nn.Hardsigmoid()
        
    def forward(self, image):
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up4(x4)
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up1(torch.cat([x, x1], dim=1))
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        return self.activation(output)
    
class ConvUNext2(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=2, channels=128, down_ratio = 2):
        super(ConvUNext2, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1, stride=2),
                                   LayerNorm(128),
                                   ConvNext(channels, blocks=blocks))
        self.down2 = nn.Sequential(LNConv(channels, channels*2, kernel=4, stride=down_ratio*2),
                                   ConvNext(channels*2, blocks=blocks))
        self.down3 = nn.Sequential(LNConv(channels*2, channels*4, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*4, blocks=blocks))
        self.down4 = nn.Sequential(LNConv(channels*4, channels*8, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*8, blocks=blocks))
        
        self.up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=down_ratio)
        self.up2 = nn.ConvTranspose2d(channels*4, channels, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up3 = nn.ConvTranspose2d(channels*8, channels*2, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up4 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=down_ratio*2, stride=down_ratio*2)
        
        self.mix_layer = nn.Conv2d(int(channels) + 3, 3, kernel_size=7, padding=3, stride=1)
        
        self.final_conv = nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3)  # 7x7 convolutional layer
        self.activation = nn.Hardsigmoid()
        
    def forward(self, image, shuffler):
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up4(x4)
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up1(torch.cat([x, x1], dim=1))
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        
        image = shuffler.unshuffle(output)
        x = self.final_conv(image)
        return self.activation(x)
    
class ConvUNext3(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=2, channels=128, down_ratio = 2):
        super(ConvUNext3, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1, stride=2),
                                   LayerNorm(128),
                                   ConvNext(channels, blocks=blocks))
        self.down2 = nn.Sequential(LNConv(channels, channels*2, kernel=4, stride=down_ratio*2),
                                   ConvNext(channels*2, blocks=blocks))
        self.down3 = nn.Sequential(LNConv(channels*2, channels*4, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*4, blocks=blocks))
        self.down4 = nn.Sequential(LNConv(channels*4, channels*8, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*8, blocks=blocks))
        
        self.up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=down_ratio)
        self.up2 = nn.ConvTranspose2d(channels*4, channels, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up3 = nn.ConvTranspose2d(channels*8, channels*2, kernel_size=down_ratio*2, stride=down_ratio*2)
        self.up4 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=down_ratio*2, stride=down_ratio*2)
        
        self.mask = self.generate_mask(64)
        self.mix_layer = nn.Conv2d(int(channels) + 3, 3, kernel_size=7, padding=3, stride=1)
        
        self.activation = nn.Hardsigmoid()
        
    def generate_mask(self, block_size, tau=0.07):
        # Create coordinate grid
        x = torch.arange(0, block_size).float().unsqueeze(0).repeat(block_size, 1)
        y = torch.arange(0, block_size).float().unsqueeze(1).repeat(1, block_size)
        
        # Center coordinates
        center = block_size // 2
        
        # Compute distances to center
        dx = torch.abs(x - center)
        dy = torch.abs(y - center)
        
        # Get maximum of absolute distances
        max_dist = torch.max(dx, dy)
        
        # Normalize max_dist to [0, 1]``
        normalized_max_dist = max_dist / (block_size / 2)
        
        # Exponential mask
        mask = torch.exp(normalized_max_dist/tau)  # exp(0) = 1, so we subtract 1 to get 0 at the center
        mask = mask / mask.max()  # Normalize to [0, 1]
        mask = torch.where(mask < 0.01, torch.zeros_like(mask), mask)
        mask = 1 - mask.repeat(4, 4)
        
        return mask
        
    def forward(self, image):
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up4(x4)
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up1(torch.cat([x, x1], dim=1))
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        
        if self.mask.get_device() == "cpu" or self.mask.get_device() < 0:
            self.mask = self.mask.to(x.get_device())
        
        return self.activation(output) * self.mask + image * (1 - self.mask)
    
class ConvUNext4(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=2, channels=128, down_ratio = 2):
        super(ConvUNext4, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1, stride=2),
                                   LayerNorm(128),
                                   ConvNext(channels, blocks=blocks))
        self.down2 = nn.Sequential(LNConv(channels, channels*2, kernel=4, stride=down_ratio*2),
                                   ConvNext(channels*2, blocks=blocks))
        self.down3 = nn.Sequential(LNConv(channels*2, channels*4, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*4, blocks=blocks))
        self.down4 = nn.Sequential(LNConv(channels*4, channels*8, kernel=4, stride=down_ratio*2),
                                      ConvNext(channels*8, blocks=blocks))

        # Upsampling layers modified with nearest-neighbor and convolutions
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=down_ratio, mode='nearest'),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=down_ratio*2, mode='nearest'),
            nn.Conv2d(channels * 4, channels, kernel_size=3, padding=1)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=down_ratio*2, mode='nearest'),
            nn.Conv2d(channels * 8, channels * 2, kernel_size=3, padding=1)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=down_ratio*2, mode='nearest'),
            nn.Conv2d(channels * 8, channels * 4, kernel_size=3, padding=1)
        )
        
        self.mask = self.generate_mask(64)
        self.mix_layer = nn.Conv2d(int(channels) + 3, 3, kernel_size=7, padding=3, stride=1)
        
        self.activation = nn.Hardsigmoid()
        
    def generate_mask(self, block_size, tau=0.07):
        # Create coordinate grid
        x = torch.arange(0, block_size).float().unsqueeze(0).repeat(block_size, 1)
        y = torch.arange(0, block_size).float().unsqueeze(1).repeat(1, block_size)
        
        # Center coordinates
        center = block_size // 2
        
        # Compute distances to center
        dx = torch.abs(x - center)
        dy = torch.abs(y - center)
        
        # Get maximum of absolute distances
        max_dist = torch.max(dx, dy)
        
        # Normalize max_dist to [0, 1]``
        normalized_max_dist = max_dist / (block_size / 2)
        
        # Exponential mask
        mask = torch.exp(normalized_max_dist/tau)  # exp(0) = 1, so we subtract 1 to get 0 at the center
        mask = mask / mask.max()  # Normalize to [0, 1]
        mask = torch.where(mask < 0.01, torch.zeros_like(mask), mask)
        mask = 1 - mask.repeat(4, 4)
        
        return mask
        
    def forward(self, image):
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up4(x4)
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up1(torch.cat([x, x1], dim=1))
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        
        if self.mask.get_device() == "cpu" or self.mask.get_device() < 0:
            self.mask = self.mask.to(x.get_device())
        
        return self.activation(output) * self.mask + image * (1 - self.mask)
    
class ConvUNext_atto(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=2, channels=40, down_ratio=2):
        super(ConvUNext_atto, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, channels, kernel_size=4, stride=4),
                                   LayerNorm(64),
                                   ConvNext(channels, blocks=blocks))
        self.down2 = nn.Sequential(LNConv(channels, channels*2, kernel=2, stride=down_ratio),
                                   ConvNext(channels*2, blocks=blocks))
        self.down3 = nn.Sequential(LNConv(channels*2, channels*4, kernel=2, stride=down_ratio),
                                   ConvNext(channels*4, blocks=blocks*3))
        self.down4 = nn.Sequential(LNConv(channels*4, channels*8, kernel=2, stride=down_ratio),
                                   ConvNext(channels*8, blocks=blocks))
        
        # Use nn.Conv2d instead of nn.ConvTranspose2d for the upsample layers
        self.up_conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, padding=1)
        self.up_conv2 = nn.Conv2d(channels*6, channels*2, kernel_size=3, padding=1)
        self.up_conv3 = nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1)
        self.up_conv4 = nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1)
        
        self.mix_layer = nn.Conv2d(int(channels), 3, kernel_size=7, padding=3, stride=1)
        self.activation = nn.Hardsigmoid()
        
    def forward(self, image):
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Interpolate and apply convolution instead of using ConvTranspose2d
        x = F.interpolate(x4, scale_factor=2, mode='nearest')
        x = self.up_conv4(x)
        x = F.interpolate(torch.cat([x, x3], dim=1), scale_factor=2, mode='nearest')
        x = self.up_conv3(x)
        x = F.interpolate(torch.cat([x, x2], dim=1), scale_factor=2, mode='nearest')
        x = self.up_conv2(x)
        x = F.interpolate(torch.cat([x, x1], dim=1), scale_factor=2, mode='nearest')
        x = self.up_conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(x)
        return output
    
class ConvUNext_atto_r(nn.Module):
    '''
    Insert a watermark into an image
    '''

    def __init__(self, blocks=2, channels=40, down_ratio=2):
        super(ConvUNext_atto_r, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, channels, kernel_size=4, stride=4),
                                   LayerNorm(64),
                                   ConvNext(channels, blocks=blocks))
        self.down2 = nn.Sequential(LNConv(channels, channels*2, kernel=2, stride=down_ratio),
                                   ConvNext(channels*2, blocks=blocks))
        self.down3 = nn.Sequential(LNConv(channels*2, channels*4, kernel=2, stride=down_ratio),
                                   ConvNext(channels*4, blocks=blocks*3))
        self.down4 = nn.Sequential(LNConv(channels*4, channels*8, kernel=2, stride=down_ratio),
                                   ConvNext(channels*8, blocks=blocks))
        
        # Use nn.Conv2d instead of nn.ConvTranspose2d for the upsample layers
        self.up_conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, padding=1)
        self.up_conv2 = nn.Conv2d(channels*6, channels*2, kernel_size=3, padding=1)
        self.up_conv3 = nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1)
        self.up_conv4 = nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1)
        
        self.mix_layer = nn.Conv2d(int(channels)+3, 3, kernel_size=7, padding=3, stride=1)
        self.activation = nn.Hardsigmoid()
        
    def forward(self, image):
        x1 = self.down1(image)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Interpolate and apply convolution instead of using ConvTranspose2d
        x = F.interpolate(x4, scale_factor=2, mode='nearest')
        x = self.up_conv4(x)
        x = F.interpolate(torch.cat([x, x3], dim=1), scale_factor=2, mode='nearest')
        x = self.up_conv3(x)
        x = F.interpolate(torch.cat([x, x2], dim=1), scale_factor=2, mode='nearest')
        x = self.up_conv2(x)
        x = F.interpolate(torch.cat([x, x1], dim=1), scale_factor=2, mode='nearest')
        x = self.up_conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        concat = torch.cat([x, image], dim=1)
        output = self.mix_layer(concat)
        return self.activation(output)