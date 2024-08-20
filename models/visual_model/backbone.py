# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor

from .position_encoding import build_position_encoding
from .SwinT import build_SwinT
from .ViTDet import build_ViTDet

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, name: str, backbone: nn.Module, num_channels: int):
        super().__init__()
        if name == "SwinT" or name == "SwinT-S" or name == "ViTDet":
            self.body = backbone
        self.num_channels = num_channels
        if name == "SwinT" or name == "SwinT-S":
            self.size = (backbone.length, backbone.length)
        elif name == "ViTDet":
            self.size = (28, 28)
        else:
            raise NotImplementedError

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=self.size).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str):
        assert name in ('SwinT', 'ViTDet', 'SwinT-S')
        if name == 'SwinT':
            backbone = build_SwinT(name)
            ckpt_path = "./checkpoints/swin_base_patch4_window12_384_22k.pth"
            backbone.init_weights(pretrained=ckpt_path)
        elif name == 'SwinT-S':
            backbone = build_SwinT(name)
            ckpt_path = "./checkpoints/swin_small_patch4_window7_224_22k.pth"
            backbone.init_weights(pretrained=ckpt_path)
        elif name == 'ViTDet':
            backbone = build_ViTDet()
            import pickle as pkl
            with open("./checkpoints/model_final_435fa9.pkl", 'rb') as f:
                info_dict = pkl.load(f)

            new_dict = {}
            for k, v in info_dict['model'].items():
                if 'backbone.net.' in k:
                    k = k.replace('backbone.net.', '')
                new_dict[k] = torch.from_numpy(v)

            backbone.load_state_dict(new_dict, strict=False)
        if name == "SwinT":
            num_channels = 512
        elif name == "SwinT-S":
            num_channels = 384
        elif name == "ViTDet":
            num_channels = 768
        else:
            raise NotImplementedError
        super().__init__(name, backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
