import torch
from torch import nn

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone

class VisualBackbone(nn.Module):

    def __init__(self, backbone, train_backbone, backbone_name):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name

        hidden_dim = backbone.num_channels

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.num_channels = hidden_dim

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        if 'SwinT' not in self.backbone_name and self.backbone_name != 'ViTDet':
            out = [mask.flatten(1), src.flatten(2).permute(2, 0, 1)]
        else:
            out = [mask.flatten(1), src.permute(1, 0, 2)]

        return out


def build_visual(args):
    backbone = build_backbone(args)
    train_backbone = args.lr_visual > 0

    model = VisualBackbone(
        backbone,
        train_backbone=train_backbone,
        backbone_name=args.backbone,
    )
    return model
