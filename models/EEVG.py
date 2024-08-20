import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_model.visual_backbone import build_visual
from .language_model.bert import build_bert
from torch import Tensor
from typing import Optional

PATCH_LEN, ELIMINATED_THRESHOLD = None, 0.015
PATCH_LEN_DICT = {
    "SwinT": 32,
    "SwinT-S": 32,
    "ViTDet": 28,
    "DarkNet53": 20,
    "ResNet101": 20
}

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class EEVG(nn.Module):
    def __init__(self, args):
        super(EEVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        self.num_visu_token = 32 ** 2 if "SwinT" in args.backbone else 28 ** 2
        self.num_text_token = args.max_query_len
        global PATCH_LEN
        PATCH_LEN = PATCH_LEN_DICT[args.backbone]

        self.visumodel = build_visual(args)
        self.textmodel = build_bert(args)
        self.is_segment = args.is_segment

        print('is_segment', self.is_segment)

        self.patch_length = 32 if "SwinT" in args.backbone else 28
        self.imsize = args.imsize

        self.reg_token = nn.Embedding(1, hidden_dim)
        self.visual_pos = nn.Embedding(self.patch_length ** 2 + 1, hidden_dim)
        self.text_pos = nn.Embedding(args.max_query_len, hidden_dim)
        decoder = dict(
            num_layers=args.vl_enc_layers,
            layer=dict(
                d_model=hidden_dim, nhead=8, dim_feedforward=args.dim_feedforward, dropout=0.1, activation='relu'),
            is_eliminate=args.is_eliminate,
        )
        print("dim_feedforward", args.dim_feedforward)

        from .decoder import TransformerDecoder
        self.decoder = TransformerDecoder(decoder, patch_length=self.patch_length, eliminated_threshold=args.eliminated_threshold)

        if self.is_segment:
            self.mask_head = MLP(hidden_dim, hidden_dim, 256, 2)
            self.mask_cnn = nn.Conv2d(1, 1, 5, padding=2)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        # visual backbone
        visu_mask, visu_src = self.visumodel(img_data)
        visu_src = self.visu_proj(visu_src)  # (N*B)xC
        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        text_pos = self.text_pos.weight.unsqueeze(1).expand(-1, bs, -1)
        visual_pos = self.visual_pos.weight.unsqueeze(1).expand(-1, bs, -1)
        tgt_src = self.reg_token.weight.unsqueeze(1).expand(-1, bs, -1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        visu_src = torch.cat([tgt_src, visu_src], dim=0)
        visu_mask = torch.cat([tgt_mask, visu_mask], dim=1)

        if self.is_segment:
            tgt, origin_idx = self.decoder(tgt=visu_src, memory=text_src,
                                           tgt_key_padding_mask=visu_mask,
                                           memory_key_padding_mask=text_mask, pos=text_pos,
                                           query_pos=visual_pos)
            tgt_visu = tgt[1:]

            mask_output = self.mask_head(tgt_visu).permute(1, 0, 2).contiguous()
            origin_idx = origin_idx[:, 1:].unsqueeze(-1)
            origin_idx -= 1
            origin_idx = origin_idx.expand(-1, -1, mask_output.shape[-1]).contiguous()
            tmp_mask_arr = torch.zeros((bs, self.patch_length ** 2, mask_output.shape[-1]),
                                       dtype=mask_output.dtype,
                                       device=mask_output.device)
            mask_output = tmp_mask_arr.scatter(1, origin_idx, mask_output)
            mask_output = mask_output.reshape(bs, self.patch_length, self.patch_length, 16, 16)
            mask_output = mask_output.permute(0, 1, 3, 2, 4).reshape(bs, 1, self.imsize,
                                                                     self.imsize).contiguous()
            mask_output = self.mask_cnn(mask_output).sigmoid()
        else:
            tgt = self.decoder(tgt=visu_src, memory=text_src,
                               tgt_key_padding_mask=visu_mask,
                               memory_key_padding_mask=text_mask, pos=text_pos,
                               query_pos=visual_pos)

        pred_box = self.bbox_embed(tgt[0]).sigmoid()
        if self.is_segment:
            return pred_box, mask_output
        else:
            return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
