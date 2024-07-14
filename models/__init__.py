from .EEVG import EEVG
from .visual_model import build_SwinT, SwinTransformer

def build_model(args):
    if args.model_name == "EEVG":
        return EEVG(args)