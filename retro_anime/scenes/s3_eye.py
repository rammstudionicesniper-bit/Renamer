"""눈 초접사 (임시 플레이스홀더)."""
import numpy as np

from ..core import Ctx, blank, fill, place_mask, text_mask, vgradient

NAME = "s3_eye"
FRAMES = 60


def render(ctx: Ctx) -> np.ndarray:
    img = vgradient([(0, (0.05, 0.05, 0.12)), (1, (0.35, 0.1, 0.3))])
    fill(img, place_mask(text_mask(NAME + " " + str(ctx.lframe), 90), 960, 540), (1, 1, 1))
    return img
