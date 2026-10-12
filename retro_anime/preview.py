"""장면 미리보기: 지정 프레임을 렌더해 JPG 와 컨택트 시트를 만든다.

    python -m retro_anime.preview s2_chase                    # 장면 전체 6프레임 간격
    python -m retro_anime.preview s2_chase --frames 0,10,20   # 장면 내부 프레임 지정
    python -m retro_anime.preview s2_chase --every 2 --raw    # 후처리 없이
    python -m retro_anime.preview s2_chase --strip 30:38      # 연속 프레임 필름스트립(동작 확인)
결과: <out>/<scene>_f###.jpg (풀 해상도) 와 <out>/<scene>_sheet.jpg
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .core import to_uint8


def main(argv=None):
    from .render import render_frame
    from .scenes import BY_NAME, starts, TIMELINE
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--frames", default=None, help="장면 내부 프레임 목록 (쉼표)")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--strip", default=None, help="a:b 연속 프레임")
    ap.add_argument("--raw", action="store_true", help="필름 후처리 없이")
    ap.add_argument("--out", default="/tmp/claude-0/-home-user-Renamer/62b663c8-a0c3-5913-95f3-aaaa66f0c362/scratchpad/preview")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--no-full", action="store_true", help="풀 해상도 JPG 저장 생략")
    a = ap.parse_args(argv)

    mod = BY_NAME[a.scene]
    base = starts()[TIMELINE.index(mod)]
    if a.strip:
        s, e = (int(x) for x in a.strip.split(":"))
        lfs = list(range(s, e))
    elif a.frames:
        lfs = [int(x) for x in a.frames.split(",")]
    else:
        lfs = list(range(0, mod.FRAMES, a.every))
        if lfs[-1] != mod.FRAMES - 1:
            lfs.append(mod.FRAMES - 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    tw, th = 640, 360
    for lf in lfs:
        t0 = time.time()
        img = render_frame(base + lf, post=not a.raw)
        dt = time.time() - t0
        bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
        if not a.no_full:
            cv2.imwrite(str(out / f"{a.scene}_f{lf:03d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        t = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{lf}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, f"{lf}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(t)
        print(f"{a.scene} f{lf:03d}  {dt:.2f}s")
    cols = min(a.cols, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    name = f"{a.scene}_strip.jpg" if a.strip else f"{a.scene}_sheet.jpg"
    cv2.imwrite(str(out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"시트: {out / name}")


if __name__ == "__main__":
    main()
