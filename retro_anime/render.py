"""전체 렌더러: 장면 → 필름 후처리 → ffmpeg(H.264) + 사운드트랙 합성.

사용법 (저장소 루트에서):
    python -m retro_anime.render                      # 전체 15초, output/zanzou.mp4
    python -m retro_anime.render --start 72 --end 144 --out /tmp/clip.mp4 --no-audio
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .core import FPS, H, W, Ctx, to_uint8

HERE = Path(__file__).resolve().parent


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def render_frame(frame: int, post: bool = True) -> np.ndarray:
    """전체 프레임 번호 → float32 (H,W,3) 0..1."""
    from . import post as postfx
    from .scenes import locate
    mod, lf = locate(frame)
    ctx = Ctx(frame=frame, lframe=lf, nframes=mod.FRAMES)
    img = mod.render(ctx)
    if img.shape != (H, W, 3):
        raise ValueError(f"{mod.NAME}: 잘못된 출력 크기 {img.shape}")
    img = img.astype(np.float32)
    if post:
        img = postfx.film(img, frame, getattr(mod, "FILM", None))
    else:
        img = np.clip(img, 0, 1)
    return img


def _job(args):
    frame, post = args
    return frame, to_uint8(render_frame(frame, post)).tobytes()


def main(argv=None):
    from .scenes import total_frames
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "output" / "zanzou.mp4"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)

    end = total_frames() if a.end is None else a.end
    frames = list(range(a.start, end))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    wav = None
    if not a.no_audio:
        from .audio import render_audio
        wav = str(Path(a.out).with_suffix(".wav"))
        render_audio(wav, start_frame=a.start, end_frame=end)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if wav:
        cmd += ["-i", wav]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-tune", "grain",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if wav:
        cmd += ["-c:a", "aac", "-b:a", "256k", "-shortest"]
    cmd += [a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0 = time.time()
    with Pool(a.jobs, initializer=_init_worker) as pool:
        for i, (f, buf) in enumerate(pool.imap(_job, [(f, not a.no_post) for f in frames], chunksize=1)):
            proc.stdin.write(buf)
            if i % 12 == 0 or i == len(frames) - 1:
                el = time.time() - t0
                print(f"\r프레임 {f + 1}/{end}  ({el:.0f}s, {el / (i + 1):.2f}s/f)", end="", file=sys.stderr)
    proc.stdin.close()
    proc.wait()
    print(f"\n완료: {a.out}", file=sys.stderr)
    if wav and os.path.exists(wav):
        os.remove(wav)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
