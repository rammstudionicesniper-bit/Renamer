"""프로시저럴 사운드트랙 (임시: 무음). render_audio(path, start_frame, end_frame)."""
import wave

import numpy as np

SR = 48000


def render_audio(path, start_frame=0, end_frame=360, fps=24):
    n = int((end_frame - start_frame) / fps * SR)
    data = np.zeros((n, 2), np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
