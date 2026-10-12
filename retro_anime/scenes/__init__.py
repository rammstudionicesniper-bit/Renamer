"""장면 타임라인. 각 장면 모듈은 NAME, FRAMES, render(ctx) 를 제공한다."""
from . import s1_city, s2_chase, s3_eye, s4_mirror, s5_blast, s6_title

TIMELINE = [s1_city, s2_chase, s3_eye, s4_mirror, s5_blast, s6_title]
BY_NAME = {m.NAME: m for m in TIMELINE}


def starts():
    """각 장면의 시작 프레임."""
    out, acc = [], 0
    for m in TIMELINE:
        out.append(acc)
        acc += m.FRAMES
    return out


def total_frames():
    return sum(m.FRAMES for m in TIMELINE)


def locate(frame):
    """전체 프레임 → (모듈, 장면 내부 프레임)."""
    acc = 0
    for m in TIMELINE:
        if frame < acc + m.FRAMES:
            return m, frame - acc
        acc += m.FRAMES
    m = TIMELINE[-1]
    return m, m.FRAMES - 1
