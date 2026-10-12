"""눈 초접사 (ECU) — 퍼펙트 블루 톤의 심리 컷.

레이어 (뒤 → 앞)
  1. 피부 플레이트 (정적 캐시, 기준 좌표): 셀 그림자(눈썹 아래/콧등/광대) + 에어브러시 조명 맵(핑크/블루)
  2. 안구 버퍼: 과슈풍 흰자 → 페인트 홍채(동공 크기에 따른 방사 리맵) → 홍채 셀 톤 → 동공
     → 각막 반사(어안 리맵된 '반사 플레이트': 네온 간판/무대 조명/고속도로 불빛/실루엣)
     → 윗눈꺼풀 그림자 → 셀 하이라이트(창문형 캐치라이트)
  3. 쌍꺼풀 그림자/선, 두꺼운 윗속눈썹 띠(톱니 + 뭉친 속눈썹 + 날개), 아랫속눈썹
  4. 눈물 메니스커스(굴절 + 투과광 하이라이트, 반짝임)
  5. 전경 머리카락 셀 + 드리운 그림자, HDR 글로우, 임팩트(반전 듀오톤/줌 스미어)
카메라: T.U.(푸시인)·셰이크는 1코마, 작화(눈꺼풀/떨림/동공/선 떨림)는 2코마.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from ..core import (H, W, Ctx, add_glow, blur, catmull, ease_in_out, ease_out, fbm, fill_ellipse,
                    fill_poly, hash01, hexc, ink, jitter, luma, noise, noise_tex, poly_mask,
                    smoothstep, zoom_blur)

NAME = "s3_eye"
FRAMES = 60

# 사운드 싱크 포인트 (장면 내부 프레임)
EVENTS = [
    (2, "blink"),        # 눈꺼풀이 열리기 시작
    (3, "pulse"),        # 핑크 네온이 깜빡이며 켜짐
    (10, "passby"),       # 헤드라이트 빛이 얼굴을 스침
    (20, "impact"),      # 동공 급수축 (조크) + 눈을 크게 뜸
    (20, "flash"),       # 반전 1프레임
    (21, "shockwave"),   # 홍채 섬유를 타고 번지는 물결
    (26, "tear"),        # 눈물이 차오르기 시작
    (30, "heartbeat"),
    (36, "spark"),       # 눈물 반짝임
    (45, "swell"),       # 반사 속 실루엣이 떠오름
    (46, "heartbeat"),
    (53, "heartbeat"),
    (54, "spark"),
]

# ---------------------------------------------------------------------------
# 색
# ---------------------------------------------------------------------------
INK = hexc("#0a0510")
SKIN = hexc("#f2d2c4")
SKIN_SH = hexc("#c08a8a")
PINK = hexc("#ff4f9a")
BLUE = hexc("#3a6cff")
CYAN = hexc("#3ff0ff")
ORANGE = hexc("#ff8a2a")
AMB = np.array([0.10, 0.08, 0.20], np.float32)          # 어두운 남보라 환경광
SCLERA = np.array([0.90, 0.91, 0.97], np.float32)
HI = np.array([1.38, 1.36, 1.45], np.float32)            # 셀 하이라이트(살짝 HDR)

# ---------------------------------------------------------------------------
# 기준 좌표계(카메라 배율 1) 기하
# ---------------------------------------------------------------------------
NS = 181
S_LIN = np.linspace(0.0, 1.0, NS)
IRIS_C = (965.0, 600.0)
IRIS_RX, IRIS_RY = 286.0, 298.0

UPPER_OPEN_CP = [(236, 640), (300, 574), (400, 494), (540, 420), (720, 370), (930, 350),
                 (1150, 354), (1360, 382), (1540, 436), (1660, 490), (1738, 540)]
LOWER_CP = [(236, 640), (330, 708), (470, 774), (650, 826), (860, 856), (1080, 858),
            (1300, 830), (1480, 766), (1630, 668), (1738, 540)]
CREASE_CP = [(310, 488), (430, 386), (600, 290), (800, 230), (1020, 206), (1240, 212),
             (1440, 246), (1600, 302), (1712, 378)]

BROW_SH = [(-60, -60), (1980, -60), (1980, 104), (1800, 112), (1600, 128), (1380, 136), (1180, 122),
           (1000, 118), (820, 126), (640, 146), (470, 186), (330, 242), (180, 322), (-60, 430)]
NOSE_SH = [(-60, 430), (180, 322), (198, 404), (206, 520), (194, 664), (166, 806), (126, 944),
           (78, 1140), (-60, 1140)]
CHEEK_SH = [(1180, 1140), (1360, 1034), (1560, 962), (1760, 922), (1980, 906), (1980, 1140)]
# 네온 림라이트 셀 (그림자 경계를 스치는 딱딱한 빛 띠)
RIM_BROW = catmull([(150, 338), (330, 244), (470, 188), (640, 148), (820, 128), (1000, 120), (1120, 122)], 12)
RIM_NOSE = catmull([(182, 330), (200, 410), (208, 520), (196, 664), (170, 800)], 12)
RIM_CHEEK = catmull([(1300, 1080), (1400, 1020), (1560, 964), (1760, 924), (1990, 908)], 12)


def _resample(P, n=NS):
    P = np.asarray(P, np.float64)
    seg = np.hypot(*np.diff(P, axis=0).T)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, L[-1], n)
    return np.stack([np.interp(t, L, P[:, 0]), np.interp(t, L, P[:, 1])], 1)


UPPER_OPEN = _resample(catmull(UPPER_OPEN_CP, 48))
LOWER = _resample(catmull(LOWER_CP, 48))
UPPER_CLOSED = LOWER + np.stack([np.zeros(NS), 7 * np.sin(np.pi * S_LIN)], 1)
CREASE = _resample(catmull(CREASE_CP, 48))


def _normals(P):
    t = np.gradient(np.asarray(P, np.float64), axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def _tangents(P):
    t = np.gradient(np.asarray(P, np.float64), axis=0)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


def _M(cam, P):
    s, tx, ty = cam
    return np.asarray(P, np.float64) * s + (tx, ty)


def _Mp(cam, x, y):
    s, tx, ty = cam
    return x * s + tx, y * s + ty


def _band(P, d0, d1, n=None):
    """곡선 P 를 따라 법선 방향 오프셋 d0 → d1 사이의 띠 다각형."""
    n = _normals(P) if n is None else n
    a = P + n * np.asarray(d0, np.float64).reshape(-1, 1)
    b = P + n * np.asarray(d1, np.float64).reshape(-1, 1)
    return np.vstack([a, b[::-1]])


def _superellipse(cx, cy, rx, ry, ang_deg, p=4.0, n=40):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s_ = np.cos(th), np.sin(th)
    x = rx * np.sign(c) * np.abs(c) ** (2 / p)
    y = ry * np.sign(s_) * np.abs(s_) ** (2 / p)
    a = math.radians(ang_deg)
    return np.stack([cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)], 1)


# ---------------------------------------------------------------------------
# 정적 플레이트 (기준 좌표, 프로세스당 1회)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _plates():
    f32 = np.float32
    # 조명 맵: 핑크(왼쪽 위) / 블루(오른쪽 아래) — 부드러우므로 1/4 해상도로 계산 후 확대
    q = 4
    wq, hq = W // q, H // q
    xq = (np.arange(wq, dtype=f32)[None, :] + 0.5) * q
    yq = (np.arange(hq, dtype=f32)[:, None] + 0.5) * q
    n1 = fbm(wq, hq, scale=260 / q, seed=31, octaves=2) - 0.5
    n2 = noise(wq, hq, scale=90 / q, seed=37, beta=1.6, sx=5.0, sy=1.0, angle=0.5) - 0.5
    P = 1.0 - smoothstep(240.0, 1400.0, xq + 0.42 * (yq - 540) + n1 * 260)
    B = smoothstep(660.0, 1660.0, xq - 0.30 * (yq - 540) + n1 * 220)
    P = np.clip(P * (1 + n2 * 0.18), 0, 1.1) ** 1.15
    B = np.clip(B * (1 - n2 * 0.15), 0, 1.1)
    P = cv2.resize(P.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(B.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR)
    # 셀 그림자 (딱딱한 경계)
    sh = poly_mask([np.array(BROW_SH, np.float64), np.array(NOSE_SH, np.float64),
                    np.array(CHEEK_SH, np.float64)])
    deep = poly_mask(np.array([(-60, -60), (1980, -60), (1980, 48), (1600, 66), (1200, 72),
                               (800, 80), (500, 116), (260, 180), (-60, 300)], np.float64))
    # 눈 밑 얇은 그림자 (아랫눈꺼풀 두께)
    Ln = _normals(LOWER)
    j0, j1 = 60, 176
    und = poly_mask(_band(LOWER[j0:j1], 10 + 0 * S_LIN[j0:j1],
                          10 + 16 * np.sin(np.pi * (S_LIN[j0:j1] - S_LIN[j0]) / (S_LIN[j1 - 1] - S_LIN[j0])) ** 0.8,
                          Ln[j0:j1]))
    sh_all = np.maximum(sh, und * 0.6)
    lscale = 1.0 - 0.78 * sh_all - 0.15 * deep
    # 알베도 (셀: 평평한 베이스 + 그림자 톤) + 아주 약한 포스터컬러 얼룩
    paint = 1.0 + 0.045 * ((noise(W, H, 6.0, 0, 0, 41, beta=1.6, sx=6.0, sy=1.0, angle=0.3) - 0.5) * 2.2
                           + (noise(W, H, 24.0, 0, 0, 48, beta=2.0) - 0.5) * 1.4)
    xs = np.arange(W, dtype=f32)
    ys = np.arange(H, dtype=f32)

    def gauss2(cx, cy, sx, sy):     # 분리 가능한 에어브러시 가우시안
        return np.exp(-((ys - cy) / sy) ** 2)[:, None] * np.exp(-((xs - cx) / sx) ** 2)[None, :]

    blush = gauss2(640, 1060, 420.0, 150.0)
    bone = gauss2(980, 190, 520.0, 46.0) * (1 - sh)
    socket = gauss2(520, 470, 260.0, 120.0)
    k = (1.0 - 0.35 * deep) * paint * (1 + 0.16 * bone) * (1 - 0.18 * socket)
    alb = np.empty((H, W, 3), f32)
    for c in range(3):
        alb[..., c] = (SKIN[c] + (SKIN_SH[c] - SKIN[c]) * sh_all) * k
    alb[..., 1] *= 1 - 0.18 * blush
    alb[..., 2] *= 1 - 0.08 * blush
    # 흰자: 둥근 안구 음영(가장자리로 갈수록 어둡게) + 희미한 실핏줄
    dx2 = ((xs - IRIS_C[0]) / 820.0) ** 2
    dy2 = ((ys - IRIS_C[1]) / 420.0) ** 2
    rnd = np.clip(1.0 - 0.80 * (dx2 ** 1.1)[None, :] - 0.40 * dy2[:, None], 0.22, 1.0)
    rnd *= 1.0 + 0.03 * (noise(W, H, 10.0, 0, 0, 37, beta=1.6) - 0.5) * 2
    veins = np.zeros((H, W), f32)
    rng = np.random.default_rng(5150)
    for side, x0 in ((-1, 300), (1, 1690)):
        for _ in range(7):
            p = np.array([x0 + rng.uniform(-30, 30), rng.uniform(580, 770)])
            a = math.pi * (0 if side < 0 else 1) + rng.uniform(-0.6, 0.6)
            pts = [p.copy()]
            for _ in range(int(rng.integers(6, 12))):
                a += rng.normal(0, 0.45)
                p = p + np.array([math.cos(a), math.sin(a)]) * -side * rng.uniform(14, 26)
                pts.append(p.copy())
            cv2.polylines(veins, [np.round(np.array(pts) * 16).astype(np.int32)], False, 1.0,
                          1, cv2.LINE_AA, 4)
    veins = cv2.GaussianBlur(veins, (0, 0), 0.8) * 0.32
    scl = np.empty((H, W, 3), f32)
    scl[..., 0] = SCLERA[0] * rnd
    scl[..., 1] = SCLERA[1] * rnd * (1 - veins)
    scl[..., 2] = SCLERA[2] * rnd * (1 - veins * 0.8)
    stack = cv2.merge([P, B, lscale.astype(f32)])
    return stack, alb, scl


# ---------------------------------------------------------------------------
# 페인트 홍채 텍스처 (정규화 좌표: 반지름 1 = 홍채 가장자리, 기준 동공 P0)
# ---------------------------------------------------------------------------
TEX_N = 1024
TEX_EXT = 1.08
P0 = 0.30


@lru_cache(maxsize=1)
def _iris_tex():
    N = TEX_N
    lin = ((np.arange(N, dtype=np.float32) + 0.5) / N * 2 - 1) * TEX_EXT
    u = np.broadcast_to(lin[None, :], (N, N))
    v = np.broadcast_to(lin[:, None], (N, N))
    r = np.hypot(u, v).astype(np.float32)
    a01 = ((np.arctan2(v, u) / (2 * np.pi)) % 1.0).astype(np.float32)

    def pn(seed, beta, rs, k=1):
        tex = noise_tex(seed, 1024, beta)
        mx = ((a01 * 1024 * k) % 1024).astype(np.float32)
        my = ((r * rs + seed * 3.7) % 1024).astype(np.float32)
        return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    stops = [(0.00, (0.02, 0.01, 0.03)), (P0, (0.04, 0.015, 0.05)), (P0 + 0.02, (0.26, 0.07, 0.13)),
             (0.40, (0.58, 0.22, 0.30)), (0.50, (0.78, 0.40, 0.46)), (0.555, (0.66, 0.33, 0.50)),
             (0.60, (0.36, 0.22, 0.62)), (0.78, (0.27, 0.20, 0.64)), (0.88, (0.15, 0.10, 0.40)),
             (0.97, (0.04, 0.025, 0.11)), (1.10, (0.03, 0.02, 0.08))]
    pos = np.array([p for p, _ in stops], np.float32)
    col = np.array([c for _, c in stops], np.float32)
    img = np.stack([np.interp(r, pos, col[:, c]) for c in range(3)], -1).astype(np.float32)

    zone = smoothstep(P0, P0 + 0.04, r) * (1 - smoothstep(0.92, 0.99, r))
    f1 = pn(301, 1.0, 34) - 0.5
    f2 = pn(302, 1.7, 14) - 0.5
    fib = f1 * 2.4 + f2 * 1.8
    img *= (1 + 0.30 * fib * zone)[..., None]
    # 칼라렛(들쭉날쭉한 고리)
    rc = 0.52 + (pn(303, 2.2, 4) - 0.5) * 0.18
    ring = np.exp(-((r - rc) / 0.022) ** 2)
    img += ring[..., None] * np.array([0.30, 0.14, 0.20], np.float32)
    # 크립트(움푹한 어두운 반점)
    c = pn(304, 1.5, 26)
    crypt = smoothstep(0.64, 0.70, c) * smoothstep(0.56, 0.62, r) * (1 - smoothstep(0.82, 0.88, r))
    img *= (1 - 0.55 * crypt)[..., None]
    # 수축 주름(끊긴 동심원)
    br = pn(305, 1.3, 2)
    for rr in (0.71, 0.80):
        img *= (1 - 0.35 * np.exp(-((r - rr) / 0.008) ** 2) * (br > 0.5))[..., None]
    # 붓으로 그은 방사 스트로크 (밝은/어두운)
    strokes = np.zeros((N, N), np.float32)
    dark = np.zeros((N, N), np.float32)
    rng = np.random.default_rng(777)
    cx = cy = N / 2
    sc = N / 2 / TEX_EXT
    for i in range(230):
        a = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0.56, 0.62) if i % 3 else rng.uniform(P0 + 0.03, 0.40)
        r1 = r0 + rng.uniform(0.12, 0.34)
        da = rng.uniform(0.004, 0.011)
        bend = rng.uniform(-0.05, 0.05)
        pts = [(cx + math.cos(a - da) * r0 * sc, cy + math.sin(a - da) * r0 * sc),
               (cx + math.cos(a + bend) * r1 * sc, cy + math.sin(a + bend) * r1 * sc),
               (cx + math.cos(a + da) * r0 * sc, cy + math.sin(a + da) * r0 * sc)]
        tgt = strokes if rng.random() < 0.62 else dark
        cv2.fillPoly(tgt, [np.round(np.array(pts) * 16).astype(np.int32)],
                     float(rng.uniform(0.5, 1.0)), cv2.LINE_AA, 4)
    strokes = cv2.GaussianBlur(strokes, (0, 0), 0.9)
    dark = cv2.GaussianBlur(dark, (0, 0), 0.9)
    img += strokes[..., None] * np.array([0.20, 0.13, 0.26], np.float32) * zone[..., None]
    img *= (1 - 0.45 * dark * zone)[..., None]
    # 세로 그라데이션: 위는 어둡게, 아래는 밝게(아래쪽에서 반사되는 빛)
    vn = v / np.maximum(r, 1e-3) * np.minimum(r, 1.0)
    vf = np.interp(vn, [-1, -0.2, 0.3, 1], [0.50, 0.90, 1.06, 1.30])
    img *= vf[..., None].astype(np.float32)
    return np.clip(img, 0, 2).astype(np.float32), fib.astype(np.float32)


def _draw_iris(eb, cam, cx, cy, p, tint, wave=None):
    """홍채 페인트 텍스처를 동공 크기 p 에 맞춰 방사 리맵하여 eb 에 합성.
    wave=(r0, amp): 동공 수축 순간 섬유를 타고 바깥으로 번지는 물결."""
    s = cam[0]
    rx, ry = IRIS_RX * s, IRIS_RY * s
    x0, x1 = int(max(0, cx - rx - 3)), int(min(W, cx + rx + 4))
    y0, y1 = int(max(0, cy - ry - 3)), int(min(H, cy + ry + 4))
    if x1 <= x0 or y1 <= y0:
        return
    tex, _ = _iris_tex()
    u = ((np.arange(x0, x1, dtype=np.float32) + 0.5 - cx) / rx)[None, :]
    v = ((np.arange(y0, y1, dtype=np.float32) + 0.5 - cy) / ry)[:, None]
    r = np.sqrt(u * u + v * v)
    rt = np.where(r < p, r * (P0 / p), P0 + (r - p) * (1 - P0) / (1 - p))
    ring = None
    if wave is not None:
        r0, amp = wave
        env = np.exp(-((r - r0) / 0.10) ** 2)
        rt = rt + amp * np.sin((r - r0) * 46.0) * env
        ring = np.exp(-((r - r0) / 0.035) ** 2)
    k = rt / np.maximum(r, 1e-5)
    mx = ((u * k / TEX_EXT + 1) * (TEX_N / 2) - 0.5).astype(np.float32)
    my = ((v * k / TEX_EXT + 1) * (TEX_N / 2) - 0.5).astype(np.float32)
    rgb = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rgb *= tint[y0:y1, x0:x1]
    if ring is not None:
        rgb += (ring * (r > p))[..., None] * rgb * np.array([0.8, 1.6, 2.4], np.float32) * (amp / 0.03)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(m, (int(round((cx - x0) * 16)), int(round((cy - y0) * 16))),
                (int(round(rx * 16)), int(round(ry * 16))), 0, 0, 360, 255, -1, cv2.LINE_AA, 4)
    a = (m.astype(np.float32) / 255.0)[..., None]
    reg = eb[y0:y1, x0:x1]
    reg *= 1 - a
    reg += a * rgb


# ---------------------------------------------------------------------------
# 각막 반사 플레이트 (네온 간판 + 무대 조명 + 고속도로 불빛 + 실루엣)
# ---------------------------------------------------------------------------
PS = 480


@lru_cache(maxsize=1)
def _plate_static():
    """반사 플레이트의 시간 불변 부분: (헤이즈+빌딩, 창문, 네온 간판, 무대 조명) 레이어."""
    base = np.zeros((PS, PS, 3), np.float32)
    yy = np.arange(PS, dtype=np.float32)[:, None]
    base += np.exp(-((yy - 300) / 50.0) ** 2)[..., None] * np.array([0.20, 0.04, 0.16], np.float32)
    win = np.zeros_like(base)
    rng = np.random.default_rng(4242)
    x = -8.0
    while x < PS:
        bw = rng.uniform(16, 44)
        bh = rng.uniform(24, 110) * (0.5 if 180 < x < 300 else 1.0)
        top = 340 - bh
        fill_poly(base, [(x, 344), (x, top), (x + bw, top), (x + bw, 344)], (0.0, 0.0, 0.0))
        for wy in np.arange(top + 5, 336, 7.0):
            for wx in np.arange(x + 3, x + bw - 3, 6.0):
                if rng.random() < 0.22:
                    hue = rng.random()
                    c = (np.array([1.0, 0.62, 0.30]) if hue < 0.55 else
                         np.array([0.40, 0.95, 1.0]) if hue < 0.8 else np.array([1.0, 0.40, 0.72]))
                    fill_poly(win, [(wx, wy), (wx + 2.6, wy), (wx + 2.6, wy + 3.2), (wx, wy + 3.2)],
                              c * rng.uniform(0.4, 1.0), mode="add")
        x += bw + rng.uniform(1, 8)
    # 고속도로 불빛 띠 (2컷과 이어지는 오렌지/레드 트레일)
    for i in range(46):
        xx0 = rng.uniform(-10, PS)
        ln = rng.uniform(10, 40)
        red = rng.random() < 0.5
        yl = 349 + (3 if red else 0) + 6 * ((xx0 - 240) / 240) ** 2
        c = np.array([1.0, 0.18, 0.12]) * 1.4 if red else np.array([1.2, 0.85, 0.55]) * 1.5
        fill_poly(win, [(xx0, yl), (xx0 + ln, yl), (xx0 + ln, yl + 1.6), (xx0, yl + 1.6)], c, mode="add")
    # 세로 네온 간판: 추상화된 한자풍 획 블록
    neon = np.zeros_like(base)
    for (sx, sy, n, cw, col, g) in ((58, 70, 6, 15, PINK, 2.2), (104, 120, 4, 10, CYAN, 1.9),
                                    (356, 82, 5, 14, PINK, 2.0), (408, 128, 5, 12, ORANGE, 1.8),
                                    (444, 60, 3, 9, CYAN, 1.5)):
        fill_poly(neon, [(sx - 3, sy - 4), (sx + cw + 3, sy - 4), (sx + cw + 3, sy + n * (cw + 5)),
                         (sx - 3, sy + n * (cw + 5))], col * 0.10, mode="add")
        for k in range(n):
            gy = sy + k * (cw + 5)
            for _ in range(int(rng.integers(2, 5))):
                if rng.random() < 0.5:   # 가로획
                    yv = gy + rng.uniform(0, cw - 2)
                    x0_, x1_ = sx + rng.uniform(0, cw * 0.4), sx + rng.uniform(cw * 0.6, cw)
                    fill_poly(neon, [(x0_, yv), (x1_, yv), (x1_, yv + 1.8), (x0_, yv + 1.8)], col * g, mode="add")
                else:                    # 세로획
                    xv = sx + rng.uniform(0, cw - 2)
                    y0_, y1_ = gy + rng.uniform(0, cw * 0.4), gy + rng.uniform(cw * 0.6, cw)
                    fill_poly(neon, [(xv, y0_), (xv + 1.8, y0_), (xv + 1.8, y1_), (xv, y1_)], col * g, mode="add")
    neon = neon + cv2.GaussianBlur(neon, (0, 0), 2.5) * 0.7
    # 무대 조명 줄 (위쪽)
    stage = np.zeros_like(base)
    for i, bx in enumerate((150, 196, 242, 288, 334)):
        by = 40 + 4 * ((bx - 242) / 92.0) ** 2
        fill_ellipse(stage, bx, by, 6.5, 5.0, np.array([2.4, 2.2, 2.1]), mode="add")
        fill_poly(stage, [(bx - 12, by), (bx, by - 0.8), (bx + 12, by), (bx, by + 0.8)],
                  np.array([1.6, 1.5, 1.5]), mode="add")
    stage = stage + cv2.GaussianBlur(stage, (0, 0), 3.0) * 0.8
    return base, win, neon, stage


def _figure_mask(cx, feet_y, sc, tilt=0.0):
    """도플갱어 실루엣(오리지널, 일반적인 성인 인물형) 마스크.
    단발 머리, 비대칭 어깨, 길게 늘어뜨린 팔, 무릎길이 원피스 — 얼굴 없음."""
    m = np.zeros((PS, PS), np.uint8)

    def T(pts):
        P = np.asarray(pts, np.float64) * sc + (cx, feet_y)
        return np.round(P * 16).astype(np.int32)

    ca, sa = math.cos(tilt), math.sin(tilt)

    def head(pts):
        P = np.asarray(pts, np.float64)
        piv = np.array([0.0, -140.0])
        Q = P - piv
        return np.stack([Q[:, 0] * ca - Q[:, 1] * sa, Q[:, 0] * sa + Q[:, 1] * ca], 1) + piv

    hair = catmull([(-12, -150), (-13, -164), (-8, -172), (1, -174), (9, -171), (13, -163), (12, -150),
                    (16, -139), (11, -141), (6, -146), (-5, -146), (-10, -142), (-15, -138), (-14, -144)],
                   6, closed=True)
    cv2.fillPoly(m, [T(head(hair))], 255, cv2.LINE_AA, 4)
    cv2.fillPoly(m, [T([(-3.5, -148), (3.5, -148), (4.5, -134), (-4.5, -134)])], 255, cv2.LINE_AA, 4)
    torso = catmull([(-19, -133), (0, -136), (20, -131), (23, -126), (17, -112), (12, -96), (-12, -96),
                     (-16, -112), (-22, -128)], 6, closed=True)
    cv2.fillPoly(m, [T(torso)], 255, cv2.LINE_AA, 4)
    cv2.fillPoly(m, [T([(-12, -98), (12, -98), (19, -52), (2, -50), (-18, -53)])], 255, cv2.LINE_AA, 4)
    thick = max(1, int(round(5.5 * sc)))
    for arm in ([(-20, -128), (-24, -108), (-25, -86), (-24, -70)], [(21, -126), (25, -104), (27, -84), (29, -71)]):
        cv2.polylines(m, [T(catmull(arm, 6))], False, 255, thick, cv2.LINE_AA, 4)
    lt = max(1, int(round(5.0 * sc)))
    for leg in ([(-6, -54), (-6.5, -26), (-6, -1)], [(6, -54), (7, -26), (8, -1)]):
        cv2.polylines(m, [T(leg)], False, 255, lt, cv2.LINE_AA, 4)
    return m.astype(np.float32) / 255.0


def _plate(f, resolve, kp):
    base, win, neon, stage = _plate_static()
    dim = 1.0 - 0.55 * resolve
    P = base * (0.5 + 0.5 * kp) * dim + win * (0.9 + 0.1 * math.sin(f * 1.7)) * dim
    P += neon * (kp * (1.0 - 0.35 * resolve))
    P += stage * (0.7 + 0.3 * kp)
    # 지나가는 헤드라이트 (얼굴을 스치는 빛과 같은 타이밍)
    if 7 <= f <= 17:
        k_ = (f - 7) / 10.0
        hxp = -60 + 600 * k_
        e_ = math.sin(math.pi * k_) ** 0.5
        fill_poly(P, [(hxp - 90, 196), (hxp, 193.5), (hxp + 26, 196), (hxp, 198.5)], np.array([1.6, 0.9, 1.2]) * e_,
                  mode="add")
        fill_ellipse(P, hxp, 196, 7, 3, np.array([3.0, 2.9, 2.8]) * e_, mode="add")
    # 지나가는 차 헤드라이트
    cxp = (f * 11.0) % 640 - 80
    fill_poly(P, [(cxp - 30, 347), (cxp + 30, 347), (cxp + 30, 350), (cxp - 30, 350)],
              np.array([2.2, 1.8, 1.3]) * dim, mode="add")
    # 중앙 스포트라이트(연무 속 빔) + 실루엣 (끝부분에 떠오름)
    if resolve > 0.01:
        spot = np.zeros_like(P)
        g = 1.75 * resolve
        yy = np.arange(PS, dtype=np.float32)[:, None, None]
        fill_poly(spot, [(237, 40), (247, 40), (300, 356), (184, 356)], np.array([0.62, 0.62, 1.0]) * g * 0.55,
                  mode="add")
        spot *= 0.35 + 0.65 * np.clip((yy - 40) / 316.0, 0, 1)
        fill_ellipse(spot, 242, 356, 62, 7, np.array([0.9, 0.8, 1.0]) * g * 0.75, mode="add")
        spot = cv2.GaussianBlur(spot, (0, 0), 3.5)
        P += spot
        sc = 1.12 + 0.26 * resolve
        tilt = -0.20 if f >= 56 else -0.05
        mf = _figure_mask(242, 356, sc, tilt)
        sig = 0.55 + (1 - resolve) ** 1.5 * 9.0
        mf = cv2.GaussianBlur(mf, (0, 0), sig) * min(1.0, resolve * 1.5)
        rim = np.clip(cv2.GaussianBlur(mf, (0, 0), 1.3) - mf, 0, 1) * resolve
        P *= (1 - mf)[..., None]
        P += rim[..., None] * np.array([1.4, 1.1, 1.7], np.float32) * (np.arange(PS)[:, None] < 250)[..., None]
        return P, mf
    return P, None


def _cornea(eb, plate, cx, cy, R, gain, iris_cx, iris_cy, irx, iry, wob=0.0, f=0, occ=None, occ_k=0.0):
    """반사 플레이트를 각막(볼록 거울)에 어안 매핑해 가산. wob: 눈물막 일렁임."""
    x0, x1 = int(max(0, cx - R)), int(min(W, cx + R + 1))
    y0, y1 = int(max(0, cy - R)), int(min(H, cy + R + 1))
    if x1 <= x0 or y1 <= y0:
        return
    qx = ((np.arange(x0, x1, dtype=np.float32) + 0.5 - cx) / R)[None, :]
    qy = ((np.arange(y0, y1, dtype=np.float32) + 0.5 - cy) / R)[:, None]
    d2 = qx * qx + qy * qy
    k = 0.40 + 0.42 * d2
    mx = (0.5 + qx * k) * PS - 0.5
    my = (0.5 + qy * k) * PS - 0.5
    if wob > 0:
        ph = f * 0.9
        mx = mx + wob * PS * np.sin(qy * 11.0 + ph) * (0.6 + 0.4 * np.sin(qx * 5.0 - ph * 0.7))
        my = my + wob * PS * 0.5 * np.sin(qx * 9.0 - ph * 1.3)
    refl = cv2.remap(plate, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fade = np.clip((1 - d2) / 0.2, 0, 1)
    ix = (np.arange(x0, x1, dtype=np.float32)[None, :] - iris_cx) / irx
    iy = (np.arange(y0, y1, dtype=np.float32)[:, None] - iris_cy) / iry
    inside = 1 - smoothstep(0.97, 1.01, np.sqrt(ix * ix + iy * iy))
    strength = (0.16 + 0.84 * inside) * fade * gain
    if occ is not None and occ_k > 0:     # 반사 속 실루엣은 홍채 빛까지 가린다
        oc_ = cv2.remap(occ, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        eb[y0:y1, x0:x1] *= (1 - oc_ * fade * occ_k)[..., None]
    eb[y0:y1, x0:x1] += refl * strength[..., None]


# ---------------------------------------------------------------------------
# 타이밍
# ---------------------------------------------------------------------------
JOLT = 20


def _openness(f):
    d = f // 2                          # 눈꺼풀 작화는 2코마
    o = {0: 0.0, 1: 0.40, 2: 1.06}.get(d, 1.0)
    if f >= JOLT:                       # 충격으로 눈을 크게 뜸 (홍채 위가 드러남)
        o += 0.10 + 0.05 * math.exp(-(f - JOLT) / 4.0)
    return o


def _pupil(f):
    fd = (f // 2) * 2
    if f < JOLT:
        return 0.43 + 0.012 * math.sin(fd * 0.6)
    if f == JOLT:
        return 0.17
    if f == JOLT + 1:
        return 0.14
    if f < JOLT + 4:
        return 0.17
    # 서서히 다시 열림 (어둠 속 실루엣을 향해)
    return 0.18 + 0.18 * ease_in_out((fd - 24) / 35.0, 1.6)


def _tremble(f):
    d = f // 2
    if f < JOLT:
        amp = 1.4
    elif f < JOLT + 6:
        amp = 6.0
    elif f < 45:
        amp = 2.4 + (f - 26) * 0.06
    else:
        amp = 3.4
    return (amp * (hash01(d, 701) - 0.5) * 2, amp * 0.7 * (hash01(d, 702) - 0.5) * 2)


def _neon(f):
    """(핑크 kp, 블루 kb) 조명 세기. 시작은 차갑고 어둡다 → 핑크 네온이 깜빡이며 켜짐."""
    kp = {0: 0.0, 1: 0.0, 2: 0.06, 3: 1.0, 4: 0.18, 5: 0.85}.get(f, 1.0)
    if f in (13, 31, 48):
        kp = 0.62
    kb = 0.68 if f < 3 else 0.95
    return kp, kb


def _camera(f):
    u = f / (FRAMES - 1)
    s = 1.0 + 0.28 * ease_in_out(u, 1.8)
    if f >= JOLT:
        s *= 1.0 + 0.03 + 0.04 * math.exp(-(f - JOLT) / 2.2)
    for fb in (30, 46, 53):
        if f >= fb:
            s *= 1.0 + 0.014 * math.exp(-(f - fb) / 2.0)
    e = ease_in_out(u, 1.5)
    ax = IRIS_C[0] + (958 - IRIS_C[0]) * e
    ay = IRIS_C[1] + (556 - IRIS_C[1]) * e
    dx = dy = 0.0
    if JOLT <= f <= JOLT + 7:
        a = 22.0 * math.exp(-(f - JOLT) / 1.8)
        dx = a * (hash01(f, 91) - 0.5) * 2
        dy = a * (hash01(f, 92) - 0.5) * 2
    return (s, ax - s * IRIS_C[0] + dx, ay - s * IRIS_C[1] + dy)


# ---------------------------------------------------------------------------
# 드로잉 헬퍼
# ---------------------------------------------------------------------------
def _star(img, x, y, size, color):
    """4방향 반짝임(크로스 플레어)."""
    if size < 1:
        return
    t = size * 0.07
    fill_poly(img, [(x - size, y), (x, y - t), (x + size, y), (x, y + t)], color, mode="add")
    fill_poly(img, [(x, y - size * 0.8), (x + t, y), (x, y + size * 0.8), (x - t, y)], color, mode="add")
    s2 = size * 0.30
    fill_poly(img, [(x - s2, y - s2), (x + t * 0.5, y - t * 0.5), (x + s2, y + s2), (x - t * 0.5, y + t * 0.5)],
              color * 0.7, mode="add")
    fill_poly(img, [(x + s2, y - s2), (x + t * 0.5, y + t * 0.5), (x - s2, y + s2), (x - t * 0.5, y - t * 0.5)],
              color * 0.7, mode="add")
    fill_ellipse(img, x, y, t * 2.2, t * 2.2, color * 1.2, mode="add")


def _flick(img, p0, d, length, width, bend, seed, color=INK):
    """속눈썹 한 가닥: 굵은 뿌리 → 날카로운 끝."""
    d = np.asarray(d, np.float64)
    d = d / (np.linalg.norm(d) + 1e-9)
    nrm = np.array([-d[1], d[0]])
    p0 = np.asarray(p0, np.float64)
    pts = np.array([p0, p0 + d * length * 0.35 + nrm * bend * 0.2,
                    p0 + d * length * 0.7 + nrm * bend * 0.6, p0 + d * length + nrm * bend])
    curve = catmull(pts, 8)
    curve = jitter(curve, 0.5, seed, 3)
    tt = np.linspace(0, 1, len(curve))
    ink(img, curve, width, color, (0.0, 0.0), pressure=np.clip(1 - tt, 0, 1) ** 0.62)


def _rot(v, a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([v[0] * c - v[1] * s_, v[0] * s_ + v[1] * c])


HAIR = [   # 이마에서 늘어진 앞머리 다발 (전경 셀) — 푸시인하면 프레임 밖으로 밀려남
    dict(sp=[(40, -170), (90, -20), (160, 120), (250, 240), (340, 318)], w=180, side=1, hi="p"),
    dict(sp=[(330, -170), (352, -40), (390, 90), (440, 196)], w=128, side=1, hi="p"),
    dict(sp=[(1880, -180), (1846, -40), (1790, 96), (1712, 228)], w=160, side=-1, hi="b"),
    dict(sp=[(1650, -160), (1636, -30), (1604, 96), (1566, 176)], w=64, side=-1, hi="b"),
]


def _hair_strand(spec, cam, t, seed, sway_amp):
    sp = np.array(spec["sp"], np.float64)
    n_ = len(sp)
    w_ = np.linspace(0, 1, n_) ** 1.5
    ph = hash01(seed, 3) * 6.28
    sway = np.stack([np.sin(t * 3.4 + ph) * sway_amp * w_, np.cos(t * 2.7 + ph) * sway_amp * 0.6 * w_], 1)
    C = catmull(_M(cam, sp + sway), 10)
    s = np.linspace(0, 1, len(C))
    wid = spec["w"] * cam[0] * np.clip(1 - s ** 2.2, 0, 1) ** 0.75
    n = _normals(C)
    L = C + n * (wid / 2)[:, None]
    R = C - n * (wid / 2)[:, None]
    outline = np.vstack([L, R[::-1]])
    sd = spec["side"]
    shade = np.vstack([C + n * (sd * wid * 0.12)[:, None], (C + n * (sd * wid * 0.56)[:, None])[::-1]])
    m = (s > 0.10) & (s < 0.70)
    hc = C[m] - n[m] * (sd * wid[m] * 0.20)[:, None]
    hw = (wid[m] * 0.11 * np.sin(np.linspace(0, np.pi, m.sum())) ** 0.7)[:, None]
    hi = np.vstack([hc + n[m] * hw, (hc - n[m] * hw)[::-1]])
    strand_lines = [C[(s > 0.05) & (s < 0.8)] + n[(s > 0.05) & (s < 0.8)] * (wid[(s > 0.05) & (s < 0.8)] * k)[:, None]
                    for k in (-0.30, 0.05, 0.32)]
    return outline, shade, hi, strand_lines


def _angel_ring(img, outline, cam, spec, idx, dwg, color):
    """80년대식 머리카락 하이라이트: 다발을 가로지르는 지그재그 띠(천사의 고리)를 다발 안으로 클리핑."""
    x0, y0 = int(max(0, outline[:, 0].min() - 2)), int(max(0, outline[:, 1].min() - 2))
    x1, y1 = int(min(W, outline[:, 0].max() + 3)), int(min(H, outline[:, 1].max() + 3))
    if x1 <= x0 or y1 <= y0:
        return
    s = cam[0]
    yb = cam[2] + (118.0 if spec["hi"] == "p" else 88.0) * s
    xs = np.arange(x0 - 40, x1 + 40, 18.0 * s)
    rng = np.random.default_rng(9100 + idx)
    zig_t = yb + 0.10 * (xs - x0) * (1 if spec["side"] > 0 else -1)
    top = np.stack([xs, zig_t + rng.uniform(-3, 3, len(xs)) * s], 1)
    teeth = rng.uniform(16, 34, len(xs)) * s
    bot = np.stack([xs, zig_t + 18 * s + teeth * (np.arange(len(xs)) % 2)], 1)
    band = np.vstack([top, bot[::-1]])
    m1 = np.zeros((y1 - y0, x1 - x0), np.uint8)
    m2 = np.zeros_like(m1)
    cv2.fillPoly(m1, [np.round((outline - (x0, y0)) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    cv2.fillPoly(m2, [np.round((band - (x0, y0)) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    m = cv2.min(m1, m2)
    a = (m.astype(np.float32) / 255.0)[..., None]
    reg = img[y0:y1, x0:x1]
    reg *= 1 - a
    reg += a * np.asarray(color, np.float32)


# ---------------------------------------------------------------------------
# 렌더
# ---------------------------------------------------------------------------
def render(ctx: Ctx) -> np.ndarray:
    f = ctx.lframe
    dwg = ctx.drawing(2)
    cam = _camera(f)
    s = cam[0]
    kp, kb = _neon(f)
    resolve = smoothstep(44.0, 57.0, float(f))
    for fb in (30, 46, 53):             # 심장 박동에 맞춰 핑크 네온이 맥동
        if f >= fb:
            kp *= 1.0 + 0.22 * math.exp(-(f - fb) / 1.6)
    kp *= 1.0 - 0.38 * resolve          # 끝으로 갈수록 방 안의 빛이 꺼지고 반사만 남는다
    kb *= 1.0 - 0.30 * resolve
    o = _openness(f)
    p = _pupil(f)
    tear_t = smoothstep(24.0, 52.0, float((f // 2) * 2))
    Mcam = np.array([[s, 0, cam[1]], [0, s, cam[2]]], np.float32)

    stack, alb, scl = _plates()
    st = cv2.warpAffine(stack, Mcam, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    Pm, Bm, ls = st[..., 0:1], st[..., 1:2], st[..., 2:3]
    light = kp * Pm * PINK + kb * Bm * BLUE                       # (H,W,3)
    img = cv2.warpAffine(alb, Mcam, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img *= AMB + ls * light
    # 네온 림라이트 셀 (핑크: 눈썹뼈/콧등, 블루: 광대)
    for rim_c, col, wmax, sign in ((RIM_BROW, PINK * 1.25 * kp, 15.0, 1), (RIM_NOSE, PINK * 1.0 * kp, 11.0, -1),
                                   (RIM_CHEEK, BLUE * 1.5 * kb, 13.0, -1)):
        Rc = _M(cam, rim_c)
        tt_ = np.linspace(0, 1, len(Rc))
        wv = wmax * s * np.sin(np.pi * tt_) ** 0.6
        fill_poly(img, _band(Rc, 0, sign * wv), col * np.array([1.0, 1.0, 1.0]), mode="add")

    # ---- 눈꺼풀 곡선 (2코마 작화 + 선 떨림) ----
    U_b = UPPER_CLOSED + (UPPER_OPEN - UPPER_CLOSED) * o
    lq = (hash01(dwg, 711) - 0.5) * (1.6 if f >= JOLT else 0.8)  # 아랫눈꺼풀 미세 떨림
    L_b = LOWER + np.stack([np.zeros(NS), lq * np.sin(np.pi * S_LIN)], 1)
    U = jitter(_M(cam, U_b), 0.7, 1000 + dwg, 9)
    L = jitter(_M(cam, L_b), 0.6, 2000 + dwg, 9)
    nU = _normals(U)
    nL = _normals(L)
    icx = icy = irx = iry = None

    # ---- 안구 버퍼 ----
    if o > 0.05:
        eb = cv2.warpAffine(scl, Mcam, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        eb *= 0.62 * (1 - 0.35 * resolve) + 0.62 * light
        tx, ty = _tremble(f)
        gaze_dy = {0: 26.0, 1: 14.0, 2: 4.0}.get(f // 2, 0.0)
        icx, icy = _Mp(cam, IRIS_C[0] + tx, IRIS_C[1] + ty + gaze_dy)
        irx, iry = IRIS_RX * s, IRIS_RY * s
        tint = (0.80 + 0.30 * light) * (1.0 - 0.30 * resolve)
        wave = None
        if JOLT <= f <= JOLT + 4:
            k_ = (f - JOLT) / 4.0
            wave = (p + (1.02 - p) * ease_out(k_, 1.6), 0.030 * (1 - k_ * 0.7))
        _draw_iris(eb, cam, icx, icy, p, tint, wave)
        x0i, x1i = int(max(0, icx - irx - 4)), int(min(W, icx + irx + 5))
        y0i, y1i = int(max(0, icy - iry - 4)), int(min(H, icy + iry + 5))
        bw_, bh_ = x1i - x0i, y1i - y0i
        lcx, lcy = icx - x0i, icy - y0i

        def _ell(cx_, cy_, rx_, ry_):
            m_ = np.zeros((bh_, bw_), np.uint8)
            cv2.ellipse(m_, (int(round(cx_ * 16)), int(round(cy_ * 16))),
                        (int(round(rx_ * 16)), int(round(ry_ * 16))), 0, 0, 360, 255, -1, cv2.LINE_AA, 4)
            return m_

        iris_m = _ell(lcx, lcy, irx, iry)
        # 위쪽 홍채 그림자 톤 (셀, 딱딱한 경계)
        cut = _ell(lcx + irx * 0.04, lcy + iry * 0.30, irx * 1.08, iry * 1.0)
        shm = cv2.subtract(iris_m, cut)
        a_sh = (shm.astype(np.float32) / 255.0)[..., None]
        eb[y0i:y1i, x0i:x1i] *= 1 - a_sh * (1 - np.array([0.46, 0.40, 0.64], np.float32))
        # 아래쪽 투과광 초승달 (홍채가 아래에서 빛남)
        cres = cv2.subtract(_ell(lcx, lcy, irx * 0.93, iry * 0.93),
                            _ell(lcx - irx * 0.05, lcy - iry * 0.24, irx * 0.97, iry * 0.98))
        a_c = (cres.astype(np.float32) / 255.0)[..., None]
        eb[y0i:y1i, x0i:x1i] += a_c * np.array([0.40, 0.12, 0.36], np.float32) * (0.5 + 0.6 * kp)
        # 동공
        prx, pry = p * irx, p * iry
        fill_ellipse(eb, icx, icy, prx * 1.07, pry * 1.07, np.array([0.14, 0.04, 0.09]))
        fill_ellipse(eb, icx, icy, prx, pry, np.array([0.008, 0.004, 0.016]))
        # 홍채 테두리 주선 (선 떨림 2코마)
        th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ring = np.stack([icx + irx * np.cos(th), icy + iry * np.sin(th)], 1)
        ring = jitter(ring, 1.0 + (1.2 if f >= JOLT else 0), 3000 + dwg, 5)
        ink(eb, ring, 8.0 * s, INK, (0, 0), closed=True,
            pressure=0.60 + 0.40 * np.clip(np.sin(th - 0.4), 0, 1) ** 0.6)
        # 각막 반사
        pl, occ = _plate(f, resolve, kp)
        _cornea(eb, pl, icx - irx * 0.03, icy + iry * 0.02, irx * 1.10,
                (0.46 + 0.85 * resolve) * (0.35 + 0.65 * max(kp, 0.25 + 0.6 * resolve)), icx, icy, irx, iry,
                wob=0.002 + 0.005 * tear_t, f=f, occ=occ, occ_k=0.88 * resolve)
        # 윗눈꺼풀이 드리운 그림자: 에어브러시 그늘(부드럽게) + 셀 그림자 띠(딱딱하게)
        sh_d = (34 + 60 * np.sin(np.pi * S_LIN) ** 0.8) * s * (0.6 + 0.4 * min(o, 1.0))
        soft = blur(poly_mask(_band(U, -4, sh_d * 2.6 + 30 * s, nU)), 22 * s)
        eb *= 1 - soft[..., None] * (1 - np.array([0.72, 0.70, 0.86], np.float32))
        fill_poly(eb, _band(U, -4, sh_d, nU), np.array([0.42, 0.38, 0.64]), mode="mul")
        # 아래쪽 흰자의 젖은 광택 (에어브러시)
        sheen = np.zeros((H, W), np.float32)
        for a0, a1 in ((0.10, 0.40), (0.62, 0.90)):
            j0, j1 = int(a0 * (NS - 1)), int(a1 * (NS - 1))
            tt_ = np.linspace(0, 1, j1 - j0)
            wv = np.sin(np.pi * tt_) ** 0.8
            cv2.fillPoly(sheen, [np.round(_band(L[j0:j1], -(16 + 6 * wv) * s, -(16 + 30 * wv) * s,
                                                nL[j0:j1]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
        sheen = blur(sheen, 10 * s)
        eb += sheen[..., None] * (0.10 + 0.22 * light) * (1 - 0.5 * resolve)
        # 눈물샘 (안쪽 눈구석)
        cx0, cy0 = _Mp(cam, 262, 634)
        car = np.array([(cx0 - 30 * s, cy0), (cx0 - 6 * s, cy0 - 26 * s), (cx0 + 28 * s, cy0 - 20 * s),
                        (cx0 + 44 * s, cy0 + 2 * s), (cx0 + 22 * s, cy0 + 24 * s), (cx0 - 8 * s, cy0 + 22 * s)])
        car = catmull(car, 6, closed=True)
        lc = light[int(min(H - 1, max(0, cy0))), int(min(W - 1, max(0, cx0)))]
        fill_poly(eb, car, SKIN_SH * (AMB + 0.9 * lc) * 1.2)
        fill_ellipse(eb, cx0 + 4 * s, cy0 - 8 * s, 9 * s, 5 * s, HI * 0.8, angle=-20)
        ink(eb, car[: len(car) // 2 + 3], 3.0 * s, INK * 2.5, (0.3, 0.3))
        # 아랫눈꺼풀 테 (젖은 점막)
        rim_w = (3 + 8 * np.sin(np.pi * np.clip((S_LIN - 0.02) / 0.96, 0, 1)) ** 0.7) * s
        fill_poly(eb, _band(L, 2, -rim_w, nL), (SKIN * 0.50 + PINK * 0.28) * (0.55 + 0.5 * kp))
        rim_hi = _band(L[20:160], -rim_w[20:160] * 0.9, -rim_w[20:160] * 0.62, nL[20:160])
        fill_poly(eb, rim_hi, np.array([1.1, 1.0, 1.25]) * 0.5, mode="add")
        # 셀 하이라이트 (각막 표면): 창문형 캐치라이트 + 작은 점들
        hx, hy = icx - irx * 0.50, icy - iry * 0.37
        ang = -14.0
        pw, ph_ = irx * 0.082, iry * 0.060
        for gx, gy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            ox, oy = _rot((gx * (pw + irx * 0.010), gy * (ph_ + irx * 0.010)), math.radians(ang))
            fill_poly(eb, _superellipse(hx + ox, hy + oy, pw, ph_, ang, 4.5, 32), HI * (1 - 0.12 * resolve))
        fill_ellipse(eb, icx - irx * 0.27, icy - iry * 0.17, irx * 0.040, irx * 0.036, HI)
        for (ux, uy, rr_) in ((0.36, 0.48, 0.040), (0.54, 0.32, 0.022), (-0.34, 0.60, 0.017)):
            fill_ellipse(eb, icx + irx * ux, icy + iry * uy, irx * rr_, irx * rr_, HI * 0.95)
        # 오른쪽 아래 반사 초승달 (블루 림)
        a_ = np.linspace(0.20, 1.20, 26)
        outer = np.stack([icx + irx * 0.92 * np.cos(a_), icy + iry * 0.92 * np.sin(a_)], 1)
        inner = np.stack([icx + irx * 0.86 * np.cos(a_) + irx * 0.03, icy + iry * 0.86 * np.sin(a_) - iry * 0.01], 1)
        fill_poly(eb, np.vstack([outer, inner[::-1]]), BLUE * 1.1 * kb + 0.15, mode="add")

        # 합성 (눈꺼풀 사이)
        open_poly = np.vstack([U, L[::-1]])
        ex0, ey0, ex1, ey1 = (int(max(0, open_poly[:, 0].min() - 3)), int(max(0, open_poly[:, 1].min() - 3)),
                              int(min(W, open_poly[:, 0].max() + 4)), int(min(H, open_poly[:, 1].max() + 4)))
        m8 = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
        cv2.fillPoly(m8, [np.round((open_poly - (ex0, ey0)) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        a = (m8.astype(np.float32) / 255.0)[..., None]
        reg = img[ey0:ey1, ex0:ex1]
        reg *= 1 - a
        reg += a * eb[ey0:ey1, ex0:ex1]

    # ---- 쌍꺼풀 ----
    oc = min(o, 1.0)
    Cr_b = CREASE + np.stack([np.zeros(NS), (1 - oc) * 70 * np.sin(np.pi * S_LIN)], 1)
    Cr = jitter(_M(cam, Cr_b), 0.8, 4000 + dwg, 9)
    nC = _normals(Cr)
    fold_d = (4 + 13 * np.sin(np.pi * S_LIN) ** 0.7) * s
    fill_poly(img, _band(Cr[10:150], 0, fold_d[10:150], nC[10:150]), np.array([0.66, 0.56, 0.78]), mode="mul")
    ink(img, Cr[12:168], 6.0 * s, np.array([0.10, 0.03, 0.07]), (0.35, 0.3))

    # ---- 윗속눈썹 띠 + 톱니 + 날개 ----
    tk = 0.50 + 0.50 * oc
    thick = np.interp(S_LIN, [0, 0.05, 0.2, 0.45, 0.7, 0.88, 1.0], [6, 16, 32, 50, 66, 80, 70]) * s * tk
    # 붓의 들쭉날쭉함 (2코마로 바뀜)
    rag = np.random.default_rng(8800 + dwg).standard_normal(NS + 8)
    rag = np.convolve(rag, np.ones(5) / 5, "same")[4:-4]
    thick = thick * (1 + 0.09 * rag * smoothstep(0.15, 0.4, S_LIN))
    # 눈두덩 그늘 (속눈썹 띠 위 에어브러시)
    occ = blur(poly_mask(_band(U, -thick * 0.9, -thick - 46 * s, nU)), 14 * s)
    img *= 1 - occ[..., None] * (1 - np.array([0.70, 0.64, 0.82], np.float32)) * 0.8
    tang = _tangents(U)
    up = -nU
    tE = tang[-3]
    wing_dir = _rot(tE, -0.55 * oc + 0.15 + 0.95 * (1 - oc))
    wing = np.array([U[-1] + wing_dir * k * (12 + 10 * oc) * s + up[-1] * (k * (2.0 * oc - 1.0) * s)
                     for k in range(1, 6)])
    Ue = np.vstack([U, wing])
    the = np.concatenate([thick, np.linspace(66 * (0.35 + 0.65 * oc), 3, 5) * s * tk])
    ne = _normals(Ue)
    center = Ue - ne * (the / 2 - 2.5 * s)[:, None]
    ink(img, center, float(the.max()), INK, (0.02, 0.0), pressure=the / the.max())
    # 바깥 속눈썹 — 굵게 뻗어 나가 바깥으로 말리는 3가닥 + 날개 끝 2가닥 (2코마 선 떨림)
    if oc > 0.5:
        for i, (sv, ln, wd, rot) in enumerate(((0.80, 118, 28, 0.20), (0.875, 158, 31, 0.50),
                                               (0.945, 176, 29, 0.85))):
            j = int(sv * (NS - 1))
            b0 = U[j] + up[j] * thick[j] * 0.55
            d = _rot(up[j], 0.12 + 0.42 * rot)
            L_ = ln * s * (0.85 + 0.15 * oc)
            _flick(img, b0, d, L_, wd * s, 0.36 * L_, 5000 + dwg * 13 + i)
        for i, (dang, ln, wd, bnd) in enumerate(((-0.30, 120, 22, 0.32), (0.30, 74, 16, -0.15))):
            L_ = ln * s
            _flick(img, wing[1], _rot(wing_dir, dang), L_, wd * s, bnd * L_, 5500 + dwg * 7 + i)
    else:   # 감은 눈: 바깥쪽으로 늘어진 3가닥
        for i, (sv, ln, wd) in enumerate(((0.78, 70, 18), (0.86, 92, 20), (0.93, 100, 18))):
            j = int(sv * (NS - 1))
            d = _rot(-up[j], -0.55 - 0.35 * i / 2)
            L_ = ln * s
            _flick(img, U[j] - up[j] * 2 * s, d, L_, wd * s, -0.30 * L_, 5000 + dwg * 13 + i)
    # 띠 위쪽 블루 림라이트 (투과광)
    rim_l = center[96:176] - ne[96:176] * (the[96:176] / 2 - 2.0 * s)[:, None]
    ink(img, rim_l, 2.4 * s, BLUE * 1.35 * kb, (0.3, 0.3))

    # ---- 아랫속눈썹 ----
    wl = np.interp(S_LIN, [0, 0.3, 0.7, 1], [2.0, 4.0, 7.0, 9.0]) * s
    for a0, a1 in ((0.16, 0.46), (0.49, 0.79), (0.81, 0.995)):
        j0, j1 = int(a0 * (NS - 1)), int(a1 * (NS - 1))
        ink(img, L[j0:j1] + nL[j0:j1] * 1.5 * s, float(wl[j1 - 1]), INK, (0.25, 0.2),
            pressure=wl[j0:j1] / wl[j1 - 1])
    if o > 0.3:   # 바깥 눈꼬리에 모인 아랫속눈썹
        tL = _tangents(L)
        for i, (sv, ln) in enumerate(((0.79, 28), (0.855, 40), (0.905, 52), (0.95, 44))):
            j = int(sv * (NS - 1))
            d = nL[j] * 0.8 + tL[j] * 0.6
            L_ = ln * s
            _flick(img, L[j] + nL[j] * 2 * s, d, L_, 8.0 * s, -0.30 * L_, 6000 + dwg * 11 + i)

    # ---- 눈물 ----
    if tear_t > 0 and o > 0.05:
        _tear(img, cam, L, nL, tear_t, f, kp)

    # ---- 전경 머리카락 (+드리운 그림자) ----
    t2 = ctx.on(2)
    hair_cam = (s * 1.05, cam[1] - IRIS_C[0] * s * 0.05, cam[2] - IRIS_C[1] * s * 0.05)
    sway = 3.0 + (2.5 if f >= JOLT else 0)
    strands = [_hair_strand(spec, hair_cam, t2, 80 + i, sway) for i, spec in enumerate(HAIR)]
    for outline, _, _, _ in strands:
        fill_poly(img, outline + np.array([26.0, 34.0]) * s, np.array([0.52, 0.45, 0.70]), mode="mul")
    for i, (spec, (outline, shade, hi, lines)) in enumerate(zip(HAIR, strands)):
        hc = PINK * kp if spec["hi"] == "p" else BLUE * kb
        fill_poly(img, outline, np.array([0.062, 0.042, 0.11]) + hc * 0.05)
        fill_poly(img, shade, np.array([0.026, 0.016, 0.05]))
        _angel_ring(img, outline, hair_cam, spec, i, dwg, np.array([0.13, 0.08, 0.22]) + hc * 0.85)
        for k, ln_ in enumerate(lines):
            ink(img, ln_, 2.0 * s, np.array([0.02, 0.01, 0.04]), (0.2, 0.5))
        ink(img, outline, 3.4 * s, INK, (0, 0), closed=True)
    # 흘러내린 잔머리 몇 가닥
    for i, (pts, w_) in enumerate(((((300, 150), (360, 240), (400, 330), (420, 400)), 3.0),
                                   (((230, 120), (250, 220), (300, 300), (330, 360)), 2.4),
                                   (((1730, 160), (1700, 250), (1690, 320)), 2.6))):
        P_ = catmull(np.array(pts, np.float64) + np.array([math.sin(t2 * 3.0 + i) * 3, 0]), 10)
        ink(img, jitter(_M(hair_cam, P_), 0.6, 7000 + dwg + i, 5), w_ * s, INK, (0.1, 0.9))

    # ---- 지나가는 헤드라이트 빛 (f7–17, 오른쪽 → 왼쪽) ----
    if 7 <= f <= 17:
        k_ = (f - 7) / 10.0
        xc = 2400 - 2900 * k_
        xs_ = np.arange(0, W, 4, dtype=np.float32)[None, :]
        ys_ = np.arange(0, H, 4, dtype=np.float32)[:, None]
        sw = np.exp(-((xs_ + 0.32 * ys_ - xc) / 200.0) ** 2) * math.sin(math.pi * k_) ** 0.5
        sw = cv2.resize(sw, (W, H), interpolation=cv2.INTER_LINEAR)
        img += sw[..., None] * np.array([0.80, 0.66, 0.58], np.float32) * np.minimum(img, 0.75)
    # ---- 끝부분: 시야가 좁아짐 (동적 비네트) ----
    if resolve > 0 and icx is not None:
        xs_ = (np.arange(0, W, 4, dtype=np.float32)[None, :] - icx) / (W * 0.5)
        ys_ = (np.arange(0, H, 4, dtype=np.float32)[:, None] - icy) / (H * 0.5)
        vg = 1 - 0.55 * resolve * np.clip(np.sqrt(xs_ * xs_ + ys_ * ys_ * 0.8) - 0.35, 0, 1) ** 1.3
        img *= cv2.resize(vg, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]

    # ---- 투과광 글로우 ----
    hot = np.clip(img - 1.05, 0, None)
    add_glow(img, hot, sigmas=(3, 10, 28), weights=(0.50, 0.32, 0.22))

    # ---- 임팩트: 반전 듀오톤 1프레임 → 줌 스미어 ----
    if f == JOLT:
        neg = np.clip(1.06 - luma(np.clip(img, 0, 1.2)), 0, 1) ** 1.9
        dark = np.array([0.05, 0.0, 0.10], np.float32)
        bright = np.array([0.95, 0.98, 1.30], np.float32)
        img = dark + neg[..., None] * (bright - dark)
    elif f in (JOLT + 1, JOLT + 2) and icx is not None:
        img = zoom_blur(img, icx, icy, amount=0.07 if f == JOLT + 1 else 0.025, steps=6)
    return np.clip(img, 0.0, 6.0).astype(np.float32)


def _tear(img, cam, L, nL, tt, f, kp):
    s = cam[0]
    T = 19.0 * tt * s
    B = 58.0 * smoothstep(0.25, 1.0, tt) ** 1.2 * s
    env = smoothstep(0.10, 0.28, S_LIN) * (1 - smoothstep(0.86, 0.97, S_LIN))
    sb = 0.50
    bulge = np.exp(-(np.abs(S_LIN - sb) / 0.052) ** 2.6)
    th = T * env + B * bulge
    rim_w = (3 + 8 * np.sin(np.pi * np.clip((S_LIN - 0.02) / 0.96, 0, 1)) ** 0.7) * s
    bottom = L - nL * (rim_w * 0.55)[:, None]
    top = bottom - nL * th[:, None]
    j0, j1 = 16, 176
    poly = np.vstack([top[j0:j1], bottom[j0:j1][::-1]])
    x0, y0 = int(max(0, poly[:, 0].min() - 6)), int(max(0, poly[:, 1].min() - 6))
    x1, y1 = int(min(W, poly[:, 0].max() + 7)), int(min(H, poly[:, 1].max() + 7))
    if x1 <= x0 or y1 <= y0:
        return
    m8 = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m8, [np.round((poly - (x0, y0)) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    a = (m8.astype(np.float32) / 255.0)[..., None]
    # 굴절: 위쪽 내용을 끌어내려 확대 + 차가운 물빛
    shift = int(max(2, th.max() * 1.1))
    ys0 = max(0, y0 - shift)
    src = img[ys0:ys0 + (y1 - y0), x0:x1]
    if src.shape[0] != y1 - y0:
        src = img[y0:y1, x0:x1]
    src = cv2.GaussianBlur(src, (0, 0), 1.2)
    water = src * np.array([0.80, 0.95, 1.15], np.float32) * 1.18 + np.array([0.05, 0.09, 0.17], np.float32)
    reg = img[y0:y1, x0:x1]
    reg *= 1 - a * 0.9
    reg += a * 0.9 * water
    # 색 트레스(아래쪽 굴절선, 위쪽 물 가장자리) + 투과광 하이라이트
    ink(img, bottom[j0 + 6:j1 - 6] - nL[j0 + 6:j1 - 6] * 1.0 * s, 2.6 * s, np.array([0.06, 0.12, 0.36]), (0.2, 0.2))
    ink(img, top[j0 + 10:j1 - 10], 1.8 * s * (0.3 + 0.7 * tt), np.array([0.55, 0.75, 1.0]) * 0.9, (0.3, 0.3))
    k0, k1 = 30, 150
    hl = top[k0:k1] + nL[k0:k1] * (1.8 * s + th[k0:k1] * 0.14)[:, None]
    for i, (a0, a1) in enumerate(((0.06, 0.20), (0.34, 0.44), (0.58, 0.63), (0.76, 0.88))):
        i0, i1 = int(a0 * (len(hl) - 1)), int(a1 * (len(hl) - 1))
        if i1 - i0 > 3:
            ink(img, hl[i0:i1], 3.4 * s * (0.4 + 0.6 * tt), np.array([1.5, 1.8, 2.2]), (0.4, 0.4))
    # 고인 방울 위의 하이라이트
    jb = int(sb * (NS - 1))
    if B > 4:
        bx, by = top[jb] + nL[jb] * th[jb] * 0.30 + np.array([-B * 0.22, 0])
        fill_ellipse(img, bx, by, B * 0.30, B * 0.15, np.array([2.0, 2.2, 2.5]), angle=-12)
        fill_ellipse(img, bx + B * 0.45, by + B * 0.22, B * 0.07, B * 0.07, np.array([1.6, 1.8, 2.2]))
    # 반짝임 (2코마)
    for (g0, sv, gain) in ((34, 0.38, 1.0), (53, 0.60, 1.15)):
        k = (f // 2) * 2 - g0
        if 0 <= k <= 6:
            size = (0, 40, 66, 28)[min(3, k // 2)] * s * gain
            j = int(sv * (NS - 1))
            gx, gy = top[j] + nL[j] * th[j] * 0.15
            _star(img, gx, gy, size, np.array([1.1, 1.25, 1.5], np.float32))
