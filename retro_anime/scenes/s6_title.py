"""타이틀 카드 「残像」 (36f, 13.5–15.0s).

연출 (장면 내부 프레임)
  f0     5의 순백에서 하드 컷 — 레드 필드에 타이틀이 1.5배로 날아와 꽂힘 (줌 블러 + 과노출)
  f1     임팩트 프레임: 흑지 + 백열 타이틀 + 붉은 집중선 (타이틀 0.93배로 압축)
  f2–5   오버슈트 바운스(1.05 → 0.985 → 1.0) + 감쇠 셰이크, 타이틀 윤곽의 '잔상'이 충격파처럼 퍼짐
  f3     로마자 "ZANZŌ" + 괘선 팝, f8–10 크레딧 라인이 네온처럼 깜빡이며 켜짐
  f5–27  홀드: 1코마 카메라 푸시인, 투과광 쉬머(2코마), 3장 루프 선 떨림, 불티 상승,
         확산 필터 광芒, 크로스 필터 글린트(f6–12 残, f16–23 像), 대각 광택 스윕(f10–20),
         세로 태그라인 「光は消えない」 비침(f12–20), 잔상 윤곽 맥동(f19–26), 배경 거대 고스트 글자 패럴랙스
  f27–32 필름 런아웃: 노출 깜빡임(f27–29) + 라이트 리크 → f30 프레임 슬립(프레임 선이 지나감)
         → f31–32 필름 번(기포가 터지며 녹은 구멍이 백열, 주황 불테 + 그을음 띠, 초점 나감)
  f33–35 완전한 검정 (필름 후처리의 그레인/먼지만 남음)

구조
  * 붓글씨풍 타이틀은 고딕 글리프를 기울이고(이탤릭), 세로 침식/가로 팽창으로 명조식 굵기 대비를 준 뒤
    가장자리 번짐(노이즈 임계), 획 방향으로 정렬된 갈필(카스레) 줄무늬, 잉크 비말을 더해 만든다.
    획 방향은 마스크 구조 텐서로 추정. 결과는 국소 좌표 캐시 → 매 프레임 cv2 아핀 배치.
  * 배경(페인티드 레드), 고스트 글자, 자막 마스크는 lru_cache 정적 레이어.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import cv2
import numpy as np

from ..core import (FONT_JP, FONT_JP_ALT, FONT_LATIN, H, W, Ctx, blur, clamp, ease_in_out, ease_out,
                    hash01, hexc, noise, painted_texture, place_mask, radial_field,
                    radial_gradient, radial_lines_mask, smoothstep, stroke_mask, text_mask,
                    transform, zoom_blur)

NAME = "s6_title"
FRAMES = 36

EVENTS = [
    (0, "title_hit"),    # 순백 → 레드 하드 컷, 타이틀 슬램
    (0, "impact"),
    (1, "flash"),        # 흑지 임팩트 프레임
    (2, "shockwave"),    # 잔상 윤곽이 퍼져 나감
    (3, "pulse"),        # ZANZŌ 팝
    (7, "blink"),        # 글린트 1 (残 위)
    (8, "blink"),        # 크레딧 네온 점등 (f9 꺼짐, f10 재점등)
    (11, "swell"),       # 광택 스윕
    (17, "spark"),       # 글린트 2 (像 오른쪽 아래)
    (19, "pulse"),       # 잔상 윤곽 심장 박동
    (28, "runout"),      # 필름 런아웃: 노출 깜빡임 + 라이트 리크
    (30, "glitch"),      # 프레임 슬립
    (31, "flash"),       # 필름 번 시작
    (32, "tear"),        # 필름이 녹아 터짐 → f33 검정
]

# 런아웃 분위기: 먼지/스크래치를 넉넉히, 레드는 덜 바래게
FILM = dict(dust=1.8, saturation=0.97, flicker=0.03, halation=0.6)

f32 = np.float32
W2, H2 = W // 2, H // 2

# ---------------------------------------------------------------------------
# 팔레트 / 레이아웃
# ---------------------------------------------------------------------------
RED = hexc("#d4141f")
RED_HOT = hexc("#ff3a1e")
RED_DEEP = hexc("#5a0510")
CREAM = hexc("#fff0d2")
CREAM_EDGE = hexc("#ffd9a0")
WHITE_CORE = hexc("#fffaf0")
EMBER = hexc("#ffb050")

TX, TY = 960.0, 452.0          # 타이틀 중심
TITLE_SCALE = 1.28             # 글리프 → 화면 배율
SUB_Y = 800.0                  # ZANZŌ
CRED_Y = 866.0                 # 크레딧

# 타이틀 글리프: 획 끝이 붓처럼 부푼 정흑체(WQY)를 우선, 없으면 IPA 고딕
FONT_TITLE = FONT_JP_ALT if os.path.exists(FONT_JP_ALT) else FONT_JP
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
if not os.path.exists(FONT_SERIF):
    FONT_SERIF = FONT_LATIN

T_RUN = 28                     # 필름 런아웃 시작
T_BLACK = 33                   # 이후 완전한 검정
GHOST_K = 0.07                 # 배경 고스트 글자 농도
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 슬램 타이밍 (프레임 → 값)
SLAM_SCALE = {0: 1.55, 1: 0.93, 2: 1.055, 3: 0.985, 4: 1.006}
SHAKE_AMP = {0: 30.0, 1: 22.0, 2: 16.0, 3: 10.0, 4: 6.0, 5: 3.0, 6: 1.5}


def _half(m):
    """화면 크기 1채널 → 반해상도 (글로우 계산용)."""
    return cv2.resize(m, (W2, H2), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# 붓글씨 타이틀 (국소 좌표)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _glyph_base():
    """기울이고 굵기 대비를 준 매끈한 글리프 + 획 방향 줄무늬장 + 글린트 앵커."""
    m = text_mask("残像", 400, font=FONT_TITLE, stroke=8, spacing=34)
    pad = 110
    m = np.pad(m, pad).astype(f32)
    h, w = m.shape
    shear = -0.13
    M = np.array([[1, shear, -shear * h / 2], [0, 1, 0]], f32)
    m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR)
    # 명조식 굵기 대비: 가로획은 가늘게
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 11)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 1)))
    # 구조 텐서 → 획 방향
    sm = blur(m, 5)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = blur(gx * gx, 12), blur(gy * gy, 12), blur(gx * gy, 12)
    th_s = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    acc = np.zeros_like(m)
    wsum = np.zeros_like(m)
    for k in range(8):
        a = k * np.pi / 8
        n = noise(w, h, scale=0.9, ox=k * 211.0, oy=k * 97.0, seed=600, beta=1.15, sx=34, sy=1, angle=a)
        wk = (np.cos(th_s + a) ** 2) ** 10
        acc += n * wk
        wsum += wk
    streak = acc / (wsum + 1e-6)
    streak = (streak - streak.mean()) / (streak.std() + 1e-6)
    # 글린트 앵커: 残 의 맨 위 획 끝, 像 의 오른쪽 아래 삐침 끝
    ys, xs = np.nonzero(m > 0.5)
    left = xs < w / 2
    i0 = np.argmin(ys[left] * 1.0 - xs[left] * 0.35)
    a0 = (float(xs[left][i0]), float(ys[left][i0]))
    right = ~left
    i1 = np.argmax(xs[right] * 1.0 + ys[right] * 0.8)
    a1 = (float(xs[right][i1]), float(ys[right][i1]))
    return m, streak.astype(f32), (a0, a1)


@lru_cache(maxsize=4)
def _splat(k: int):
    """획 가장자리에서 튄 잉크 비말 (동화 k 마다 미세하게 다름)."""
    m, _, _ = _glyph_base()
    h, w = m.shape
    sm = blur(m, 3)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    crowd = blur(m, 22)
    ys, xs = np.nonzero((sm > 0.4) & (sm < 0.6) & (crowd < 0.3))
    rng = np.random.default_rng([1988, 66])
    idx = rng.choice(len(xs), size=min(len(xs), 38), replace=False)
    out = np.zeros((h, w), np.uint8)
    S = 16
    for j, i in enumerate(idx):
        x, y = float(xs[i]), float(ys[i])
        nx, ny = -float(gx[ys[i], xs[i]]), -float(gy[ys[i], xs[i]])
        ln = math.hypot(nx, ny) + 1e-6
        nx, ny = nx / ln, ny / ln
        d = 6 + rng.exponential(22)
        jit = rng.normal(0, 0.35)
        c, s = math.cos(jit), math.sin(jit)
        dx, dy = nx * c - ny * s, nx * s + ny * c
        r = max(1.0, 6.0 * math.exp(-d / 30) * rng.uniform(0.35, 1.1))
        ex, ey = x + dx * (d + 8), y + dy * (d + 8)
        if not (0 <= int(ex) < w and 0 <= int(ey) < h) or crowd[int(ey), int(ex)] > 0.12:
            continue
        bx = x + dx * d + (hash01(66, j, k) - 0.5) * 2.0
        by = y + dy * d + (hash01(67, j, k) - 0.5) * 2.0
        ang = math.degrees(math.atan2(dy, dx))
        cv2.ellipse(out, (int(bx * S), int(by * S)), (int(r * 1.5 * S), int(r * S)), ang, 0, 360,
                    255, -1, cv2.LINE_AA, 4)
        if rng.random() < 0.4:   # 꼬리 위성
            d2 = d + r * 2.5 + rng.uniform(3, 14)
            r2 = r * rng.uniform(0.3, 0.55)
            cv2.circle(out, (int((x + dx * d2) * S), int((y + dy * d2) * S)), int(r2 * S), 255, -1,
                       cv2.LINE_AA, 4)
    return out.astype(f32) / 255.0


def _z(n):
    return (n - n.mean()) / (n.std() + 1e-6)


EXT_L = 26.0                   # 돌출(익스트루드) 깊이 (국소 px)
EXT_DIR = (0.52, 1.0)          # 오른쪽 아래


@lru_cache(maxsize=4)
def _title_drawing(k: int):
    """선 떨림 루프용 동화 k (0..2).
    반환: (h,w,4) [면 마스크, 코어(획 중심 밝기), 돌출 마스크, 돌출 깊이 0..1] 국소 좌표."""
    m, streak, _ = _glyph_base()
    h, w = m.shape
    nf = noise(w, h, scale=0.32, ox=k * 173.0, oy=k * 59.0, seed=700, beta=1.0)
    nl = noise(w, h, scale=3.5, ox=k * 31.0, oy=k * 17.0, seed=710, beta=2.0)
    # 번진 가장자리 + 붓털 결(획 방향 줄무늬)이 가장자리에 남음
    f = blur(m, 1.6) + (nf - 0.5) * 0.30 + (nl - 0.5) * 0.10 + streak * 0.07
    mm = smoothstep(0.47, 0.53, f)
    # 갈필(카스레): 획 방향 줄무늬 틈, 일부 구역에만 (동화 간 고정 — 종이 위 붓자국)
    dist = cv2.distanceTransform((mm > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    gate = smoothstep(0.45, 1.1, _z(noise(w, h, scale=4.5, ox=400.0, seed=710, beta=2.0)))
    gaps = smoothstep(-0.62, -0.50, streak + _z(nl) * 0.12)
    edgy = np.clip(1.4 - dist / 14.0, 0, 1)
    mm = mm * (1 - gate * (1 - gaps) * (0.55 + 0.45 * edgy))
    mm = np.maximum(mm, _splat(k))
    core = blur(np.clip(dist / 16.0, 0, 1) ** 0.8, 2.0)
    # 돌출: 면을 오른쪽 아래로 밀어 쌓는다 (깊이 기록)
    dx, dy = EXT_DIR
    ln = math.hypot(dx, dy)
    dx, dy = dx / ln, dy / ln
    n = int(EXT_L)
    ext = np.zeros_like(mm)
    dep = np.zeros_like(mm)
    solid = (blur(m, 1.0) > 0.5).astype(f32)
    for i in range(n, 0, -1):
        M = np.array([[1, 0, dx * i], [0, 1, dy * i]], f32)
        sh = cv2.warpAffine(solid, M, (w, h), flags=cv2.INTER_LINEAR)
        dep = np.where(sh > 0.5, i / n, dep)
        ext = np.maximum(ext, sh)
    ext = ext * (1 - smoothstep(0.3, 0.7, blur(m, 1.0)))
    return np.dstack([mm, core, ext, dep]).astype(f32)


def _place_M(shape, x, y, s, rot=0.0):
    h, w = shape[:2]
    ax, ay = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((ax, ay), rot, s)
    M[0, 2] += x - ax
    M[1, 2] += y - ay
    return M


def _warp(arr, M):
    return cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# 정적 레이어
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _bg():
    """포스터컬러 레드 필드: 중심이 뜨겁고 가장자리로 갈수록 깊은 크림슨."""
    img = radial_gradient(TX, TY + 30, 1060,
                          [(0.0, RED_HOT * 1.08), (0.28, hexc("#ea1d1e")), (0.55, RED),
                           (0.8, hexc("#9c0b18")), (1.0, RED_DEEP)], sx=1.35, sy=1.0)
    painted_texture(img, amount=0.075, seed=6060, scale=3.0, stretch=5.0, angle=0.22)
    n = noise(W, H, scale=10, seed=6070, beta=2.4)
    img *= (0.95 + 0.1 * n)[..., None]
    # 가로 붓자국 (밝은 결)
    bs = noise(W, H, scale=1.6, seed=6080, beta=1.4, sx=18, sy=1, angle=0.05)
    img *= (1 + 0.07 * smoothstep(0.55, 0.75, bs) - 0.04 * smoothstep(0.45, 0.3, bs))[..., None]
    return img.astype(f32)


@lru_cache(maxsize=1)
def _ghost_local():
    m, _, _ = _glyph_base()
    return blur(m, 2.5)


def _crop(m, pad=24):
    """화면 크기 마스크 → (크롭, (y0, y1, x0, x1))."""
    ys, xs = np.nonzero(m > 0.002)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    return m[y0:y1, x0:x1].copy(), (y0, y1, x0, x1)


@lru_cache(maxsize=1)
def _subs():
    """ZANZŌ, 괘선, 크레딧, 세로 태그라인: 크롭 마스크 + bbox + 반해상도 글로우 원본."""
    z = text_mask("ZANZŌ", 60, font=FONT_SERIF, spacing=44)
    zm = place_mask(z, TX, SUB_Y)
    half = z.shape[1] / 2
    rules = np.zeros((H, W), f32)
    for sgn in (-1, 1):
        x0 = TX + sgn * (half + 46)
        x1 = TX + sgn * (half + 300)
        rules = np.maximum(rules, stroke_mask([(x0, SUB_Y), (x1, SUB_Y)], 2.2))
        # 마름모 끝장식
        cx = x1 + sgn * 12
        d = np.array([(cx - 8, SUB_Y), (cx, SUB_Y - 8), (cx + 8, SUB_Y), (cx, SUB_Y + 8)])
        dm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(dm, [np.round(d * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        rules = np.maximum(rules, dm.astype(f32) / 255)
    cred = text_mask("AN ORIGINAL PROCEDURAL ANIMATION", 21, font=FONT_LATIN, spacing=9)
    cm = place_mask(cred, TX, CRED_Y)
    tag = text_mask("光は消えない", 32, font=FONT_TITLE, vertical=True, stroke=1, spacing=6)
    tg = place_mask(tag, 1652, TY + 10)
    out = {"half": half}
    for name, m in (("z", zm), ("rules", rules), ("cred", cm), ("tag", tg)):
        out[name] = _crop(m)
        out[name + "_h"] = _half(m)
    return out


@lru_cache(maxsize=1)
def _ember_params():
    rng = np.random.default_rng([1988, 6600])
    n = 54
    return dict(x=rng.uniform(-40, W + 40, n), y=rng.uniform(0, H + 200, n),
                sp=rng.uniform(70, 260, n), sz=rng.uniform(1.0, 3.6, n),
                ph=rng.uniform(0, 6.283, n), sw=rng.uniform(10, 45, n), br=rng.uniform(0.4, 1.0, n))


# ---------------------------------------------------------------------------
# 효과
# ---------------------------------------------------------------------------
def _star(img, x, y, size, strength, rot=0.0):
    """크로스 필터 글린트 (4줄기 + 짧은 대각 4줄기)."""
    if strength <= 0.01:
        return
    R = int(size) + 4
    x0, y0 = int(x) - R, int(y) - R
    xs = np.arange(2 * R + 1, dtype=f32)[None, :] + x0 - x
    ys = np.arange(2 * R + 1, dtype=f32)[:, None] + y0 - y
    acc = np.zeros((2 * R + 1, 2 * R + 1), f32)
    for i in range(8):
        a = math.radians(rot + i * 45)
        L = size if i % 2 == 0 else size * 0.38
        u = xs * math.cos(a) + ys * math.sin(a)
        v = -xs * math.sin(a) + ys * math.cos(a)
        t = np.clip(1 - u / L, 0, 1) * (u > 0)
        wdt = 0.6 + 3.2 * t
        acc += np.exp(-(v / wdt) ** 2) * t ** 2.2
    r2 = xs * xs + ys * ys
    acc += np.exp(-r2 / (2 * (size * 0.05) ** 2)) * 2.0 + np.exp(-r2 / (2 * (size * 0.16) ** 2)) * 0.5
    # 클리핑
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + 2 * R + 1), min(H, y0 + 2 * R + 1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    patch = acc[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    img[sy0:sy1, sx0:sx1] += patch[..., None] * hexc("#fff4e0") * strength


def _embers(img, halo, ctx, amt=1.0):
    """폭발의 잔불: 아래에서 위로 흔들리며 오르는 불티 (2코마)."""
    p = _ember_params()
    t = ctx.on(2) + 0.5
    lay = np.zeros((H, W), np.uint8)
    S = 16
    for i in range(len(p["x"])):
        y = (p["y"][i] - p["sp"][i] * t) % (H + 260) - 60
        x = p["x"][i] + p["sw"][i] * math.sin(p["ph"][i] + t * 2.3)
        b = p["br"][i] * (0.55 + 0.45 * hash01(6601, i, ctx.drawing(2)))
        b *= 0.45 + 0.55 * clamp(y / H)          # 아래쪽일수록 뜨거움
        r = p["sz"][i]
        cv2.ellipse(lay, (int(x * S), int(y * S)), (int(r * S), int(r * 1.9 * S)),
                    10.0 * math.sin(p["ph"][i]), 0, 360, int(255 * b), -1, cv2.LINE_AA, 4)
    e = lay.astype(f32) * (amt / 255.0)
    img += e[..., None] * (EMBER * 2.6)
    halo += blur(_half(e), 2.5)[..., None] * (hexc("#ff7a30") * 2.2)


def _rays(src_small, cx, cy, amount, steps=10):
    """저해상도 방사 줌 블러 → 투과광 광선."""
    return zoom_blur(src_small, cx, cy, amount=amount, steps=steps)


@lru_cache(maxsize=1)
def _ray_angle(q=4):
    xs = np.arange(W // q, dtype=f32)[None, :] - TX / q
    ys = np.arange(H // q, dtype=f32)[:, None] - TY / q
    return np.arctan2(ys, xs * 0.8)


def _ray_streaks(lf, q=4):
    """확산 필터의 광芒 줄기: 각도 방향 줄무늬 (천천히 회전, 2코마)."""
    a = _ray_angle(q) + 0.012 * (lf // 2)
    acc = np.zeros_like(a)
    for i, (nn, wt) in enumerate(((7, 1.0), (13, 0.8), (29, 0.6), (47, 0.45))):
        acc += wt * np.sin(a * nn + hash01(6530, i) * 6.283)
    acc /= 2.85
    return 0.35 + 1.3 * smoothstep(-0.1, 0.6, acc)


def _burn(img, R, seed):
    """필름 번: 녹아 뚫린 구멍(백열) → 가는 주황 불테 → 그을린 갈색 띠 → 열로 바랜 상.
    가장자리는 부드러운 노이즈 + 녹아 터진 기포(원)들의 합집합."""
    if R <= 0:
        return img
    ox, oy = 1390.0, 600.0
    n = _z(noise(W, H, scale=2.6, seed=6900, beta=2.2))
    n2 = _z(noise(W, H, scale=1.3, ox=seed * 157.0, oy=seed * 83.0, seed=6910, beta=2.0))
    d = radial_field(ox, oy, sx=1.2, sy=1.0)
    dd = d - R * (1 + 0.20 * n) - n2 * (4 + 0.03 * R)
    hole = (dd < 0).astype(np.uint8) * 255
    rng = np.random.default_rng([1988, 6920, seed])
    for _ in range(int(14 + R / 25)):
        a = rng.uniform(0, 2 * np.pi)
        rr = R * rng.uniform(0.85, 1.12)
        cx, cy = ox + math.cos(a) * rr * 1.2, oy + math.sin(a) * rr
        br = R * rng.uniform(0.03, 0.11) + 4
        cv2.circle(hole, (int(cx * 16), int(cy * 16)), int(br * 16), 255, -1, cv2.LINE_AA, 4)
    hole = cv2.GaussianBlur(hole, (0, 0), 1.2)
    dist = cv2.distanceTransform((hole < 128).astype(np.uint8), cv2.DIST_L2, 5)
    dist = dist + n2 * 2.0
    w1 = 5 + 0.012 * R
    w2 = w1 + 14 + 0.05 * R
    hm = hole.astype(f32) / 255.0
    rim = smoothstep(-1, 1.5, dist) * smoothstep(w1 + 2, w1 - 2, dist) * (1 - hm)
    scorch = smoothstep(w1 - 3, w1 + 3, dist) * smoothstep(w2 + 8, w2 - 8, dist) * (1 - hm)
    heat = np.exp(-dist / (w2 * 3.5)) * (1 - hm)
    img = img * (1 + heat[..., None] * np.array([0.55, 0.2, -0.25], f32))
    img = img * (1 - scorch[..., None] * 0.88) + scorch[..., None] * hexc("#2e0e03") * 0.5
    img = img * (1 - rim[..., None]) + rim[..., None] * np.array([2.8, 1.25, 0.3], f32)
    img = img * (1 - hm[..., None]) + hm[..., None] * np.array([3.4, 3.1, 2.6], f32)
    return img


def _leak(k):
    """오른쪽 가장자리에서 번지는 라이트 리크 (주황→적)."""
    a = [0.45, 0.85, 1.25, 1.6, 2.0][k]
    cx, cy = W + 160 - 40 * k, 260 + 120 * k
    d = radial_field(cx, cy, sx=0.75, sy=1.25) / (620 + 140 * k)
    n = noise(W, H, scale=3.0, ox=k * 61.0, oy=k * 29.0, seed=6950, beta=2.0)
    f = np.exp(-(d * (0.8 + 0.5 * n)) ** 2 * 2.2)
    out = f[..., None] * hexc("#ff6a1a") * a
    out += (f ** 3)[..., None] * hexc("#ffe08a") * a * 1.2
    # 아래 왼쪽 모서리의 약한 붉은 번짐
    d2 = radial_field(-120, H + 80, sx=1.4, sy=0.8) / 700
    out += np.exp(-d2 ** 2 * 2.5)[..., None] * hexc("#ff3010") * a * 0.45
    return out.astype(f32)


# ---------------------------------------------------------------------------
# 렌더
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _hotspot():
    """백라이트 광원의 핫스팟: 타이틀 중심이 가장 뜨겁고 양 끝은 살짝 식음."""
    d = radial_field(TX, TY, sx=1.6, sy=1.0) / 620.0
    return (0.82 + 0.3 * np.exp(-d * d * 1.5)).astype(f32)


def _shake(lf):
    a = SHAKE_AMP.get(lf, 0.0)
    if a <= 0:
        return 0.0, 0.0, 0.0
    ang = hash01(6501, lf) * 2 * math.pi
    sgn = -1 if lf % 2 else 1
    return (a * math.cos(ang), a * 0.7 * math.sin(ang), sgn * a * 0.03)


def _title_scale(lf):
    return SLAM_SCALE.get(lf, 1.0)


def _sweep_at(lf, xs, ys):
    """광택 스윕 밴드 값 (xs: (1,n), ys: (m,1) 화면 좌표)."""
    u = (lf - 10) / 10.0
    c = -500 + 1900 * ease_in_out(u)
    proj = (xs - TX) * 0.86 + (ys - TY) * 0.5
    return np.exp(-((proj - c) / 150.0) ** 2) * math.sin(math.pi * u) ** 0.5


def _over(img, crop, box, color, alpha=1.0):
    y0, y1, x0, x1 = box
    a = crop[..., None] * alpha
    R = img[y0:y1, x0:x1]
    R *= 1 - a
    R += a * np.asarray(color, f32)


def _compose(ctx: Ctx) -> np.ndarray:
    lf = ctx.lframe
    impact = lf == 1
    ts = _title_scale(lf) * TITLE_SCALE
    k = (lf // 2) % 3                          # 3장 루프 선 떨림 (2코마)
    drw = _title_drawing(k)
    gm, _, (a0, a1) = _glyph_base()
    M = _place_M(drw.shape, TX, TY, ts)
    st = _warp(drw, M)
    tm, core, ext, dep = st[..., 0], st[..., 1], st[..., 2], st[..., 3]
    # 타이틀 bbox (국소 배열 모서리의 아핀 상)
    hh, ww = drw.shape[:2]
    cs = np.array([[0, 0, 1], [ww, 0, 1], [0, hh, 1], [ww, hh, 1]], np.float64) @ M.T
    bx0, by0 = max(0, int(cs[:, 0].min()) - 4), max(0, int(cs[:, 1].min()) - 4)
    bx1, by1 = min(W, int(cs[:, 0].max()) + 5), min(H, int(cs[:, 1].max()) + 5)
    tsl = (slice(by0, by1), slice(bx0, bx1))

    # 투과광 쉬머 (2코마 + 완만한 맥동)
    sh = 1.0 + 0.07 * (hash01(6510, lf // 2) - 0.5) * 2 + 0.05 * math.sin(lf * 0.9)
    boost = {0: 1.5, 1: 1.25, 2: 1.4, 3: 1.2, 4: 1.08}.get(lf, 1.0)
    sweeping = 10 <= lf <= 20

    halo = np.zeros((H2, W2, 3), f32)          # 저주파 빛(글로우/광선)은 반해상도에 모아서 한 번에 더함
    tm_h = _half(tm)
    core_h = _half(core)

    # --- 배경 ---
    if impact:
        img = np.empty((H, W, 3), f32)
        img[:] = hexc("#0c0204")
        lines = radial_lines_mask(TX, TY, n=150, seed=6520, r_in=560, r_jit=0.8,
                                  width=(0.002, 0.009))
        img += lines[..., None] * hexc("#ff2a14") * 1.6
    else:
        img = _bg().copy()
        if lf <= 6:                            # 컷 직후 중심 에어브러시가 과열됐다 식음
            img *= 1.0 + 0.5 * math.exp(-lf / 2.0)
        # 거대 고스트 글자 (원경, 패럴랙스 — 카메라보다 느리게, 반대로 표류)
        gs = 2.55 * (1 + 0.008 * lf) * (1.0 + 0.25 * math.exp(-lf / 1.5))
        gM = _place_M(gm.shape, TX - 40 + 2.2 * lf, TY + 70, gs)
        g = _warp(_ghost_local(), gM)
        img *= 1 - g[..., None] * (GHOST_K * np.array([1.0, 1.5, 0.8], f32))

    # --- 광선 (1/4 해상도 방사 블러 × 각도 줄무늬) ---
    q = 4
    src = cv2.resize(tm_h * (0.5 + 0.5 * core_h), (W // q, H // q), interpolation=cv2.INTER_AREA)
    ray_amt = {0: 0.55, 1: 0.45, 2: 0.45, 3: 0.38}.get(lf, 0.32)
    ray_k = {0: 0.9, 1: 0.7, 2: 1.2, 3: 0.95, 4: 0.8}.get(lf, 0.62) * sh
    rays = _rays(src, TX / q, TY / q, ray_amt, steps=12) * _ray_streaks(lf, q)
    rays = cv2.resize(rays, (W2, H2), interpolation=cv2.INTER_LINEAR)
    rays = np.maximum(rays - tm_h * 0.6, 0)
    halo += rays[..., None] * (hexc("#ff9a48") * ray_k)

    # --- 잔상 에코 (슬램 직후 퍼지는 윤곽 / 홀드 중 한 번의 맥동) ---
    echoes = []
    if 1 <= lf <= 10:
        col = hexc("#ffd0a0") if not impact else hexc("#ff4020")
        for e in range(1, 4):
            ea = 0.55 / e * math.exp(-(lf - 1) / 3.0)
            if ea >= 0.02:
                echoes.append((ts * (1 + (0.07 + 0.075 * (lf - 1)) * e), col * (ea * 2.2), col * (ea * 0.25)))
    if 19 <= lf <= 26:
        u = (lf - 19) / 7.0
        ea = 0.9 * (1 - u) ** 1.5
        echoes.append((ts * (1.0 + 0.32 * ease_out(u, 2)), hexc("#ffd8a8") * ea, hexc("#ff9a50") * (ea * 0.12)))
    for es, c_ring, c_soft in echoes:
        em = _warp(gm, _place_M(gm.shape, TX, TY, es))
        ring = np.clip(em - cv2.erode(em, _K5), 0, 1)
        img += ring[..., None] * c_ring
        halo += blur(_half(em), 2.5)[..., None] * c_soft

    # --- 글로우 (투과광 후광) ---
    gsrc = tm_h * (0.55 + 0.45 * core_h) * (sh * boost)
    sw_h = None
    if sweeping:
        xs_h = (np.arange(W2, dtype=f32) * 2 + 1)[None, :]
        ys_h = (np.arange(H2, dtype=f32) * 2 + 1)[:, None]
        sw_h = _sweep_at(lf, xs_h, ys_h)
        gsrc = gsrc * (1 + sw_h * 1.2)
    halo += blur(gsrc, 2.5)[..., None] * (hexc("#fff0d0") * 0.45)
    halo += blur(gsrc, 10)[..., None] * (hexc("#ffa850") * 0.5)
    halo += blur(gsrc, 35)[..., None] * (hexc("#ff5a1e") * 0.55)

    # 후광은 타이틀 셀 '뒤'에 깔린다 (돌출 옆면이 빛을 가림)
    img += cv2.resize(halo, (W, H), interpolation=cv2.INTER_LINEAR)
    halo[:] = 0.0                              # 이후는 셀 '위'로 번지는 빛

    # --- 돌출 (빛을 받지 않는 어두운 옆면, 셀 2톤) — bbox 안에서만 ---
    R = img[tsl]
    if not impact:
        tone = blur((dep[tsl] < 0.45).astype(f32), 0.8)[..., None]
        ecol = hexc("#2a0207") * (1 - tone) + hexc("#6e0812") * tone
        ea_ = ext[tsl][..., None]
        R *= 1 - ea_
        R += ea_ * ecol

    # --- 타이틀 본체 (투과광 셀) ---
    tmr, cr = tm[tsl], core[tsl]
    cc = cr[..., None]
    col = CREAM_EDGE * (1 - cc) + WHITE_CORE * cc
    bright = (0.92 + 0.5 * cr) * _hotspot()[tsl] * (sh * boost)
    if sweeping:
        sw = _sweep_at(lf, np.arange(bx0, bx1, dtype=f32)[None, :], np.arange(by0, by1, dtype=f32)[:, None])
        bright = bright * (1 + 1.5 * sw)
    a = tmr[..., None]
    R *= 1 - a
    R += a * col * bright[..., None]
    # 빛이 매트 가장자리를 먹어 들어감 (투과광 번짐)
    fl = np.zeros((H, W), f32)
    fl[tsl] = tmr * bright
    fl_h = _half(fl)
    halo += blur(fl_h, 1.25)[..., None] * (hexc("#fff0d8") * 0.22)
    halo += blur(fl_h, 4.5)[..., None] * (hexc("#ffb060") * 0.16)
    if sw_h is not None:
        halo += (sw_h * blur(tm_h, 14))[..., None] * (hexc("#ffc070") * 0.9)

    # --- 불티 ---
    if not impact:
        _embers(img, halo, ctx, amt=1.0 if lf >= 2 else 0.5)

    # --- 자막 ---
    sb = _subs()
    if lf >= 3 and not impact:
        zb = {3: 2.0, 4: 1.4}.get(lf, 1.0)
        halo += blur(sb["z_h"], 1.5)[..., None] * (hexc("#ffcf90") * 0.55 * zb)
        halo += blur(sb["z_h"], 6)[..., None] * (hexc("#ff8a40") * 0.38 * zb)
        _over(img, *sb["z"], CREAM * 1.1 * zb)
        # 괘선이 중심에서 바깥으로 그어짐
        prog = ease_out(clamp((lf - 3) / 4.0), 3)
        lim = sb["half"] + 46 + 270 * prog
        rc, (y0, y1, x0, x1) = sb["rules"]
        win = (np.abs(np.arange(x0, x1, dtype=f32) - TX) < lim).astype(f32)[None, :]
        _over(img, rc * win, (y0, y1, x0, x1), CREAM * 1.05, 0.9)
        win_h = (np.abs(np.arange(W2, dtype=f32) * 2 - TX) < lim).astype(f32)[None, :]
        halo += blur(sb["rules_h"] * win_h, 2)[..., None] * (hexc("#ffb070") * 0.4)
    if lf >= 12:
        # 세로 태그라인: 위에서 아래로 한 글자씩 비쳐 나옴
        rev = clamp((lf - 12) / 8.0)
        ylim = 260 + 420 * ease_out(rev, 1.5)
        tc, (y0, y1, x0, x1) = sb["tag"]
        ysm = (np.arange(y0, y1, dtype=f32) < ylim).astype(f32)[:, None]
        _over(img, tc * ysm, (y0, y1, x0, x1), CREAM, 0.85)
        ysm_h = (np.arange(H2, dtype=f32) * 2 < ylim).astype(f32)[:, None]
        halo += blur(sb["tag_h"] * ysm_h, 1.5)[..., None] * (hexc("#ffb070") * 0.3)
    if lf >= 8:
        ca = {8: 0.8, 9: 0.12, 10: 1.0}.get(lf, 0.9)     # 네온처럼 켜짐-꺼짐-켜짐
        _over(img, *sb["cred"], CREAM, ca)
        halo += blur(sb["cred_h"], 1.5)[..., None] * (hexc("#ffb070") * 0.3 * ca)

    img += cv2.resize(halo, (W, H), interpolation=cv2.INTER_LINEAR)

    # --- 글린트 ---
    def to_screen(p):
        return (M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2])
    for (t0, dur, anchor, size, rot) in ((6, 7, a0, 300, 8.0), (16, 8, a1, 380, -6.0)):
        if t0 <= lf < t0 + dur:
            u = (lf - t0) / dur
            env = math.sin(math.pi * u ** 0.6)
            x, y = to_screen(anchor)
            _star(img, x, y, size * (0.6 + 0.5 * env), 2.4 * env, rot + 6 * u)
    if lf == 0:
        x, y = to_screen(a1)
        _star(img, x, y, 520, 2.0, 0)
    return img


def _black():
    return np.zeros((H, W, 3), f32)


def render(ctx: Ctx) -> np.ndarray:
    lf = ctx.lframe
    if lf >= T_BLACK:
        return _black()
    img = _compose(ctx)

    # --- 촬영: 푸시인 + 셰이크 ---
    push = 1.0 + 0.04 * ease_in_out(clamp((lf - 4) / 29.0))
    dx, dy, rot = _shake(lf)
    img = transform(img, dx, dy, 1.02 * push, rot, cx=TX, cy=TY + 60, border=cv2.BORDER_REFLECT)

    if lf == 0:
        img = zoom_blur(img, TX, TY, amount=0.11, steps=8)
        img = img * np.array([1.12, 1.0, 0.95], f32) + np.array([0.06, 0.02, 0.01], f32)

    # --- 필름 런아웃 ---
    if lf == T_RUN - 1:
        img = img * 0.9
    if T_RUN <= lf < T_BLACK:
        k = lf - T_RUN
        expo = [1.25, 0.55, 1.0, 1.1, 0.9][k]
        img = img * expo
        if k >= 3:   # 영사기 열로 초점이 나감
            img = blur(img, 2.0 + 2.5 * (k - 3))
        img = img + _leak(k)
        if k == 2:   # 프레임 슬립: 프레임 선(검은 띠)이 화면을 가로지르며 세로로 미끄러짐
            off = 380
            img = np.roll(img, off, axis=0)
            img[off - 44:off] = hexc("#060203")
            img = cv2.GaussianBlur(img, (1, 41), sigmaX=0.1, sigmaY=9)
        img = _burn(img, [0, 0, 0, 170, 760][k], k)
    return np.ascontiguousarray(img, dtype=f32)
