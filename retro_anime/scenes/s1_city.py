"""Scene 1 — 네온 메가시티 설정 샷 (72f, 0.0–3.0s).

멀티플레인 촬영대를 흉내 낸 틸트다운 + 약한 팬:
  L0 하늘(p=0.20)  : 울트라마린→보라→마젠타 스모그, 별, 거대한 창백한 달, 스모그 구름 띠
       + 비행선 셀(p=0.22, 광고 스크린/항법등) + 서치라이트 빔(투과광, 원경 지면에서 솟음)
  L1 원경(p=0.33)  : 수백 개의 빌딩 실루엣 + 중앙의 초고층 '메가타워'(크라운 링/비컨) + 스모그 글로우
  L2 중경(p=0.60)  : 수천 개의 창문 불빛, 옥상 물탱크/안테나/광고판, 고가 철도(열차 통과)
  L3 근경(p=1.00)  : 세로 네온 간판(「電脳」「残像」「楽園」「電脳街」「極楽」「ホテル」…), 항공 장애등, 전선
  L4 고가도로(p=1.15): 원근 곡선 고속도로, 나트륨 가로등, 붉은 테일라이트/흰 헤드라이트 강(1코마)
카메라: f0–6 검정에서 페이드인 → 천천히 출발해 가속하는 틸트다운, 끝에서 고속도로 높이로 내려와
         소실점(메가타워 밑동의 마젠타 글로우)으로 시선을 모아 s2 추격으로 넘긴다.
배경은 포스터컬러풍(그라디언트·붓결·에어브러시), 비행선/근경 차량은 셀(잉크+1단 그림자),
빛(네온·빔·창문·차량등)은 HDR 투과광.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from ..core import (FPS, H, W, Ctx, bezier, cel_shape, fill_ellipse, fill_poly, glow, hexc, ink,
                    noise, painted_texture, smoothstep, text_mask)

NAME = "s1_city"
FRAMES = 72

EVENTS = [
    (0, "swell"),        # 검정에서 도시 앰비언스 페이드인
    (10, "pulse"),       # 메가타워 크라운 펄스
    (20, "glitch"),      # 「ホテル」 네온관 지직거림
    (24, "blink"),       # 항공 장애등 동기 점멸
    (34, "pulse"),
    (36, "glitch"),      # 비행선 광고 스크린 전환 글리치
    (42, "whoosh"),      # 틸트 가속, 고가 철도 열차가 중앙 틈으로 진입
    (47, "glitch"),      # 네온관 다시 지직
    (48, "blink"),
    (52, "passby"),      # 열차 통과
    (58, "pulse"),
    (58, "whoosh"),      # 카메라가 고속도로 높이로 내려감
    (62, "passby"),      # 차 한 대가 화면 좌하단에서 튀어나와 소실점으로 질주
    (71, "swell"),       # 추격 컷으로
]

# ---------------------------------------------------------------------------
# 카메라 (멀티플레인)
# ---------------------------------------------------------------------------
D_TILT = 1250.0      # 근경(p=1) 기준 세로 이동량(px)
D_PAN = 200.0        # 근경 기준 가로 이동량(px) — 레이어가 왼쪽으로 흐름(카메라 우측 팬)
ZOOM = 0.05          # 근경 기준 T.U. 배율
CX, CY = W / 2.0, H / 2.0
P_SKY, P_SHIP, P_FAR, P_MID, P_NEAR, P_HWY = 0.20, 0.22, 0.33, 0.60, 1.0, 1.15


def _cam(lf):
    u = min(max(lf / (FRAMES - 1), 0.0), 1.0)
    tilt = 0.75 * u * u * (3 - 2 * u) + 0.25 * u * u      # 천천히 출발, 끝에서도 아직 내려가는 중
    pan = u * u * (3 - 2 * u)
    return tilt, pan


def _xf(p, lf):
    tilt, pan = _cam(lf)
    return D_PAN * p * pan, D_TILT * p * tilt, 1.0 + ZOOM * p * tilt


def _w2s(p, lf, x, y):
    ox, oy, z = _xf(p, lf)
    return (np.asarray(x, np.float64) - ox - CX) * z + CX, (np.asarray(y, np.float64) - oy - CY) * z + CY, z


# ---------------------------------------------------------------------------
# 팔레트
# ---------------------------------------------------------------------------
INK4 = (0.035, 0.02, 0.045, 1.0)
PANEL = hexc("#120d1c")
PANEL_FR = hexc("#3b3352")
N_PINK = np.array([3.0, 0.55, 1.55], np.float32)
N_CYAN = np.array([0.40, 2.5, 3.1], np.float32)
N_ORANGE = np.array([3.2, 1.25, 0.32], np.float32)
N_RED = np.array([3.3, 0.38, 0.30], np.float32)
N_GREEN = np.array([0.55, 3.0, 1.15], np.float32)
N_VIOLET = np.array([1.7, 0.65, 3.3], np.float32)
N_WHITE = np.array([2.7, 2.55, 2.35], np.float32)
SODIUM = np.array([3.2, 1.55, 0.5], np.float32)
TAIL = np.array([3.4, 0.32, 0.22], np.float32)
HEAD = np.array([3.0, 2.8, 2.35], np.float32)
AVI = np.array([3.6, 0.35, 0.22], np.float32)

WARM = [hexc(h) for h in ("#ffd27a", "#ffe2a8", "#ffb85c", "#fff0d2", "#ffc890", "#ffd9a0")]
COOL = [hexc(h) for h in ("#d2ecff", "#bdf4ff", "#eaf6ff", "#c4ffd6", "#a8c8ff")]
ODD = [hexc(h) for h in ("#ff9cc4", "#ff7060", "#9cffea", "#d0a0ff", "#ffef7a")]


# ---------------------------------------------------------------------------
# 레이어 캔버스 (프리멀티플라이드 RGBA, 월드 좌표)
# ---------------------------------------------------------------------------
class Layer:
    def __init__(self, p, x0, y0, x1, y1):
        self.p = p
        self.x0, self.y0 = int(math.floor(x0)), int(math.floor(y0))
        self.w = int(math.ceil(x1)) - self.x0
        self.h = int(math.ceil(y1)) - self.y0
        self.rgba = np.zeros((self.h, self.w, 4), np.float32)
        self.emis = np.zeros((self.h, self.w, 3), np.float32)
        self.blink = []          # (x, y, r, period, phase, on_len, col)
        self.jobs = []           # 셰이딩 후에 실행할 작업(창문/네온)
        self.extra = {}

    def P(self, pts):
        return np.asarray(pts, np.float64).reshape(-1, 2) - (self.x0, self.y0)

    def poly(self, pts, col, a=1.0):
        fill_poly(self.rgba, self.P(pts), (float(col[0]), float(col[1]), float(col[2]), 1.0), a)

    def rect(self, x0, y0, x1, y1, col, a=1.0):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], col, a)

    def addp(self, pts, col, a=1.0):
        fill_poly(self.rgba, self.P(pts), (float(col[0]), float(col[1]), float(col[2]), 0.0), a, mode="add")

    def ell(self, cx, cy, rx, ry, col, a=1.0, add=False, emit=0.0):
        c4 = (float(col[0]), float(col[1]), float(col[2]), 0.0 if add else 1.0)
        fill_ellipse(self.rgba, cx - self.x0, cy - self.y0, rx, ry, c4, a, mode="add" if add else "over")
        if emit:
            fill_ellipse(self.emis, cx - self.x0, cy - self.y0, rx, ry, np.asarray(col) * emit, a, mode="add")

    def line(self, pts, w, col, a=1.0, emit=0.0, closed=False):
        P = self.P(pts)
        ink(self.rgba, P, w, (float(col[0]), float(col[1]), float(col[2]), 1.0), (0, 0), a, closed=closed)
        if emit:
            ink(self.emis, P, w, np.asarray(col, np.float32) * emit, (0, 0), a, closed=closed)

    def emit_poly(self, pts, col, k=1.0):
        self.poly(pts, col)
        fill_poly(self.emis, self.P(pts), np.asarray(col, np.float32) * k, 1.0, mode="add")

    def sl(self, x0, y0, x1, y1):
        ix0 = max(0, int(round(x0 - self.x0)))
        ix1 = min(self.w, int(round(x1 - self.x0)))
        iy0 = max(0, int(round(y0 - self.y0)))
        iy1 = min(self.h, int(round(y1 - self.y0)))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return slice(iy0, iy1), slice(ix0, ix1)

    def ys(self):
        return np.arange(self.h, dtype=np.float32) + self.y0 + 0.5

    def xs(self):
        return np.arange(self.w, dtype=np.float32) + self.x0 + 0.5


def _shade_layer(L, y_top, y_bot, dark=0.45, under=None, u0=0.0, u1=1.0, upow=2.0):
    """레이어 전체 세로 명암(위는 어둡게) + 아래에서 올라오는 거리 불빛 반사."""
    ys = L.ys()
    k = dark + (1 - dark) * smoothstep(y_top, y_bot, ys)
    L.rgba[..., :3] *= k[:, None, None]
    if under is not None:
        g = smoothstep(u0, u1, ys) ** upow
        L.rgba[..., :3] += (L.rgba[..., 3] * g[:, None])[..., None] * np.asarray(under, np.float32)


def _light_pool(L, cx, cy, rx, ry, col, k=1.0, on_alpha=True):
    """빌딩 면에 번지는 색광(네온/거리 불빛 반사) — 알파가 있는 곳에만."""
    sl = L.sl(cx - rx * 2.2, cy - ry * 2.2, cx + rx * 2.2, cy + ry * 2.2)
    if sl is None:
        return
    ys = (np.arange(sl[0].start, sl[0].stop, dtype=np.float32) + L.y0 - cy) / ry
    xs = (np.arange(sl[1].start, sl[1].stop, dtype=np.float32) + L.x0 - cx) / rx
    g = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) * 1.2) * k
    if on_alpha:
        g = g * L.rgba[sl][..., 3]
    L.rgba[sl][..., :3] += g[..., None] * np.asarray(col, np.float32)


def _fog_layer(L, y_a, y_b, amt, col, base=0.0, pw=1.3, seed=0, patch=0.35, scale=2.0):
    """레이어 앞을 덮는 스모그(층 사이 헤이즈). 알파도 함께 채워 뒤 레이어를 가린다."""
    ys = L.ys()
    f = base + (amt - base) * smoothstep(y_a, y_b, ys) ** pw
    f2 = f[:, None] * np.ones((1, L.w), np.float32)
    if patch > 0:
        n = noise(L.w, L.h, scale, 0, 0, seed, beta=2.0, sx=5.0, sy=1.0)
        f2 = f2 * (1 - patch + patch * 2 * n)
    f2 = np.clip(f2, 0, 0.97)[..., None]
    L.rgba *= (1 - f2)
    L.rgba[..., :3] += f2 * np.asarray(col, np.float32)
    L.rgba[..., 3:] += f2


def _finish(L, texture=0.06, tex_seed=0, glow_s=(2.5, 9, 26), glow_w=(0.35, 0.22, 0.14), fog=None):
    for job in L.jobs:
        job()
    L.jobs = []
    if texture:
        painted_texture(L.rgba[..., :3], texture, seed=tex_seed, scale=2.5, stretch=7.0, angle=math.pi / 2)
    if fog is not None:
        _fog_layer(L, **fog)
    g = glow(L.emis, glow_s, glow_w)
    L.rgba[..., :3] += g
    L.emis = None
    return L


# ---------------------------------------------------------------------------
# 창문 (손으로 찍은 듯한 불규칙 불빛)
# ---------------------------------------------------------------------------
def _windows(L, x0, y0, x1, y1, cell, wsize, lit, temp, bright, seed, style="grid", sill=False):
    rng = np.random.default_rng(seed)
    cw, ch = cell
    ww, wh = wsize
    if style == "ribbon":
        ww = cw - 1
    elif style == "slit":
        ww, wh = max(1.0, cw * 0.32), ch * 0.82
    ncol = int((x1 - x0) // cw)
    nrow = int((y1 - y0) // ch)
    if ncol < 1 or nrow < 1:
        return
    bx = x0 + ((x1 - x0) - ncol * cw) * 0.5 + (cw - ww) * 0.5
    by = y0 + (ch - wh) * 0.5
    fp = np.clip(lit * (0.25 + 1.5 * rng.random(nrow)), 0, 0.95)
    fp[rng.random(nrow) < 0.14] *= 0.08
    fp[rng.random(nrow) < 0.07] = 0.95
    cm = np.clip(0.45 + 1.0 * rng.random(ncol), 0, 1.4)
    bh, bw = nrow // 3 + 2, ncol // 3 + 2
    blob = rng.random((bh, bw))
    blob = np.repeat(np.repeat(blob, 3, 0), 3, 1)[:nrow, :ncol]
    on = rng.random((nrow, ncol)) < fp[:, None] * cm[None, :] * (0.35 + 1.25 * blob)
    cool_floor = rng.random(nrow) < temp
    rr, cc = np.nonzero(on)
    if len(rr) == 0:
        return
    rnd = rng.random((len(rr), 5))
    rgba, emis = L.rgba, L.emis
    hh, wwid = L.h, L.w
    for i in range(len(rr)):
        r, c = rr[i], cc[i]
        q = rnd[i]
        if q[0] < 0.035:
            col = ODD[int(q[1] * len(ODD))]
        else:
            pal = COOL if (cool_floor[r] ^ (q[0] > 0.88)) else WARM
            col = pal[int(q[1] * len(pal))]
        b = bright * (0.3 + 1.05 * q[2] ** 1.6)
        ix0 = int(bx + c * cw - L.x0 + 0.5)
        iy0 = int(by + r * ch - L.y0 + 0.5)
        ix1 = ix0 + max(1, int(ww + (q[3] - 0.5) * 1.3 + 0.5))
        iy1 = iy0 + max(1, int(wh + (q[4] - 0.5) * 1.3 + 0.5))
        if ix0 < 0 or iy0 < 0 or ix1 > wwid or iy1 > hh:
            continue
        cb = col * b
        rgba[iy0:iy1, ix0:ix1, :3] = cb
        emis[iy0:iy1, ix0:ix1] += cb
        if sill and iy1 - iy0 >= 6:
            k = iy1 - max(2, int((iy1 - iy0) * 0.32))
            rgba[k:iy1, ix0:ix1, :3] *= 0.5
            if q[2] > 0.7:     # 블라인드 줄
                rgba[iy0 + 2:k:3, ix0:ix1, :3] *= 0.72


# ---------------------------------------------------------------------------
# 빌딩
# ---------------------------------------------------------------------------
def _tower(L, rng, x, w, top, bot, pal, side=1, sw=0.0, nset=0, crown="flat", cell=(11, 15),
           wsize=(5, 8), lit=0.45, temp=0.5, bright=1.3, wstyle="grid", rim=None, sc=1.0,
           blink_p=0.5, sill=False, ribs=0.0, win=True, bands=0.0, pil=0.0):
    """한 동의 빌딩: 세트백 층, 측면(딱딱한 그림자 톤), 림라이트, 옥상 구조물, 창문 작업 예약."""
    front, shade, hi = pal
    tiers = []
    xl, xr = x, x + w
    hgt = bot - top
    fr = sorted(rng.uniform(0.05, 0.42, nset).tolist(), reverse=True) if nset else []
    yb = bot
    for ytop in [top + hgt * f for f in fr] + [top]:
        tiers.append((xl, xr, ytop, yb))
        yb = ytop
        ins = (xr - xl) * rng.uniform(0.07, 0.2)
        a = rng.uniform(0.2, 0.8)
        xl, xr = xl + ins * a, xr - ins * (1 - a)
    for (a0, a1, yt, yb_) in tiers:
        L.rect(a0, yt, a1, yb_ + 1, front)
        swk = sw * (a1 - a0) / max(w, 1)
        if swk > 0.6:
            e = a1 if side > 0 else a0
            e2 = e + side * swk
            L.poly([(e, yt), (e2, yt + swk * 0.35), (e2, yb_ + 1), (e, yb_ + 1)], shade)
            if rim is not None:
                rw = max(1.0, 1.6 * sc)
                L.poly([(e2 - side * rw, yt + swk * 0.35), (e2, yt + swk * 0.35), (e2, yb_ + 1),
                        (e2 - side * rw, yb_ + 1)], rim)
        if ribs > 0 and (a1 - a0) > 30 * sc:
            n = int((a1 - a0) / (ribs * sc))
            for k in range(1, n):
                rx = a0 + (a1 - a0) * k / n
                L.rect(rx - 0.6 * sc, yt + 4 * sc, rx + 0.6 * sc, yb_, shade, 0.55)
        if pil > 0 and (a1 - a0) > pil * 1.5:
            n = int((a1 - a0) / pil)
            pw = 5.0 * sc
            for k in range(1, n):
                px = a0 + (a1 - a0) * k / n
                L.rect(px - pw, yt + 3 * sc, px + pw, yb_, front * 1.18)
                L.rect(px + pw * 0.35, yt + 3 * sc, px + pw, yb_, shade * 1.1)
                L.rect(px - pw, yt + 3 * sc, px - pw + max(1.0, 0.6 * sc), yb_, hi * 0.7, 0.6)
        if bands > 0:
            for yy in np.arange(yt + bands, yb_, bands):
                L.rect(a0, yy, a1, yy + max(1.0, 0.9 * sc), hi * 0.55, 0.45)
                L.rect(a0, yy + max(1.0, 0.9 * sc), a1, yy + max(2.0, 2.2 * sc), shade, 0.5)
        L.rect(a0 - 0.5, yt, a1 + 0.5, yt + max(1.2, 2.2 * sc), hi)
        if win:
            m = max(2.0, (a1 - a0) * 0.06)
            L.jobs.append(lambda a0=a0, a1=a1, yt=yt, yb_=yb_, m=m, s=int(rng.integers(1 << 30)):
                          _windows(L, a0 + m, yt + cell[1] * 0.8, a1 - m, yb_, cell, wsize, lit, temp,
                                   bright, s, wstyle, sill))
            if swk > cell[0] * 1.6:
                e = a1 if side > 0 else a0
                sx0, sx1 = (e + 1, e + swk - 1) if side > 0 else (e - swk + 1, e - 1)
                L.jobs.append(lambda sx0=sx0, sx1=sx1, yt=yt, yb_=yb_, s=int(rng.integers(1 << 30)):
                              _windows(L, sx0, yt + cell[1] * 1.2, sx1, yb_, (cell[0] * 0.7, cell[1]),
                                       (wsize[0] * 0.6, wsize[1]), lit * 0.8, temp, bright * 0.6, s,
                                       wstyle, False))
    a0, a1, yt, _ = tiers[-1]
    tw = a1 - a0
    _crown(L, rng, a0, a1, yt, crown, pal, sc, blink_p)
    # 옥상 잡동사니
    for _ in range(int(rng.integers(0, 3))):
        bw = tw * rng.uniform(0.08, 0.2)
        bh = bw * rng.uniform(0.4, 0.9)
        bx = a0 + rng.uniform(0.05, 0.9) * (tw - bw)
        L.rect(bx, yt - bh, bx + bw, yt + 1, shade)
        L.rect(bx, yt - bh, bx + bw, yt - bh + max(1.0, sc), hi)
    return tiers


def _crown(L, rng, a0, a1, yt, kind, pal, sc, blink_p):
    front, shade, hi = pal
    tw = a1 - a0
    if kind == "antenna":
        mx = a0 + tw * rng.uniform(0.25, 0.75)
        mh = max(18 * sc, tw * rng.uniform(0.6, 1.5))
        mw = max(0.8, 1.6 * sc)
        L.rect(mx - tw * 0.1, yt - tw * 0.07, mx + tw * 0.1, yt + 1, shade)
        L.poly([(mx - mw * 2.2, yt), (mx - mw * 0.5, yt - mh), (mx + mw * 0.5, yt - mh), (mx + mw * 2.2, yt)], shade)
        for k in range(3):
            yk = yt - mh * (0.35 + 0.2 * k)
            cw = mw * (5 - k)
            L.rect(mx - cw, yk, mx + cw, yk + max(0.8, sc), shade)
        L.blink.append((mx, yt - mh, 2.6 * sc + 1.0, 24, int(rng.integers(0, 2)) * 12, 9, AVI))
        if rng.random() < 0.5:
            L.blink.append((mx, yt - mh * 0.55, 1.8 * sc + 0.8, 24, int(rng.integers(0, 2)) * 12 + 3, 7, AVI))
    elif kind == "tank":
        tw_ = float(np.clip(tw * rng.uniform(0.22, 0.32), 5 * sc, 70 * sc))
        tx = a0 + rng.uniform(0.1, 0.6) * (tw - tw_)
        legs = tw_ * 0.45
        th = tw_ * 0.85
        for k in (0.1, 0.5, 0.9):
            L.rect(tx + tw_ * k - 0.7 * sc, yt - legs, tx + tw_ * k + 0.7 * sc, yt + 1, shade)
        L.line([(tx, yt - legs * 0.5), (tx + tw_, yt - legs * 0.1)], max(0.8, 0.9 * sc), shade)
        L.rect(tx, yt - legs - th, tx + tw_, yt - legs, front)
        L.rect(tx + tw_ * 0.55, yt - legs - th, tx + tw_, yt - legs, shade)
        L.poly([(tx - 1, yt - legs - th), (tx + tw_ * 0.5, yt - legs - th - tw_ * 0.28),
                (tx + tw_ + 1, yt - legs - th)], shade)
        L.rect(tx, yt - legs - th, tx + tw_, yt - legs - th + max(1.0, sc), hi)
        for k in range(1, 4):
            yy = yt - legs - th * k / 4
            L.rect(tx, yy, tx + tw_, yy + max(0.7, 0.6 * sc), shade, 0.8)
    elif kind == "step":
        n = int(rng.integers(2, 4))
        cx_ = a0 + tw * 0.5
        y = yt
        for k in range(n):
            hw = tw * (0.4 - 0.1 * k)
            hh = tw * 0.12
            L.rect(cx_ - hw, y - hh, cx_ + hw, y + 1, front)
            L.rect(cx_ + hw * 0.3, y - hh, cx_ + hw, y + 1, shade)
            L.rect(cx_ - hw, y - hh, cx_ + hw, y - hh + max(1.0, sc), hi)
            y -= hh
        if rng.random() < blink_p:
            L.blink.append((cx_, y - 2 * sc, 2.4 * sc + 1, 24, int(rng.integers(0, 2)) * 12, 9, AVI))
    elif kind == "slant":
        hh = tw * rng.uniform(0.25, 0.6)
        if rng.random() < 0.5:
            L.poly([(a0, yt + 1), (a0, yt - hh), (a1, yt), (a1, yt + 1)], front)
            L.line([(a0, yt - hh), (a1, yt)], max(1.0, 1.6 * sc), hi)
            bx, by = a0 + 2 * sc, yt - hh
        else:
            L.poly([(a0, yt + 1), (a0, yt), (a1, yt - hh), (a1, yt + 1)], shade)
            L.line([(a0, yt), (a1, yt - hh)], max(1.0, 1.6 * sc), hi)
            bx, by = a1 - 2 * sc, yt - hh
        if rng.random() < blink_p:
            L.blink.append((bx, by - 2 * sc, 2.4 * sc + 1, 24, int(rng.integers(0, 2)) * 12, 9, AVI))
    elif kind == "spire":
        cx_ = a0 + tw * 0.5
        hh = tw * rng.uniform(0.8, 1.6)
        L.poly([(a0 + tw * 0.2, yt + 1), (cx_ - 0.8 * sc, yt - hh), (cx_ + 0.8 * sc, yt - hh), (a1 - tw * 0.2, yt + 1)],
               front)
        L.poly([(cx_, yt + 1), (cx_, yt - hh), (cx_ + 0.8 * sc, yt - hh), (a1 - tw * 0.2, yt + 1)], shade)
        L.blink.append((cx_, yt - hh, 2.6 * sc + 1, 24, int(rng.integers(0, 2)) * 12, 9, AVI))
    elif kind == "billboard":
        bw = tw * (rng.uniform(0.6, 0.9) if sc >= 0.6 else rng.uniform(0.4, 0.6))
        bh = bw * rng.uniform(0.3, 0.45)
        bx = a0 + (tw - bw) * rng.uniform(0.2, 0.8)
        leg = bh * 0.35
        for k in np.linspace(0.08, 0.92, 4):
            L.rect(bx + bw * k - 0.8 * sc, yt - leg, bx + bw * k + 0.8 * sc, yt + 1, shade)
        L.line([(bx, yt - leg * 0.2), (bx + bw, yt - leg * 0.9)], max(0.7, 0.7 * sc), shade)
        L.rect(bx - sc, yt - leg - bh - sc, bx + bw + sc, yt - leg + sc, PANEL_FR * 0.6)
        lit = rng.random() < 0.8
        if lit:
            dimk = 0.45 if sc < 0.6 else 1.0
            c1 = [N_CYAN, N_PINK, N_ORANGE, N_VIOLET, N_GREEN][int(rng.integers(0, 5))] * 0.38 * dimk
            c2 = [N_CYAN, N_PINK, N_ORANGE, N_WHITE][int(rng.integers(0, 4))] * 0.55 * dimk
            L.jobs.append(lambda bx=bx, by=yt - leg - bh, bw=bw, bh=bh, c1=c1, c2=c2, s=int(rng.integers(1 << 30)):
                          _billboard(L, bx, by, bw, bh, c1, c2, s))
        else:
            L.rect(bx, yt - leg - bh, bx + bw, yt - leg, shade)
    if kind in ("flat", "billboard") and rng.random() < blink_p * 0.6:
        L.blink.append((a0 + 2 * sc, yt - 2 * sc, 2.0 * sc + 1, 24, int(rng.integers(0, 2)) * 12, 9, AVI))
        L.blink.append((a1 - 2 * sc, yt - 2 * sc, 2.0 * sc + 1, 24, int(rng.integers(0, 2)) * 12, 9, AVI))


KATA = ["サイバー", "ネオン", "デンキ", "ミライ", "ユメ", "ホシ", "ソーダ", "テレビ", "ラジオ", "カメラ"]
KANJI1 = ["酒", "薬", "宿", "夢", "光", "華", "楽", "湯", "質", "金", "愛", "麺", "銀", "電", "星"]


def _billboard(L, bx, by, bw, bh, c1, c2, seed):
    """옥상 광고판: 백라이트 패널 + 굵은 글자."""
    rng = np.random.default_rng(seed)
    sl = L.sl(bx, by, bx + bw, by + bh)
    if sl is None:
        return
    reg = L.rgba[sl]
    hh, ww = reg.shape[:2]
    yy = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
    col = c1[None, None, :] * (0.7 + 0.6 * yy)
    reg[..., :3] = col
    reg[..., 3] = 1
    L.emis[sl] += col * 0.6
    txt = KATA[int(rng.integers(0, len(KATA)))]
    size = max(8, int(bh * 0.55))
    m = text_mask(txt, size, stroke=max(0, size // 14))
    if m.shape[1] > ww * 0.92:
        sc = ww * 0.92 / m.shape[1]
        m = cv2.resize(m, (max(1, int(m.shape[1] * sc)), max(1, int(m.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    mh, mw = m.shape
    if mh >= hh or mw >= ww:
        return
    oy, ox = (hh - mh) // 2, (ww - mw) // 2
    sub = reg[oy:oy + mh, ox:ox + mw]
    a = m[..., None]
    sub[..., :3] = sub[..., :3] * (1 - a) + a * c2
    L.emis[sl][oy:oy + mh, ox:ox + mw] += a * c2 * 0.5


def _neon(L, text, cx, cy, size, tube, border=None, vertical=True, panel=True, pad=None, stroke=2,
          spill=0.22, dynamic=False, brackets=None):
    """네온 간판: 어두운 백패널 + 금속 테 + (선택) 테두리 관 + 글자 관(HDR, 흰 코어) + 벽면 색광.
    dynamic=True 면 글자 발광을 레이어에 굽지 않고 스프라이트로 반환(깜빡임용)."""
    m = text_mask(text, size, vertical=vertical, stroke=stroke)
    mh, mw = m.shape
    pad = size * 0.22 if pad is None else pad
    x0, y0 = cx - mw / 2.0, cy - mh / 2.0
    px0, py0, px1, py1 = x0 - pad, y0 - pad, x0 + mw + pad, y0 + mh + pad
    if spill:
        _light_pool(L, cx, cy, (px1 - px0) * 1.6, (py1 - py0) * 0.9, tube * 0.12, spill)
    if panel:
        if brackets is not None:
            bxw = brackets
            for yy in (py0 + (py1 - py0) * 0.12, py1 - (py1 - py0) * 0.12):
                if bxw > 0:
                    L.rect(px1, yy - 2, px1 + bxw, yy + 2, PANEL_FR * 0.7)
                else:
                    L.rect(px0 + bxw, yy - 2, px0, yy + 2, PANEL_FR * 0.7)
        L.rect(px0 - 2, py0 - 2, px1 + 2, py1 + 2, PANEL_FR)
        L.rect(px0, py0, px1, py1, PANEL)
        if border is not None:
            ins = max(3.0, pad * 0.3)
            L.line([(px0 + ins, py0 + ins), (px1 - ins, py0 + ins), (px1 - ins, py1 - ins), (px0 + ins, py1 - ins)],
                   max(2.0, size * 0.035), border, emit=0.9, closed=True)
    ix, iy = int(round(x0 - L.x0)), int(round(y0 - L.y0))
    if ix < 0 or iy < 0 or ix + mw > L.w or iy + mh > L.h:
        return None
    a = m[..., None]
    reg = L.rgba[iy:iy + mh, ix:ix + mw]
    core = cv2.erode(m, np.ones((3, 3), np.uint8))[..., None] if size > 40 else m[..., None] * 0.5
    lit = a * tube * 0.8 + core * np.array([0.7, 0.7, 0.7], np.float32)
    if dynamic:
        ghost = np.clip(tube / tube.max(), 0, 1) * 0.16 + 0.04
        reg[..., :3] = reg[..., :3] * (1 - a) + a * ghost
        reg[..., 3:] = np.maximum(reg[..., 3:], a)
        g = 60
        spr = np.zeros((mh + 2 * g, mw + 2 * g, 3), np.float32)
        spr[g:g + mh, g:g + mw] = lit
        e = np.zeros_like(spr)
        e[g:g + mh, g:g + mw] = a * tube
        spr += glow(e, (2.5, 9, 26), (0.45, 0.3, 0.18))
        return spr, x0 - g, y0 - g
    reg[..., :3] = reg[..., :3] * (1 - a) + lit
    reg[..., 3:] = np.maximum(reg[..., 3:], a)
    L.emis[iy:iy + mh, ix:ix + mw] += a * tube * 0.55
    return None


# ---------------------------------------------------------------------------
# L0 하늘
# ---------------------------------------------------------------------------
MOON = (1110.0, 300.0, 180.0)


@lru_cache(maxsize=1)
def _sky():
    p = P_SKY
    L = Layer(p, -60, -60, W + D_PAN * p + 60, H + D_TILT * p + 60)
    ys = L.ys()
    xs = L.xs()
    stops = [(-60, "#04051a"), (240, "#090b2e"), (520, "#15113f"), (760, "#2a1655"), (930, "#491e66"),
             (1080, "#782a72"), (1220, "#b0457e"), (1400, "#de6e8a")]
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([hexc(s[1]) for s in stops], np.float32)
    grad = np.stack([np.interp(ys, pos, cols[:, c]) for c in range(3)], 1)
    rgb = np.broadcast_to(grad[:, None, :], (L.h, L.w, 3)).copy()
    # 에어브러시: 메가타워 뒤쪽 하늘이 약간 더 밝게
    d = np.sqrt(((xs[None, :] - 1060) / 900) ** 2 + ((ys[:, None] - 1150) / 520) ** 2)
    rgb += np.exp(-d * d * 1.6)[..., None] * np.array([0.22, 0.05, 0.14], np.float32)
    # 별
    rng = np.random.default_rng(11)
    n = 900
    sx = rng.uniform(L.x0, L.x0 + L.w, n)
    sy = rng.uniform(L.y0, 900, n) ** 1.0
    keep = rng.random(n) < (1 - np.clip(sy / 900, 0, 1)) ** 1.4
    for x, y, b, r in zip(sx[keep], sy[keep], rng.uniform(0.25, 1.0, n)[keep], rng.uniform(0.5, 1.4, n)[keep]):
        fill_ellipse(rgb, x - L.x0, y - L.y0, r, r, np.array([0.75, 0.8, 1.0]) * b * (1 - y / 1000), 1.0, mode="add")
    # 스모그 구름 띠 (아랫면이 도시 불빛에 물듦) — 포스터컬러풍 층운
    def clouds(yc, sig, thr, seed, body, litc, amt, scale=8.0, stretch=6.0, k=0.45, rim=10):
        nn = noise(L.w, L.h, scale, 0, 0, seed, beta=2.0, sx=stretch, sy=1.0)
        nn2 = noise(L.w, L.h, scale * 0.35, 0, 0, seed + 3, beta=1.9, sx=stretch * 0.6, sy=1.0)
        row = noise(L.w, 64, 6.0, 0, 0, seed + 9, beta=2.4, sx=6.0, sy=1.0)[32]
        ycx = yc + (row - 0.5) * sig * 2.0
        band = np.exp(-((ys[:, None] - ycx[None, :]) / sig) ** 2)
        v = nn * 0.72 + nn2 * 0.28
        m = smoothstep(thr, thr + 0.06, v + (band - 1) * k)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        under = np.clip(m - np.roll(m, -rim, axis=0), 0, 1)
        top = np.clip(m - np.roll(m, rim // 2, axis=0), 0, 1)
        rgb[:] = rgb * (1 - m[..., None] * amt) + m[..., None] * amt * body
        rgb[:] += under[..., None] * litc + top[..., None] * np.array([0.10, 0.10, 0.18], np.float32)
    clouds(960, 120, 0.46, 21, hexc("#2c1a50"), np.array([0.62, 0.18, 0.34], np.float32), 0.85)
    clouds(700, 80, 0.47, 33, hexc("#1f1546"), np.array([0.45, 0.13, 0.30], np.float32), 0.8, 7.0, 7.0)
    # 달
    mx, my, mr = MOON
    sl = L.sl(mx - mr * 4, my - mr * 4, mx + mr * 4, my + mr * 4)
    yy = (np.arange(sl[0].start, sl[0].stop, dtype=np.float32) + L.y0 + 0.5)[:, None]
    xx = (np.arange(sl[1].start, sl[1].stop, dtype=np.float32) + L.x0 + 0.5)[None, :]
    dd = np.sqrt((xx - mx) ** 2 + (yy - my) ** 2)
    dfull = np.sqrt((xs[None, :] - mx) ** 2 + (ys[:, None] - my) ** 2)
    rgb += (np.exp(-np.maximum(dfull - mr, 0) / 70.0) * 0.55 +
            np.exp(-np.maximum(dfull - mr, 0) / 260.0) * 0.3)[..., None] * np.array([0.42, 0.30, 0.55], np.float32)
    reg = rgb[sl]
    disk = np.clip(mr - dd + 0.5, 0, 1)
    nx, ny = (xx - mx) / mr, (yy - my) / mr
    shade = smoothstep(0.1, 1.05, nx * 0.55 + ny * 0.75)
    hh, ww = reg.shape[:2]
    mar = cv2.GaussianBlur(noise(ww, hh, 0.5, 40, 70, 77, beta=2.8), (0, 0), 2.0)
    mar2 = cv2.GaussianBlur(noise(ww, hh, 0.2, 10, 20, 78, beta=2.4), (0, 0), 1.5)
    maria = smoothstep(0.53, 0.58, mar) * 0.6 + smoothstep(0.6, 0.66, mar2) * 0.3
    mc = np.array([1.0, 0.94, 0.88], np.float32) * 1.08
    mc_sh = np.array([0.80, 0.62, 0.78], np.float32) * 0.86
    mcol = mc * (1 - shade[..., None]) + mc_sh * shade[..., None]
    mcol = mcol * (1 - maria[..., None] * 0.38)
    limb = np.clip(1 - dd / mr, 0, 1) ** 0.3
    mcol *= (0.86 + 0.14 * limb)[..., None]
    reg[:] = reg * (1 - disk[..., None]) + disk[..., None] * mcol
    rgb[sl] = reg
    # 달 앞을 가로지르는 가는 구름
    clouds(390, 34, 0.47, 51, hexc("#251a4c"), np.array([0.38, 0.18, 0.34], np.float32), 0.9, 6.0, 8.0, 0.5, 7)
    clouds(215, 26, 0.50, 57, hexc("#1c1542"), np.array([0.24, 0.13, 0.28], np.float32), 0.8, 6.0, 9.0, 0.5, 6)
    painted_texture(rgb, 0.05, seed=5, scale=3.0, stretch=8.0)
    L.rgba[..., :3] = rgb
    L.rgba[..., 3] = 1.0
    L.emis = None
    return L


# ---------------------------------------------------------------------------
# L1 원경 + 메가타워
# ---------------------------------------------------------------------------
MEGA_X = 1060.0
MEGA_CROWN = (MEGA_X, 318.0)


def _megatower(L):
    cx = MEGA_X
    col = hexc("#2b1d50")
    sh = hexc("#1b1240")
    hi = hexc("#8a63b0")
    tiers = [(1000, L.y0 + L.h, 175), (700, 1000, 128), (480, 700, 92), (338, 480, 62), (262, 338, 38)]
    for i, (yt, yb, hw) in enumerate(tiers):
        L.poly([(cx - hw * 0.9, yt), (cx + hw * 0.9, yt), (cx + hw, yb), (cx - hw, yb)], col)
        L.poly([(cx + hw * 0.22, yt), (cx + hw * 0.9, yt), (cx + hw, yb), (cx + hw * 0.25, yb)], sh)
        # 링 플랫폼
        L.poly([(cx - hw * 1.3, yt - 5), (cx + hw * 1.3, yt - 5), (cx + hw * 1.2, yt + 7), (cx - hw * 1.2, yt + 7)],
               hi * 0.75)
        L.poly([(cx - hw * 1.2, yt + 4), (cx + hw * 1.2, yt + 4), (cx + hw * 1.2, yt + 7), (cx - hw * 1.2, yt + 7)], sh)
        for k in np.linspace(-1.15, 1.15, int(hw * 0.35) | 1):
            L.ell(cx + hw * k, yt + 7.5, 1.3, 1.3, np.array([2.4, 1.6, 2.6]), emit=0.8)
        # 창문 밴드
        def job(yt=yt, yb=yb, hw=hw, i=i):
            rng = np.random.default_rng(900 + i)
            for yy in np.arange(yt + 16, min(yb, 1500), 11.0):
                frac = (yy - yt) / max(1, yb - yt)
                w_ = hw * (0.9 + 0.1 * frac)
                if rng.random() < 0.18:
                    continue
                for xx in np.arange(cx - w_ + 5, cx + w_ - 5, 4.0):
                    if rng.random() < 0.55:
                        c = (WARM if rng.random() < 0.6 else COOL)[int(rng.integers(0, 5))] * rng.uniform(0.5, 1.1)
                        s_ = L.sl(xx, yy, xx + 2, yy + 3)
                        if s_ is not None:
                            L.rgba[s_][..., :3] = c * (0.55 if xx > cx + w_ * 0.22 else 1.0)
                            L.emis[s_] += c * 0.6
        L.jobs.append(job)
    # 수직 광선 띠
    for k, c in ((-0.55, N_CYAN * 0.55), (0.0, N_WHITE * 0.6), (0.55, N_CYAN * 0.45)):
        pts = []
        for (yt, yb, hw) in tiers[:4]:
            pts += [(cx + hw * k * 0.9, yt + 9), (cx + hw * k, min(yb, 1500) - 2)]
        L.jobs.append(lambda pts=pts, c=c: [L.line([pts[j], pts[j + 1]], 2.2, c, emit=1.0)
                                            for j in range(0, len(pts), 2)])
    # 첨탑
    L.poly([(cx - 9, 264), (cx - 1.2, 120), (cx + 1.2, 120), (cx + 9, 264)], col)
    L.poly([(cx, 264), (cx, 120), (cx + 1.2, 120), (cx + 9, 264)], sh)
    for yy in (170, 205, 240):
        L.rect(cx - 6 - (yy - 170) * 0.03, yy, cx + 6 + (yy - 170) * 0.03, yy + 2, sh)
    # 크라운 링(투과광)
    L.jobs.append(lambda: _crown_ring(L, cx))


def _crown_ring(L, cx):
    th = np.linspace(0, 2 * np.pi, 90)
    pts = np.stack([cx + 72 * np.cos(th), MEGA_CROWN[1] + 13 * np.sin(th)], 1)
    L.line(pts, 3.2, N_CYAN * 0.85, emit=1.3)


@lru_cache(maxsize=1)
def _far():
    p = P_FAR
    L = Layer(p, -60, 100, W + D_PAN * p + 60, H + D_TILT * p + 60)
    rng = np.random.default_rng(101)
    rows = [
        ((hexc("#4a3270"), hexc("#3b2860"), hexc("#7a5c9c")), (700, 850), (26, 80), 0.16, 0.85),
        ((hexc("#3c2a66"), hexc("#2d1f56"), hexc("#6a4f90")), (770, 910), (30, 95), 0.20, 0.9),
    ]
    front_row = ((hexc("#30225a"), hexc("#22184a"), hexc("#5c4384")), (840, 980), (34, 110), 0.25, 1.0)
    crowns = ["flat", "flat", "antenna", "step", "slant", "tank", "spire", "billboard"]

    def row(spec, seed):
        pal, (t0, t1), (w0, w1), lit, br = spec
        x = L.x0 - rng.uniform(0, 40)
        while x < L.x0 + L.w:
            w = rng.uniform(w0, w1)
            # 메가타워 근처는 조금 더 높게(도심부)
            boost = 90 * math.exp(-((x - MEGA_X) / 420) ** 2)
            top = rng.uniform(t0, t1) - boost - (rng.random() < 0.12) * rng.uniform(40, 140)
            side = 1 if x + w / 2 < MEGA_X else -1
            _tower(L, rng, x, w, top, L.y0 + L.h, pal, side=side, sw=w * rng.uniform(0.1, 0.22),
                   nset=int(rng.integers(0, 2)), crown=crowns[int(rng.integers(0, len(crowns)))],
                   cell=(5, 7), wsize=(2, 3), lit=lit, temp=rng.random(), bright=br, sc=0.45, blink_p=0.35,
                   wstyle=["grid", "grid", "ribbon", "slit"][int(rng.integers(0, 4))])
            x += w * rng.uniform(0.55, 1.05)

    for i, spec in enumerate(rows):
        row(spec, i)
    _megatower(L)
    row(front_row, 3)
    _shade_layer(L, 300, 1300, dark=0.55, under=np.array([0.40, 0.10, 0.22]), u0=900, u1=1500, upow=1.5)
    # 스카이라인 뒤 스모그 글로우 (빈 하늘 쪽 위주)
    ys = L.ys()
    gb = np.exp(-((ys - 860) / 210) ** 2)[:, None]
    a = L.rgba[..., 3]
    L.rgba[..., :3] += (gb * (1 - a * 0.7))[..., None] * np.array([0.42, 0.10, 0.26], np.float32)
    return _finish(L, texture=0.05, tex_seed=8,
                   fog=dict(y_a=880, y_b=1520, amt=0.82, col=np.array([0.62, 0.22, 0.48]), base=0.06, seed=12,
                            patch=0.3, scale=3.0))


# ---------------------------------------------------------------------------
# L2 중경
# ---------------------------------------------------------------------------
VIADUCT_Y = 1385.0


@lru_cache(maxsize=1)
def _mid():
    p = P_MID
    L = Layer(p, -60, 300, W + D_PAN * p + 60, H + D_TILT * p + 60)
    rng = np.random.default_rng(202)
    crowns = ["flat", "antenna", "tank", "step", "slant", "billboard", "tank", "antenna", "billboard"]

    def prof(x, lo, hi):
        c = smoothstep(480, 700, x) * (1 - smoothstep(1420, 1640, x))
        return lo * (1 - c) + hi * c

    specs = [
        ((hexc("#33275e"), hexc("#241b4c"), hexc("#6a5394")), (470, 720), (1000, 1160), (60, 150), 0.35, 1.05),
        ((hexc("#271e50"), hexc("#191340"), hexc("#5a4888")), (640, 880), (1140, 1290), (70, 170), 0.42, 1.25),
    ]
    for si, (pal, (e0, e1), (c0, c1), (w0, w1), lit, br) in enumerate(specs):
        x = L.x0 - rng.uniform(0, 60)
        while x < L.x0 + L.w:
            w = rng.uniform(w0, w1)
            top = prof(x + w / 2, rng.uniform(e0, e1), rng.uniform(c0, c1))
            side = 1 if x + w / 2 < 1000 else -1
            tiers = _tower(L, rng, x, w, top, L.y0 + L.h, pal, side=side, sw=w * rng.uniform(0.12, 0.26),
                           nset=int(rng.integers(0, 3)), crown=crowns[int(rng.integers(0, len(crowns)))],
                           cell=(10, 14), wsize=(5, 7), lit=lit * rng.uniform(0.7, 1.3), temp=rng.random(),
                           bright=br, sc=0.8, blink_p=0.55, rim=None if rng.random() < 0.5 else hexc("#ff5fa8") * 0.7,
                           wstyle=["grid", "grid", "grid", "ribbon", "slit"][int(rng.integers(0, 5))],
                           ribs=rng.choice([0, 0, 14, 22]))
            # 작은 세로 간판
            if si == 1 and rng.random() < 0.45 and w > 70 and tiers[0][2] + 60 < 1480:
                a0, a1, yt, yb = tiers[0]
                txt = KANJI1[int(rng.integers(0, len(KANJI1)))]
                if rng.random() < 0.5:
                    txt += KANJI1[int(rng.integers(0, len(KANJI1)))]
                col = [N_PINK, N_CYAN, N_ORANGE, N_GREEN, N_RED, N_VIOLET][int(rng.integers(0, 6))] * 0.8
                sx = a0 + 10 if side < 0 else a1 - 10
                sy = rng.uniform(max(yt + 60, 900), 1500)
                L.jobs.append(lambda txt=txt, sx=sx, sy=sy, col=col: _neon(L, txt, sx, sy, 30, col, vertical=True,
                                                                           spill=0.25))
            x += w * rng.uniform(0.6, 1.05)
    _shade_layer(L, 450, 1700, dark=0.5, under=np.array([0.45, 0.12, 0.25]), u0=1150, u1=1900, upow=1.6)
    # 고가 철도
    vy = VIADUCT_Y
    for x in np.arange(L.x0 + 40, L.x0 + L.w, 170.0):
        L.rect(x - 9, vy + 14, x + 9, L.y0 + L.h, hexc("#1a1436"))
        L.rect(x - 9, vy + 14, x - 6, L.y0 + L.h, hexc("#5a3a6a"))
        L.poly([(x - 30, vy + 14), (x + 30, vy + 14), (x + 9, vy + 36), (x - 9, vy + 36)], hexc("#1a1436"))
    L.rect(L.x0, vy, L.x0 + L.w, vy + 15, hexc("#2a2046"))
    L.rect(L.x0, vy, L.x0 + L.w, vy + 2, hexc("#8a6aa0"))
    L.rect(L.x0, vy + 11, L.x0 + L.w, vy + 15, hexc("#140f2a"))
    for x in np.arange(L.x0 + 10, L.x0 + L.w, 26.0):
        L.ell(x, vy + 13, 1.1, 1.1, np.array([2.6, 1.3, 0.5]), emit=0.6)
    for x in np.arange(L.x0, L.x0 + L.w, 8.0):
        L.rect(x, vy - 6, x + 0.9, vy, hexc("#2a2046"))
    L.rect(L.x0, vy - 7, L.x0 + L.w, vy - 5.8, hexc("#2a2046"))
    return _finish(L, texture=0.06, tex_seed=9,
                   fog=dict(y_a=1300, y_b=1920, amt=0.72, col=np.array([0.55, 0.18, 0.42]), base=0.04, seed=22,
                            patch=0.35, scale=2.5))


# ---------------------------------------------------------------------------
# L3 근경 (세로 네온 간판)
# ---------------------------------------------------------------------------
NEAR_PAL = (hexc("#1d1838"), hexc("#100c25"), hexc("#4c4078"))


@lru_cache(maxsize=1)
def _near():
    p = P_NEAR
    L = Layer(p, -60, 140, W + D_PAN * p + 60, H + D_TILT * p + 60)
    rng = np.random.default_rng(303)
    bot = L.y0 + L.h
    pal = NEAR_PAL
    rimP = hexc("#ff4f9a") * 0.42
    rimC = hexc("#3ff0ff") * 0.32
    kw = dict(cell=(20, 30), wsize=(10, 14), sc=1.6, sill=True, bright=1.15, bands=30)
    # 가운데 건물 먼저(뒤), 바깥쪽 건물이 앞
    _tower(L, rng, 762, 250, 1640, bot, pal, side=1, sw=34, crown="tank", lit=0.32, temp=0.3, rim=rimP,
           ribs=0, **kw)
    _tower(L, rng, 1330, 262, 1520, bot, pal, side=-1, sw=36, crown="slant", lit=0.36, temp=0.7, rim=rimC,
           blink_p=1.0, wstyle="ribbon", **kw)
    # 탑 B (좌중) — 옥상 「楽園」
    tB = _tower(L, rng, 200, 560, 1000, bot, pal, side=1, sw=40, nset=0, crown="flat", lit=0.3, temp=0.35,
                rim=rimP, pil=112, **kw)
    _tower(L, rng, 262, 430, 822, 1002, pal, side=1, sw=30, crown="flat", lit=0.3, temp=0.4, rim=rimP, **kw)
    # 탑 A (왼쪽 끝)
    _tower(L, rng, -80, 255, 450, bot, pal, side=1, sw=26, crown="tank", lit=0.34, temp=0.5, rim=rimC,
           wstyle="slit", **kw)
    # 탑 C (오른쪽)
    _tower(L, rng, 1590, 660, 380, bot, pal, side=-1, sw=44, nset=1, crown="step", lit=0.3, temp=0.55,
           rim=rimC * 0.4, pil=130, **kw)
    # 탑 C 안테나 마스트
    L.poly([(1830, 330), (1858, 160), (1862, 160), (1890, 330)], pal[1])
    for k, yy in enumerate((200, 240, 280, 315)):
        L.rect(1840 + k * 3 - 6, yy, 1880 - k * 3 + 6, yy + 3, pal[1])
    L.blink += [(1860, 158, 5.0, 24, 0, 9, AVI), (1860, 240, 3.6, 24, 12, 8, AVI)]
    # 탑 A 안테나
    L.poly([(70, 450), (84, 250), (87, 250), (100, 450)], pal[1])
    L.rect(66, 330, 104, 333, pal[1])
    L.blink += [(85, 248, 4.4, 24, 12, 9, AVI)]
    # 옥상 「楽園」 프레임
    for xx in np.arange(300, 660, 36.0):
        L.rect(xx - 2, 700, xx + 2, 824, pal[1])
    L.line([(300, 820), (660, 704)], 2.0, pal[1])
    L.line([(300, 704), (660, 820)], 2.0, pal[1])
    L.rect(292, 806, 668, 812, pal[1])
    # 전선 (중앙 틈을 가로지르는 늘어진 케이블)
    for (x0, y0, x1, y1, sag) in ((1008, 1720, 1334, 1690, 60), (1008, 1760, 1334, 1745, 75),
                                  (1008, 1905, 1334, 1880, 40), (760, 1150, 1000, 1660, 30)):
        tt = np.linspace(0, 1, 40)
        pts = np.stack([x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt + sag * 4 * tt * (1 - tt)], 1)
        L.line(pts, 2.2, hexc("#0a0814"))
    # 비상계단 (탑 E 측면)
    for yy in np.arange(1760, 2350, 54.0):
        L.rect(1012, yy, 1046, yy + 3, pal[1])
        L.line([(1014, yy + 3), (1044, yy + 51)], 1.6, pal[1])
    _shade_layer(L, 300, 2200, dark=0.42, under=np.array([0.5, 0.14, 0.24]), u0=1700, u1=2400, upow=2.0)
    _light_pool(L, 1000, 2350, 700, 260, np.array([0.5, 0.2, 0.25]), 0.6)

    # 파사드 잡동사니 (창문 위에 그려지도록 작업 큐에)
    for (x0, x1, y0, y1, n) in ((210, 740, 1040, 2380, 34), (270, 690, 860, 1000, 4), (-60, 160, 500, 2380, 12),
                                (775, 1000, 1700, 2380, 9), (1345, 1580, 1580, 2380, 10), (1600, 2230, 450, 2380, 42)):
        L.jobs.append(lambda x0=x0, x1=x1, y0=y0, y1=y1, n=n, sd=int(rng.integers(1 << 30)):
                      _clutter(L, np.random.default_rng(sd), x0, x1, y0, y1, n))
    # 중앙 틈(소실점 쪽)에서 오는 색광
    _light_pool(L, 760, 1900, 170, 600, np.array([0.55, 0.12, 0.35]), 0.5)
    _light_pool(L, 1335, 1950, 150, 520, np.array([0.15, 0.35, 0.55]), 0.4)
    _light_pool(L, 1600, 900, 160, 700, np.array([0.12, 0.3, 0.5]), 0.35)

    # 네온 (셰이딩 이후)
    dyn = {}

    def signs():
        _neon(L, "電脳", 112, 720, 112, N_PINK, border=N_CYAN * 0.8, brackets=-20)
        _neon(L, "楽園", 480, 752, 116, N_RED, panel=False, stroke=3, spill=0.3)
        _neon(L, "残像", 1700, 760, 150, N_CYAN, border=N_PINK * 0.8, brackets=-60)
        _neon(L, "電脳街", 790, 1560, 128, N_PINK, border=N_VIOLET * 0.8, brackets=-30)
        _neon(L, "麻雀", 795, 2090, 96, N_GREEN, border=N_ORANGE * 0.8, brackets=-32)
        _neon(L, "極楽", 1690, 1560, 132, N_ORANGE, border=N_PINK * 0.7, brackets=-50)
        _neon(L, "月光荘", 600, 1300, 64, N_VIOLET, border=None, vertical=False, spill=0.2)
        _neon(L, "酒場", 1100 - 120, 1990, 54, N_RED, border=N_WHITE * 0.5, vertical=False, spill=0.2)
        _neon(L, "薬", 1220, 2120, 60, N_GREEN, border=N_WHITE * 0.5)
        _neon(L, "ネオン", 2010, 1320, 70, N_PINK, border=N_CYAN * 0.6, vertical=False)
        _neon(L, "カラオケ", 2000, 2180, 64, N_CYAN, border=N_PINK * 0.6, vertical=False)
        # 대형 광고판 (탑 C 정면)
        _bigboard(L, 1790, 1760, 400, 230)
        dyn["hotel"] = _neon(L, "ホテル", 1300, 1830, 92, N_PINK, border=N_WHITE * 0.6, dynamic=True, brackets=40)
    L.jobs.append(signs)
    # 「極楽」 테두리 체이스 전구 위치
    L.extra["dyn"] = dyn
    _finish(L, texture=0.06, tex_seed=10,
            fog=dict(y_a=1950, y_b=bot, amt=0.6, col=np.array([0.78, 0.28, 0.40]), base=0.0, seed=33, patch=0.3,
                     scale=2.0))
    m = text_mask("極楽", 132, vertical=True, stroke=2)
    mh, mw = m.shape
    pad = 132 * 0.22 + 14
    bx0, by0, bx1, by1 = 1690 - mw / 2 - pad, 1560 - mh / 2 - pad, 1690 + mw / 2 + pad, 1560 + mh / 2 + pad
    bulbs = []
    per = 2 * ((bx1 - bx0) + (by1 - by0))
    n = int(per / 22)
    for i in range(n):
        d = per * i / n
        if d < bx1 - bx0:
            bulbs.append((bx0 + d, by0))
        elif d < (bx1 - bx0) + (by1 - by0):
            bulbs.append((bx1, by0 + d - (bx1 - bx0)))
        elif d < 2 * (bx1 - bx0) + (by1 - by0):
            bulbs.append((bx1 - (d - (bx1 - bx0) - (by1 - by0)), by1))
        else:
            bulbs.append((bx0, by1 - (d - 2 * (bx1 - bx0) - (by1 - by0))))
    L.extra["bulbs"] = bulbs
    return L


def _clutter(L, rng, x0, x1, y0, y1, n):
    """근경 파사드 디테일: 실외기, 꺼진 작은 간판, 배수관."""
    dark = hexc("#0d0a1c")
    mid = hexc("#2a2448")
    lite = hexc("#6a5a90")
    for _ in range(n):
        x = rng.uniform(x0, x1 - 30)
        y = rng.uniform(y0, y1)
        k = rng.random()
        if k < 0.55:
            w = rng.uniform(16, 28)
            h = w * 0.62
            L.rect(x, y, x + w, y + h, mid)
            L.rect(x + w * 0.6, y, x + w, y + h, dark)
            L.rect(x, y, x + w, y + 1.6, lite)
            L.ell(x + w * 0.3, y + h * 0.5, w * 0.2, w * 0.2, dark)
            L.rect(x + 2, y + h, x + 4, y + h + 6, dark)
        elif k < 0.78:
            w = rng.uniform(26, 60)
            h = rng.uniform(16, 26)
            c = [hexc("#241a36"), hexc("#1a2036"), hexc("#2e1a28")][int(rng.integers(0, 3))]
            L.rect(x - 2, y - 2, x + w + 2, y + h + 2, dark)
            L.rect(x, y, x + w, y + h, c)
            L.rect(x + 3, y + h * 0.35, x + w - 3, y + h * 0.35 + 2, c * 1.6)
        else:
            hh = rng.uniform(90, 300)
            L.rect(x, y, x + 4, y + hh, dark)
            L.rect(x, y, x + 1.2, y + hh, lite * 0.7)
            for yy in np.arange(y + 40, y + hh, 90):
                L.rect(x - 2, yy, x + 6, yy + 3, dark)


def _bigboard(L, x, y, w, h):
    """근경 대형 광고판: 백라이트 그라디언트 + 원형 심볼 + 글자 (오리지널)."""
    L.rect(x - 8, y - 8, x + w + 8, y + h + 8, PANEL_FR * 0.8)
    sl = L.sl(x, y, x + w, y + h)
    reg = L.rgba[sl]
    hh, ww = reg.shape[:2]
    yy = np.linspace(0, 1, hh, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, ww, dtype=np.float32)[None, :]
    c_top = np.array([0.15, 0.55, 0.85], np.float32)
    c_bot = np.array([0.95, 0.2, 0.6], np.float32)
    col = c_top * (1 - yy[..., None]) + c_bot * yy[..., None]
    d = np.sqrt((xx - 0.27) ** 2 * (ww / hh) ** 2 + (yy - 0.5) ** 2)
    sun = np.clip((0.36 - d) * 60, 0, 1)
    stripes = ((yy * 9 % 1.0) < 0.45) & (yy > 0.5)
    col = col * (1 - sun[..., None]) + sun[..., None] * np.where(stripes[..., None], c_bot * 0.5,
                                                                  np.array([1.3, 1.0, 0.45], np.float32))
    reg[..., :3] = col
    reg[..., 3] = 1
    L.emis[sl] += col * 0.3
    m = text_mask("未来都市", int(h * 0.26), stroke=2)
    mh, mw = m.shape
    ox, oy = int(ww * 0.52), int(hh * 0.18)
    mw2 = min(mw, ww - ox - 4)
    a = m[:, :mw2, None]
    sub = reg[oy:oy + mh, ox:ox + mw2]
    sub[..., :3] = sub[..., :3] * (1 - a) + a * np.array([1.5, 1.45, 1.4], np.float32)
    m2 = text_mask("ZANZO-88", int(h * 0.13), font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    mh2, mw2b = m2.shape
    oy2 = oy + mh + 14
    mw2b = min(mw2b, ww - ox - 4)
    a2 = m2[:, :mw2b, None]
    sub = reg[oy2:oy2 + mh2, ox:ox + mw2b]
    sub[..., :3] = sub[..., :3] * (1 - a2) + a2 * np.array([0.1, 0.05, 0.15], np.float32)
    # 조명등 (광고판 위 갓등)
    for k in np.linspace(0.1, 0.9, 5):
        L.rect(x + w * k - 3, y - 26, x + w * k + 3, y - 8, PANEL_FR)
        L.ell(x + w * k, y - 26, 7, 4, N_WHITE * 0.7, emit=0.8)


# ---------------------------------------------------------------------------
# L4 고가 고속도로 (원근 곡선)
# ---------------------------------------------------------------------------
HW_P = [(-300.0, 1130.0), (380.0, 960.0), (800.0, 720.0), (985.0, 612.0)]   # 종료 화면 좌표 기준
HW_OFF = (D_PAN * P_HWY, D_TILT * P_HWY)
ZF = 30.0
W0 = 760.0


@lru_cache(maxsize=1)
def _curve():
    B = bezier(*HW_P, n=2001)
    d = np.gradient(B, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    return np.linspace(0, 1, 2001), B, nrm


def _road(z, l, h=0.0):
    """도로 좌표 (깊이 z>=0.5, 횡 l∈[-1,1], 높이 h) → 고가도로 레이어 월드 좌표."""
    z = np.asarray(z, np.float64)
    l = np.asarray(l, np.float64)
    t, B, N = _curve()
    s = np.clip((1 - 1 / z) / (1 - 1 / ZF), -0.6, 1.0)
    # s<0 는 시작점에서 접선 방향으로 외삽
    bx = np.interp(s, t, B[:, 0])
    by = np.interp(s, t, B[:, 1])
    tx, ty = B[1] - B[0]
    tl = math.hypot(tx, ty)
    neg = np.minimum(s, 0)
    bx = bx + neg * tx / tl * 1500
    by = by + neg * ty / tl * 1500
    nx = np.interp(s, t, N[:, 0])
    ny = np.interp(s, t, N[:, 1])
    hw = W0 * 0.5 / z
    x = bx + nx * l * hw + HW_OFF[0]
    y = by + ny * l * hw - h / z + HW_OFF[1]
    return np.stack([x, y], -1)


def _zs(z0, z1, n):
    return z0 * (z1 / z0) ** (np.arange(n + 1) / n)


@lru_cache(maxsize=1)
def _hwy():
    p = P_HWY
    L = Layer(p, 120, 1880, 2340, 2640)
    bot = L.y0 + L.h
    fogc = np.array([0.62, 0.24, 0.44], np.float32)

    def fogmix(c, z, k=1.0):
        f = min(0.85, (1 - math.exp(-(z - 1) / 9.0)) * k)
        return np.asarray(c, np.float32) * (1 - f) + fogc * f

    # 교각 (데크 뒤)
    for z in _zs(1.05, 26, 18)[::-1]:
        top = _road(z, 0.55, -40)
        w = 110 / z
        c = fogmix(hexc("#241c36"), z, 0.8)
        L.poly([(top[0] - w / 2, top[1]), (top[0] + w / 2, top[1]), (top[0] + w / 2, bot), (top[0] - w / 2, bot)], c)
        L.poly([(top[0] + w * 0.2, top[1]), (top[0] + w / 2, top[1]), (top[0] + w / 2, bot), (top[0] + w * 0.2, bot)],
               fogmix(hexc("#6a3a58"), z, 0.8))
    zs = _zs(0.55, ZF, 90)
    # 데크 측면(파사드) — 근거리 쪽 가장자리
    for z0, z1 in zip(zs[:-1], zs[1:]):
        a, b = _road(z0, 1.0, 0), _road(z1, 1.0, 0)
        c_, d_ = _road(z1, 1.0, -80), _road(z0, 1.0, -80)
        L.poly([a, b, c_, d_], fogmix(hexc("#3c2a48"), z0))
        e, f = _road(z0, 1.0, -8), _road(z1, 1.0, -8)
        L.poly([a, b, f, e], fogmix(hexc("#9a6a78"), z0))
        g, h_ = _road(z0, 1.0, -62), _road(z1, 1.0, -62)
        L.poly([g, h_, c_, d_], fogmix(hexc("#1e1630"), z0))
    # 노면
    for z0, z1 in zip(zs[:-1], zs[1:]):
        quad = [_road(z0, -1.0), _road(z1, -1.0), _road(z1, 1.0), _road(z0, 1.0)]
        L.poly(quad, fogmix(hexc("#1f1a2e"), z0))
    # 가로등 빛 웅덩이 (노면 위 에어브러시 가산) — 노면 마스크로 클리핑
    lamp_z = _zs(0.9, 26, 22)
    road_m = np.zeros((L.h, L.w), np.uint8)
    deck = np.vstack([_road(zs, -1.0), _road(zs[::-1], 1.0)]) - (L.x0, L.y0)
    cv2.fillPoly(road_m, [np.round(deck * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    pools = np.zeros((L.h, L.w, 3), np.float32)
    for i, z in enumerate(lamp_z):
        side = 1 if i % 2 == 0 else -1
        c = _road(z, side * 0.62)
        f = 1 - min(0.75, (z - 1) / 30)
        for k in range(5):
            rr = 1 - k * 0.19
            fill_ellipse(pools, c[0] - L.x0, c[1] - L.y0, 230 / z * rr, 95 / z * rr,
                         np.array([0.10, 0.05, 0.02]) * f, 1.0, angle=-28, mode="add")
    pools = cv2.GaussianBlur(pools, (0, 0), 5)
    L.rgba[..., :3] += pools * (road_m.astype(np.float32) / 255.0)[..., None]
    # 차선
    for l in (-2 / 3, -1 / 3, 1 / 3, 2 / 3):
        zz = 0.6
        while zz < ZF:
            z1 = zz + 0.35 * (zz ** 0.5) * 0.5
            q = [_road(zz, l - 0.012), _road(z1, l - 0.012), _road(z1, l + 0.012), _road(zz, l + 0.012)]
            L.poly(q, fogmix(hexc("#b8a6b4"), zz) * 0.9)
            zz = z1 + 0.35 * (zz ** 0.5) * 0.6
    # 중앙 분리대
    for z0, z1 in zip(zs[:-1], zs[1:]):
        a, b = _road(z0, 0.0, 0), _road(z1, 0.0, 0)
        c_, d_ = _road(z1, 0.0, 16), _road(z0, 0.0, 16)
        L.poly([a, b, c_, d_], fogmix(hexc("#4a3a5a"), z0))
        e, f = _road(z0, 0.0, 13), _road(z1, 0.0, 13)
        L.poly([e, f, c_, d_], fogmix(hexc("#a88aa0"), z0))
    # 가장자리 방호벽
    for side in (-1.0, 1.0):
        for z0, z1 in zip(zs[:-1], zs[1:]):
            a, b = _road(z0, side, 0), _road(z1, side, 0)
            c_, d_ = _road(z1, side, 26), _road(z0, side, 26)
            col = hexc("#5a4466") if side < 0 else hexc("#3a2c4c")
            L.poly([a, b, c_, d_], fogmix(col, z0))
            e, f = _road(z0, side, 22), _road(z1, side, 22)
            L.poly([e, f, c_, d_], fogmix(hexc("#c49aa8"), z0))
    # 방호벽 반사경(델리네이터) — 소실점으로 이어지는 리딩 라인
    zz = 0.6
    while zz < ZF:
        for side in (-1.0, 1.0):
            q = _road(zz, side * 0.995, 24)
            r = max(0.5, 3.2 / zz)
            L.ell(q[0], q[1], r, r * 0.8, np.array([2.4, 1.3, 0.45]) * (1 - min(0.6, zz / 40)), emit=0.7)
        zz *= 1.11
    # 가로등 기둥 + 등
    lamps = []
    for i, z in enumerate(lamp_z):
        side = 1.0 if i % 2 == 0 else -1.0
        base = _road(z, side * 1.03, 20)
        top = _road(z, side * 1.03, 190)
        arm = _road(z, side * 0.72, 200)
        wcol = fogmix(hexc("#3a2a44"), z)
        L.line([base, top], max(1.2, 15 / z), wcol)
        L.line([base, top], max(0.6, 4 / z), fogmix(hexc("#a0607a"), z))
        L.line([top, arm], max(1.0, 9 / z), wcol)
        lamps.append((arm, z))
    for arm, z in lamps:
        r = max(1.4, 16 / z)
        L.ell(arm[0], arm[1] + r * 0.2, r * 1.6, r * 0.7, hexc("#2a2238"))
        L.ell(arm[0], arm[1] + r * 0.5, r * 1.3, r * 0.45, SODIUM * (1 - min(0.6, z / 40)), emit=1.4)
    L.extra["lamps"] = lamps
    _finish(L, texture=0.07, tex_seed=13, glow_s=(3, 12, 34), glow_w=(0.5, 0.35, 0.22))
    return L


# 차량 (테일라이트/헤드라이트의 강)
ZMIN, ZR = 0.5, 30.0
AWAY_LANES = (0.83, 0.5, 0.17)
ONC_LANES = (-0.17, -0.5, -0.83)


@lru_cache(maxsize=1)
def _cars():
    rng = np.random.default_rng(404)
    cars = []
    for away, lanes, v0 in ((True, AWAY_LANES, 2.1), (False, ONC_LANES, 2.6)):
        for li, l in enumerate(lanes):
            n = 38 if away else 24
            d0 = np.sort(rng.uniform(0, ZR, n))
            v = v0 * (1.0 + 0.12 * (li - 1)) * rng.uniform(0.95, 1.05)
            for d in d0:
                kind = "truck" if rng.random() < 0.12 else "car"
                cars.append(dict(away=away, l=l + rng.uniform(-0.03, 0.03), d0=float(d), v=v,
                                 kind=kind, body=rng.uniform(0.0, 1.0), brake=rng.random() < 0.15))
    # 히어로 차: f62 에 화면 좌하단에서 튀어나와 소실점으로 달려감
    t = 62 / FPS
    v = 3.2
    cars.append(dict(away=True, l=0.5, d0=float((1.25 - ZMIN - v * t) % ZR), v=v, kind="car", body=0.05,
                     brake=False))
    return cars


def _car_z(c, t):
    if c["away"]:
        return ZMIN + (c["d0"] + c["v"] * t) % ZR
    return ZMIN + (c["d0"] - c["v"] * t) % ZR


def _scr_at(pt, lf):
    ox, oy, z = _xf(P_HWY, lf)
    return np.array([(pt[0] - ox - CX) * z + CX, (pt[1] - oy - CY) * z + CY])


def _draw_cars(img, lights, lf):
    """고가도로 위 차량: 근거리는 셀 차체(잉크), 원거리는 빛 점과 짧은 잔상."""
    t = lf / FPS
    dt = 1.6 / FPS
    ox, oy, zc = _xf(P_HWY, lf)
    pox, poy, pzc = _xf(P_HWY, lf - 1.6)

    def scr(pt, prev=False):
        if prev:
            return np.stack([(pt[..., 0] - pox - CX) * pzc + CX, (pt[..., 1] - poy - CY) * pzc + CY], -1)
        return np.stack([(pt[..., 0] - ox - CX) * zc + CX, (pt[..., 1] - oy - CY) * zc + CY], -1)

    # 화면에 고속도로가 없으면 생략
    if scr(_road(ZF, 0.0))[1] > H + 40 and scr(_road(1.3, 0.0))[1] > H + 40:
        return
    items = []
    for c in _cars():
        z = _car_z(c, t)
        zp = _car_z(c, t - dt)
        if abs(z - zp) > 1.0:          # 순환 경계
            zp = z + (-c["v"] if c["away"] else c["v"]) * dt
        items.append((z, zp, c))
    items.sort(key=lambda it: -it[0])
    for z, zp, c in items:
        l = c["l"]
        fog = min(0.85, max(0.0, (1 - math.exp(-(z - 1) / 9.0))))
        if z < 7.0:
            _car_body(img, c, z, l, scr, fog)
        near_z = z - 0.045
        lc = TAIL * (1.35 if c["brake"] else 1.0) if c["away"] else HEAD
        gain = (1 - fog * 0.75)
        for s in (-1, 1):
            p_now = scr(_road(near_z, l + s * 0.06, 9))
            r = max(0.75, (6.5 if c["away"] else 8.0) / z) * zc
            if p_now[0] < -200 or p_now[0] > W + 200 or p_now[1] < -200 or p_now[1] > H + 200:
                continue
            # 잔상 리본: 과거 6코마 위치를 꼬리로
            tr = [p_now]
            for k in range(1, 7):
                zk = _car_z(c, t - k / FPS)
                if abs(zk - z) > 3.0:
                    break
                tr.append(scr(_road(zk - 0.045, l + s * 0.06, 9)))
            if len(tr) > 1:
                ink(lights, np.array(tr[::-1]), r * 1.7, lc * gain * (0.62 if c["away"] else 0.32), (0.95, 0.0))
            fill_ellipse(lights, p_now[0], p_now[1], r * 1.25, r, lc * gain, 1.0, mode="add")
            if z < 12:
                ink(lights, np.array([p_now + (0, r * 1.5), p_now + (0, r * 1.5 + 70.0 / z)]), r * 1.1,
                    lc * gain * 0.10, (0.1, 0.8))
        if c["kind"] == "truck" and z < 20:
            for k in np.linspace(-0.07, 0.07, 4):
                pp = scr(_road(z + 0.08, l + k, 30))
                rr = max(0.6, 3.0 / z) * zc
                fill_ellipse(lights, pp[0], pp[1], rr, rr, np.array([3.0, 1.6, 0.3]) * gain, 1.0, mode="add")


def _car_body(img, c, z, l, scr, fog):
    zn, zx = z - 0.05, z + (0.16 if c["kind"] == "truck" else 0.10)
    hw = 0.075
    fc = np.array([0.62, 0.24, 0.44], np.float32)
    base = [hexc("#2a2340"), hexc("#3a2030"), hexc("#d8c890") * 0.55, hexc("#202a40")][int(c["body"] * 4) % 4]
    base = base * (1 - fog) + fc * fog
    roofc = (base * 1.45 + np.array([0.10, 0.05, 0.07])) * (1 - fog * 0.5)
    ht = 34 if c["kind"] == "truck" else 13
    body = scr(np.array([_road(zn, l - hw, 2), _road(zx, l - hw, 2), _road(zx, l - hw, ht), _road(zx, l + hw, ht),
                         _road(zx, l + hw, 2), _road(zn, l + hw, 2), _road(zn, l + hw, ht), _road(zn, l - hw, ht)]))
    if body[:, 0].max() < -20 or body[:, 1].min() > H + 20:
        return
    lw = max(0.8, 4.0 / z)
    cel_shape(img, body, base, line=(0.035, 0.02, 0.045), line_w=lw)
    if c["kind"] == "truck":
        top = scr(np.array([_road(zn, l - hw, ht), _road(zx, l - hw, ht), _road(zx, l + hw, ht),
                            _road(zn, l + hw, ht)]))
        cel_shape(img, top, roofc * 0.8, line=(0.035, 0.02, 0.045), line_w=lw * 0.8)
    else:
        cab = scr(np.array([_road(zn + 0.025, l - hw * 0.8, ht), _road(zx - 0.035, l - hw * 0.8, ht),
                            _road(zx - 0.045, l - hw * 0.72, 24), _road(zn + 0.04, l - hw * 0.72, 24),
                            _road(zn + 0.04, l + hw * 0.72, 24), _road(zx - 0.045, l + hw * 0.72, 24),
                            _road(zx - 0.035, l + hw * 0.8, ht), _road(zn + 0.025, l + hw * 0.8, ht)]))
        cel_shape(img, cab, base * 0.7, line=(0.035, 0.02, 0.045), line_w=lw * 0.8)
        roof = scr(np.array([_road(zn + 0.04, l - hw * 0.72, 24), _road(zx - 0.045, l - hw * 0.72, 24),
                             _road(zx - 0.045, l + hw * 0.72, 24), _road(zn + 0.04, l + hw * 0.72, 24)]))
        cel_shape(img, roof, roofc, line=None)


# ---------------------------------------------------------------------------
# 비행선 (셀)
# ---------------------------------------------------------------------------
SHIP_CX, SHIP_CY, SHIP_A, SHIP_B = 330.0, 110.0, 270.0, 64.0


def _ship_x(lf):
    return 40.0 + 4.6 * lf


SHIP_Y = 120.0


@lru_cache(maxsize=3)
def _airship(state=0):
    w, h = 640, 250
    spr = np.zeros((h, w, 4), np.float32)
    cx, cy, a, b = SHIP_CX, SHIP_CY, SHIP_A, SHIP_B
    base = (*hexc("#463e66"), 1.0)
    shade = (*hexc("#262040"), 1.0)
    hi = (*hexc("#9a90c4"), 1.0)
    # 꼬리 날개 (뒤)
    fin_t = [(115, cy - 30), (52, cy - 100), (26, cy - 98), (60, cy - 22)]
    fin_b = [(115, cy + 26), (52, cy + 88), (26, cy + 86), (60, cy + 16)]
    for f in (fin_t, fin_b):
        cel_shape(spr, f, shade, line=INK4, line_w=2.2)
    th = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    hull = np.stack([cx + a * np.cos(th), cy + b * np.sin(th) * (0.72 + 0.28 * np.cos(th))], 1)
    sh_pts = [(cx - a - 20, cy + 14), (cx + a + 20, cy + 2), (cx + a + 20, cy + b + 30), (cx - a - 20, cy + b + 30)]
    # 하이라이트 초승달
    ht = np.linspace(np.pi + 0.45, 2 * np.pi - 0.3, 40)
    up = np.stack([cx + a * np.cos(ht), cy + b * np.sin(ht) * (0.72 + 0.28 * np.cos(ht))], 1)
    dn = up[::-1] + np.array([0, 9.0])
    cel_shape(spr, hull, base, shade=shade, shade_pts=sh_pts, hi=hi, hi_pts=np.vstack([up, dn]),
              line=INK4, line_w=2.8)
    # 골조 리브
    for x in np.arange(cx - a + 50, cx + a - 20, 42.0):
        u = (x - cx) / a
        bl = b * math.sqrt(max(0.0, 1 - u * u)) * (0.72 + 0.28 * u)
        yy = np.linspace(-bl * 0.96, bl * 0.96, 16)
        pts = np.stack([x + 7 * np.cos(yy / bl * 1.4), cy + yy], 1)
        ink(spr, pts, 1.1, (0.08, 0.05, 0.12, 1.0), (0.2, 0.2), 0.8)
    # 아랫면 림(도시 불빛)
    bt = np.linspace(0.25, np.pi - 0.25, 50)
    rim = np.stack([cx + a * 0.97 * np.cos(bt), cy - 2 + b * np.sin(bt) * (0.72 + 0.28 * np.cos(bt))], 1)
    ink(spr, rim, 2.4, (0.95, 0.32, 0.6, 1.0), (0.2, 0.3))
    # 옆 날개(앞)
    cel_shape(spr, [(120, cy + 6), (34, cy + 18), (22, cy + 30), (112, cy + 22)], base, line=INK4, line_w=2.0)
    # 곤돌라
    gon = [(300, cy + 52), (420, cy + 50), (408, cy + 80), (312, cy + 82)]
    cel_shape(spr, gon, shade, line=INK4, line_w=2.2)
    for k in range(6):
        x = 318 + k * 15
        fill_poly(spr, [(x, cy + 60), (x + 9, cy + 60), (x + 9, cy + 69), (x, cy + 69)], (2.2, 1.6, 0.8, 1.0))
    # 광고 스크린
    sx0, sy0, sx1, sy1 = 196, cy - 40, 448, cy + 16
    fill_poly(spr, [(sx0 - 4, sy0 - 4), (sx1 + 4, sy0 - 4), (sx1 + 4, sy1 + 4), (sx0 - 4, sy1 + 4)], INK4)
    ww, hh = int(sx1 - sx0), int(sy1 - sy0)
    scr_ = np.zeros((hh, ww, 3), np.float32)
    yy = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
    if state == 0:
        scr_[:] = np.array([0.05, 0.35, 0.5], np.float32) * (0.8 + 0.6 * yy)
        m = text_mask("月面旅行", 38, stroke=1)
        mh, mw = m.shape
        oy, ox = (hh - mh) // 2, ww - mw - 10
        scr_[oy:oy + mh, ox:ox + mw] = scr_[oy:oy + mh, ox:ox + mw] * (1 - m[..., None]) + m[..., None] * N_PINK * 0.85
        ys_, xs_ = np.mgrid[0:hh, 0:ww]
        moon = np.clip(16.5 - np.hypot(xs_ - 28, ys_ - hh / 2), 0, 1)
        scr_ = scr_ * (1 - moon[..., None]) + moon[..., None] * np.array([2.2, 2.0, 1.7], np.float32)
    elif state == 1:
        scr_[:] = np.array([0.6, 0.1, 0.35], np.float32) * (0.7 + 0.7 * yy)
        bands = (np.arange(ww)[None, :] // 9 % 2 == 0) & (np.arange(hh)[:, None] > hh * 0.62)
        scr_[bands] = np.array([2.4, 1.2, 0.3], np.float32)
        m = text_mask("未来へ", 34, stroke=1)
        mh, mw = m.shape
        oy, ox = 4, (ww - mw) // 2
        scr_[oy:oy + mh, ox:ox + mw] = scr_[oy:oy + mh, ox:ox + mw] * (1 - m[..., None]) + m[..., None] * N_WHITE * 0.85
    else:
        rng = np.random.default_rng(5)
        scr_[:] = rng.random((hh, 1, 1)).astype(np.float32) * np.array([0.5, 1.6, 1.8], np.float32)
        scr_ += rng.random((hh, ww, 1)).astype(np.float32) * 0.6
        for r0 in range(0, hh, 7):
            scr_[r0:r0 + 3] = np.roll(scr_[r0:r0 + 3], int(rng.integers(-30, 30)), axis=1)
    reg = spr[int(sy0):int(sy0) + hh, sx0:sx0 + ww]
    reg[..., :3] = scr_
    reg[..., 3] = 1.0
    e = np.zeros((h, w, 3), np.float32)
    e[int(sy0):int(sy0) + hh, sx0:sx0 + ww] = scr_ * 0.6
    spr[..., :3] += glow(e, (3, 10), (0.35, 0.2))
    return spr


# ---------------------------------------------------------------------------
# 합성 유틸
# ---------------------------------------------------------------------------
def _comp(img, L, lf):
    ox, oy, z = _xf(L.p, lf)
    a = 1.0 / z
    # 화면에서 이 레이어가 차지하는 영역만 워프
    sy0 = (L.y0 - oy - CY) * z + CY
    sy1 = (L.y0 + L.h - oy - CY) * z + CY
    sx0 = (L.x0 - ox - CX) * z + CX
    sx1 = (L.x0 + L.w - ox - CX) * z + CX
    r0, r1 = max(0, int(math.floor(sy0))), min(H, int(math.ceil(sy1)))
    c0, c1 = max(0, int(math.floor(sx0))), min(W, int(math.ceil(sx1)))
    if r1 <= r0 or c1 <= c0:
        return
    M = np.array([[a, 0, (1 - a) * CX + ox - L.x0 + a * c0], [0, a, (1 - a) * CY + oy - L.y0 + a * r0]], np.float32)
    out = cv2.warpAffine(L.rgba, M, (c1 - c0, r1 - r0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    reg = img[r0:r1, c0:c1]
    reg *= (1.0 - out[..., 3:4])
    reg += out[..., :3]


def _blit(img, spr, p, lf, wx, wy, k=1.0, premult=False):
    """작은 스프라이트(월드 좌표 좌상단 wx,wy)를 레이어 변환으로 화면에 합성."""
    ox, oy, z = _xf(p, lf)
    sx0 = (wx - ox - CX) * z + CX
    sy0 = (wy - oy - CY) * z + CY
    sh, sw = spr.shape[:2]
    bx0, by0 = max(0, int(math.floor(sx0))), max(0, int(math.floor(sy0)))
    bx1, by1 = min(W, int(math.ceil(sx0 + sw * z)) + 1), min(H, int(math.ceil(sy0 + sh * z)) + 1)
    if bx1 <= bx0 or by1 <= by0:
        return
    a = 1.0 / z
    M = np.array([[a, 0, (bx0 - sx0) * a], [0, a, (by0 - sy0) * a]], np.float32)
    out = cv2.warpAffine(spr, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT)
    reg = img[by0:by1, bx0:bx1]
    if premult:
        reg *= (1 - out[..., 3:4])
        reg += out[..., :3]
    else:
        reg += out * k


@lru_cache(maxsize=1)
def _half_grid():
    ys, xs = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
    return xs * 2 + 1, ys * 2 + 1


BEAMS = [
    dict(p=P_FAR, x=250, y=1010, a0=-0.38, amp=0.30, per=5.2, ph=0.3, len=2300, w0=7, w1=150,
         col=np.array([0.85, 0.95, 1.25], np.float32), I=0.55),
    dict(p=P_FAR, x=620, y=990, a0=0.28, amp=0.34, per=6.4, ph=2.1, len=2500, w0=7, w1=170,
         col=np.array([0.95, 0.9, 1.2], np.float32), I=0.5),
    dict(p=P_FAR, x=880, y=1050, a0=-0.08, amp=0.42, per=8.8, ph=4.0, len=2200, w0=6, w1=140,
         col=np.array([1.2, 0.55, 0.85], np.float32), I=0.4),
    dict(p=P_FAR, x=1480, y=1000, a0=-0.30, amp=0.32, per=5.8, ph=1.2, len=2400, w0=7, w1=160,
         col=np.array([0.85, 0.95, 1.25], np.float32), I=0.55),
    dict(p=P_FAR, x=1830, y=1030, a0=0.18, amp=0.28, per=7.0, ph=5.0, len=2300, w0=7, w1=150,
         col=np.array([1.15, 0.95, 0.75], np.float32), I=0.45),
    dict(p=P_MID, x=330, y=1150, a0=0.42, amp=0.22, per=6.0, ph=0.8, len=2600, w0=10, w1=190,
         col=np.array([0.8, 0.95, 1.3], np.float32), I=0.42),
    dict(p=P_MID, x=1660, y=1200, a0=-0.5, amp=0.2, per=7.5, ph=3.3, len=2600, w0=10, w1=190,
         col=np.array([0.8, 0.95, 1.3], np.float32), I=0.42),
]
MEGA_BEAMS = [
    dict(p=P_FAR, x=MEGA_X, y=MEGA_CROWN[1], a0=-0.72, amp=0.18, per=4.0, ph=0.0, len=2400, w0=4, w1=110,
         col=np.array([0.5, 1.2, 1.5], np.float32), I=0.5),
    dict(p=P_FAR, x=MEGA_X, y=MEGA_CROWN[1], a0=0.72, amp=0.18, per=4.0, ph=math.pi, len=2400, w0=4, w1=110,
         col=np.array([0.5, 1.2, 1.5], np.float32), I=0.5),
]


def _beams(img, lf, specs, extra=()):
    xs, ys = _half_grid()
    acc = np.zeros((H // 2, W // 2, 3), np.float32)
    t = lf / FPS
    smoke = noise(W // 2, H // 2, 0.45, 40 + lf * 0.9, 25 + lf * 0.25, 61, beta=1.7, sx=2.5, sy=1.0)
    for b in list(specs) + list(extra):
        sx, sy, z = _w2s(b["p"], lf, b["x"], b["y"])
        sx, sy = float(sx), float(sy)
        th = b["a0"] + b["amp"] * math.sin(2 * math.pi * t / b["per"] + b["ph"])
        dx, dy = math.sin(th), -math.cos(th)
        Lb = b["len"] * z
        # 화면과 겹치지 않으면 생략
        ex, ey = sx + dx * Lb, sy + dy * Lb
        if max(sx, ex) < -300 or min(sx, ex) > W + 300 or max(sy, ey) < -300 or min(sy, ey) > H + 300:
            continue
        rx = xs - sx
        ry = ys - sy
        al = rx * dx + ry * dy
        ac = np.abs(rx * dy - ry * dx)
        u = np.clip(al / Lb, 0, 1)
        wid = (b["w0"] + (b["w1"] - b["w0"]) * u) * z
        q = ac / wid
        fall = np.clip(al / 30.0, 0, 1) * (1 - u) ** 1.5
        I = (np.exp(-q * q * 2.0) * (0.45 + 1.0 * smoke) + np.exp(-q * q * 26.0) * 0.55) * fall * b["I"]
        acc += I[..., None] * b["col"]
    img += cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def _blinkers(lights, L, lf):
    for (x, y, r, per, ph, on, col) in L.blink:
        ph_ = (lf + ph) % per
        if ph_ >= on:
            continue
        k = 1.0 if ph_ < on - 2 else 0.45
        sx, sy, z = _w2s(L.p, lf, x, y)
        if -20 < sx < W + 20 and -20 < sy < H + 20:
            rr = r * z
            fill_ellipse(lights, float(sx), float(sy), rr, rr, col * k, 1.0, mode="add")


def _smoke_band(img, lf, p, ya, yb, col, amt, seed, drift=0.6):
    _, sa, z = _w2s(p, lf, 0, ya)
    _, sb, _ = _w2s(p, lf, 0, yb)
    sa, sb = float(sa), float(sb)
    if sa > H:
        return
    sw, sh = W // 8, H // 8
    ys = (np.arange(sh, dtype=np.float32) + 0.5) * 8
    f = smoothstep(sa, sb, ys) * amt
    n = noise(sw, sh, 0.22, lf * drift, (sa / 8.0), seed, beta=1.9, sx=4.0, sy=1.0)
    m = f[:, None] * np.clip(0.2 + 1.3 * n, 0, 1.4)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    img *= (1 - m)
    img += m * np.asarray(col, np.float32)


def _radial_add(img, cx, cy, rx, ry, col):
    x0, x1 = max(0, int(cx - rx * 2.2)), min(W, int(cx + rx * 2.2))
    y0, y1 = max(0, int(cy - ry * 2.2)), min(H, int(cy + ry * 2.2))
    if x1 <= x0 or y1 <= y0:
        return
    xs = ((np.arange(x0, x1, dtype=np.float32) + 0.5 - cx) / rx) ** 2
    ys = ((np.arange(y0, y1, dtype=np.float32) + 0.5 - cy) / ry) ** 2
    g = np.exp(-(ys[:, None] + xs[None, :]) * 1.1)
    img[y0:y1, x0:x1] += g[..., None] * col


PLUMES = [
    # 탑 B 옥상 환기구 — 「楽園」 붉은 네온에 물든 증기
    dict(p=P_NEAR, x=640.0, y=822.0, w=24.0, h=420.0, drift=230.0, rise=1.6, seed=81,
         lo=np.array([1.05, 0.52, 0.58], np.float32), hi=np.array([0.40, 0.27, 0.50], np.float32), a=0.8),
    # 탑 E 옥상 — 소실점 앞을 흐르는 분홍 증기
    dict(p=P_NEAR, x=930.0, y=1642.0, w=20.0, h=360.0, drift=200.0, rise=1.9, seed=83,
         lo=np.array([0.95, 0.55, 0.85], np.float32), hi=np.array([0.42, 0.30, 0.55], np.float32), a=0.75),
]


def _plume(img, lf, pl):
    """옥상 증기 기둥: 월드 좌표 스프라이트로 만들어 레이어 변환으로 합성(1코마로 피어오름)."""
    w, h, x, y = pl["w"], pl["h"], pl["x"], pl["y"]
    x0, x1 = x - w * 4, x + w * 4 + pl["drift"]
    y0 = y - h
    sx, sy, _ = _w2s(pl["p"], lf, np.array([x0, x1]), np.array([y0, y]))
    if sy[1] < -10 or sy[0] > H + 10 or sx[1] < -10 or sx[0] > W + 10:
        return
    q = 4
    gw, gh = int((x1 - x0) / q) + 1, int(h / q) + 1
    xs = x0 + (np.arange(gw, dtype=np.float32) + 0.5) * q
    ys = y0 + (np.arange(gh, dtype=np.float32) + 0.5) * q
    v = np.clip((y - ys) / h, 0, 1)[:, None]                        # 0=분출구, 1=위
    t = lf / FPS
    xc = x + pl["drift"] * v ** 1.6 + 10 * np.sin(v * 5.0 + t * 1.3)
    wid = w * (0.5 + 3.6 * v)
    d = (xs[None, :] - xc) / wid
    dens = np.exp(-d * d * 1.8) * (1 - v) ** 0.9 * smoothstep(0.0, 0.06, v)
    n = noise(gw, gh, 7.0 / q, -lf * 0.4, lf * pl["rise"], pl["seed"], beta=2.0)
    n2 = noise(gw, gh, 2.5 / q, -lf * 0.6, lf * pl["rise"] * 1.6, pl["seed"] + 1, beta=1.7)
    f = dens * (0.1 + 1.7 * n + 0.8 * (n2 - 0.5))
    a = np.clip(smoothstep(0.16, 0.42, f) * 0.75 + f * 0.35, 0, 1) * pl["a"]
    # 아랫면(네온 쪽)이 밝은 셀 스모크: 밀도 경계에 하이라이트
    rim = np.clip(f - np.roll(f, -2, axis=0), 0, 1) * 2.0
    col = pl["lo"][None, None, :] * (1 - v[..., None]) + pl["hi"][None, None, :] * v[..., None]
    spr = np.zeros((gh, gw, 4), np.float32)
    spr[..., :3] = (col + rim[..., None] * pl["lo"] * 0.8) * a[..., None]
    spr[..., 3] = a
    spr = cv2.resize(spr, (gw * q, gh * q), interpolation=cv2.INTER_LINEAR)
    _blit(img, spr, pl["p"], lf, x0, y0, premult=True)


def _train(img, lights, lf):
    """고가 철도 위 열차 (중경, 1코마 이동)."""
    head = 1380.0 - 40.0 * (lf - 40)
    ox, oy, z = _xf(P_MID, lf)
    if (VIADUCT_Y - oy - CY) * z + CY > H + 20:
        return
    car_l, gap, ncar = 66.0, 4.0, 9
    for k in range(ncar):
        x0 = head + k * (car_l + gap)
        x1 = x0 + car_l
        pts = np.array([(x0 + (6 if k == 0 else 0), VIADUCT_Y - 17), (x1, VIADUCT_Y - 17), (x1, VIADUCT_Y - 2),
                        (x0, VIADUCT_Y - 2), (x0, VIADUCT_Y - 11)])
        sx, sy, _ = _w2s(P_MID, lf, pts[:, 0], pts[:, 1])
        sp = np.stack([sx, sy], 1)
        if sp[:, 0].max() < 0 or sp[:, 0].min() > W:
            continue
        fill_poly(img, sp, hexc("#2c2446") * 0.9)
        for j in range(7):
            u0 = x0 + 6 + j * 8.3
            wx, wy, _ = _w2s(P_MID, lf, np.array([u0, u0 + 5, u0 + 5, u0]),
                             np.array([VIADUCT_Y - 13, VIADUCT_Y - 13, VIADUCT_Y - 8, VIADUCT_Y - 8]))
            fill_poly(img, np.stack([wx, wy], 1), np.array([1.25, 1.1, 0.8]) * (0.7 + 0.3 * ((j * 7 + k * 3) % 4 > 0)))
        sx2, sy2, _ = _w2s(P_MID, lf, np.array([x0, x1]), np.array([VIADUCT_Y - 17.5, VIADUCT_Y - 17.5]))
        ink(img, np.stack([sx2, sy2], 1), 1.2 * z, (0.6, 0.4, 0.7), (0, 0))
        if k == 0:
            hx, hy, _ = _w2s(P_MID, lf, x0 + 2, VIADUCT_Y - 8)
            fill_ellipse(lights, float(hx), float(hy), 3.0 * z, 2.2 * z, HEAD * 1.2, 1.0, mode="add")
            # 전조등 빔
            bx, by, _ = _w2s(P_MID, lf, np.array([x0, x0 - 260, x0 - 260]),
                             np.array([VIADUCT_Y - 9, VIADUCT_Y - 24, VIADUCT_Y + 6]))
            fill_poly(lights, np.stack([bx, by], 1), np.array([0.18, 0.17, 0.15]), 1.0, mode="add")


# ---------------------------------------------------------------------------
# 렌더
# ---------------------------------------------------------------------------
def _flush(img, lights, sig=(2.5, 9, 28), wts=(0.7, 0.45, 0.3)):
    """현재 깊이까지의 점광원(투과광)을 합성 — 다음(앞) 레이어가 가릴 수 있게 단계별로."""
    if not lights.any():
        return
    img += lights
    img += glow(lights, sig, wts)
    lights[:] = 0


def render(ctx: Ctx) -> np.ndarray:
    lf = ctx.lframe
    t = lf / FPS
    sky, far, mid, near, hwy = _sky(), _far(), _mid(), _near(), _hwy()
    img = np.zeros((H, W, 3), np.float32)
    lights = np.zeros((H, W, 3), np.float32)

    # L0 하늘
    _comp(img, sky, lf)
    # 서치라이트 (원경 뒤에서 솟음)
    ship_x = _ship_x(lf)
    ship_beam = dict(p=P_SHIP, x=ship_x + 360, y=SHIP_Y + SHIP_CY + 80, a0=math.pi - 0.3, amp=0.1, per=3.0,
                     ph=0.0, len=800, w0=5, w1=130, col=np.array([1.1, 1.0, 0.85], np.float32), I=0.22)
    _beams(img, lf, BEAMS, (ship_beam,))
    # 비행선
    state = 0 if lf < 36 else (2 if lf == 36 else 1)
    _blit(img, _airship(state), P_SHIP, lf, ship_x, SHIP_Y, premult=True)
    for (x, y, per, ph, col, r) in ((ship_x + 30, SHIP_Y + SHIP_CY - 98, 24, 0, AVI, 3.4),
                                    (ship_x + SHIP_CX + SHIP_A - 4, SHIP_Y + SHIP_CY, 16, 5, N_WHITE * 1.3, 3.0),
                                    (ship_x + 360, SHIP_Y + SHIP_CY + 84, 24, 12, AVI, 2.6)):
        if (lf + ph) % per < (3 if per == 16 else 8):
            sx, sy, z = _w2s(P_SHIP, lf, x, y)
            fill_ellipse(lights, float(sx), float(sy), r * z, r * z, col, 1.0, mode="add")
    _flush(img, lights)

    # L1 원경
    _comp(img, far, lf)
    _beams(img, lf, MEGA_BEAMS)
    # 메가타워 크라운 펄스 + 첨탑 비컨
    ph = (lf - 10) % 24
    pulse = math.exp(-ph / 3.0) if lf >= 10 else 0.0
    cx_, cy_, z = _w2s(P_FAR, lf, MEGA_CROWN[0], MEGA_CROWN[1])
    if pulse > 0.02:
        fill_ellipse(lights, float(cx_), float(cy_), 74 * z, 14 * z, N_CYAN * 0.5 * pulse, 1.0, mode="add")
        fill_ellipse(lights, float(cx_), float(cy_), 74 * z * (1 + (1 - pulse) * 0.6), 14 * z * (1 + (1 - pulse)),
                     N_CYAN * 0.25 * pulse, 1.0, mode="add")
    bx, by, _ = _w2s(P_FAR, lf, MEGA_X, 118)
    if lf % 24 in (10, 11, 14, 15):
        fill_ellipse(lights, float(bx), float(by), 4.0 * z, 4.0 * z, N_WHITE * 1.6, 1.0, mode="add")
    else:
        fill_ellipse(lights, float(bx), float(by), 2.6 * z, 2.6 * z, AVI * 0.8, 1.0, mode="add")
    _blinkers(lights, far, lf)
    _flush(img, lights)
    _smoke_band(img, lf, P_FAR, 900, 1300, np.array([0.6, 0.22, 0.46]), 0.35, 71, drift=0.5)

    # L2 중경
    _comp(img, mid, lf)
    # 소실점(메가타워 밑동)의 스모그 글로우 — 고속도로가 빨려드는 곳
    vx, vy, vz = _w2s(P_MID, lf, 1105, 1360)
    _radial_add(img, float(vx), float(vy), 640 * vz, 300 * vz, np.array([0.55, 0.15, 0.32], np.float32) * 0.75)
    _train(img, lights, lf)
    _blinkers(lights, mid, lf)
    _flush(img, lights)
    _smoke_band(img, lf, P_MID, 1350, 1800, np.array([0.62, 0.22, 0.42]), 0.45, 72, drift=0.9)

    # L3 근경
    _comp(img, near, lf)
    dyn = near.extra["dyn"].get("hotel")
    if dyn is not None:
        flick = {20: 0.12, 21: 0.6, 22: 0.0, 23: 0.85, 47: 0.0, 48: 0.5, 49: 0.05}.get(lf, 1.0)
        if flick > 0:
            spr, wx, wy = dyn
            _blit(img, spr, P_NEAR, lf, wx, wy, k=flick)
    step = lf // 2
    for i, (x, y) in enumerate(near.extra["bulbs"]):
        on = (i + step) % 3 == 0
        sx, sy, z = _w2s(P_NEAR, lf, x, y)
        if -10 < sx < W + 10 and -10 < sy < H + 10:
            fill_ellipse(lights, float(sx), float(sy), 3.4 * z, 3.4 * z,
                         np.array([3.0, 2.2, 1.0]) * (1.0 if on else 0.12), 1.0, mode="add")
    _blinkers(lights, near, lf)
    _flush(img, lights)
    for pl in PLUMES:
        _plume(img, lf, pl)

    # L4 고가도로
    _comp(img, hwy, lf)
    _draw_cars(img, lights, lf)

    _flush(img, lights, (2.5, 9, 30), (0.8, 0.5, 0.35))

    # 페이드 인
    if lf < 7:
        img *= (lf / 7.0) ** 1.6
    return img
