"""Scene 2 — 고속도로 바이크 추격 → 파워 슬라이드 정지 (72f, 3.0–6.0s).

컷 A (f0–33)  : 로우 앵글 추격 카메라(후방). 붉은 근미래 바이크가 고가 고속도로를 질주,
               나트륨 가로등·오버헤드 표지판이 휙휙 지나가고, 테일라이트가 빛의 리본(残像)으로 남는다.
               후방 추격자의 헤드라이트가 화면 좌하단에서 파고든다.
컷 B (f34–71) : 사이드 뷰 팔로 팬. 바이크가 옆으로 미끄러지며(파워 슬라이드) 불꽃·타이어 연기,
               정지 순간 카메라 셰이크 + 플래시, 마지막 ~10프레임 영웅 포즈 홀드. 붉은 잔광이 남는다.
바이크/라이더는 2코마 셀(line boil), 배경/카메라는 1코마.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from ..core import (H, W, Ctx, add_glow, blur, catmull, clamp, ease_in_out, ease_out, hexc, ink, jitter, lerp,
                    mix, painted_texture, place_mask, remap, smoothstep, text_mask, vgradient, wobble, hash01,
                    _fix, _SHIFT, _blend_region)

NAME = "s2_chase"
FRAMES = 72
CUT = 34            # 컷 B 시작 (장면 내부 프레임)
STOP = 58           # 바이크 정지 프레임

EVENTS = [
    (0, "whoosh"),       # 추격 시작 (엔진/풍절음)
    (2, "passby"),       # 가로등 (좌)
    (7, "passby"),       # 가로등 (우)
    (12, "passby"),      # 가로등 (좌)
    (13, "passby"),      # 추월하는 차
    (16, "whoosh"),      # 오버헤드 표지판 머리 위 통과
    (21, "passby"),
    (22, "flash"),       # 추격자 헤드라이트 프레임 인
    (26, "passby"),
    (30, "passby"),
    (34, "whoosh"),      # 사이드 뷰로 하드 컷
    (37, "passby"),      # 근경 기둥 와이프
    (40, "skid"),        # 파워 슬라이드 진입
    (42, "spark"),       # 발판 긁힘 불꽃 시작
    (45, "passby"),      # 근경 기둥 와이프
    (48, "spark"),
    (52, "spark"),
    (58, "impact"),      # 정지 + 플래시 + 셰이크
    (58, "shockwave"),
    (62, "swell"),       # 영웅 포즈
    (66, "blink"),       # 바이저 반짝
]

# ---------------------------------------------------------------------------
# 팔레트
# ---------------------------------------------------------------------------
INK = (0.035, 0.02, 0.045)
RED = hexc("#d4141f")
RED_SH = hexc("#6a0a24")
RED_HI = hexc("#ff8a72")
DARK = hexc("#22202e")
DARK_SH = hexc("#0e0d16")
DARK_HI = hexc("#565a78")
TYRE = hexc("#141219")
DISC = hexc("#3a3c52")
DISC_SH = hexc("#20212f")
JACKET = hexc("#2c3150")
JACKET_SH = hexc("#151729")
JACKET_HI = hexc("#6a7bb0")
PANTS = hexc("#23263c")
BOOT = hexc("#121019")
HELM = hexc("#2e3046")
HELM_SH = hexc("#14141f")
WHITE = hexc("#f1ece2")
CYAN = hexc("#3ff0ff")
SODIUM = hexc("#ff8a2a")
TAIL_HDR = np.array([3.4, 0.30, 0.22], np.float32)
TAIL_CORE = np.array([4.0, 1.6, 1.3], np.float32)
TRAIL_CORE = np.array([3.6, 0.75, 0.6], np.float32)
HEAD_HDR = np.array([3.0, 3.4, 3.8], np.float32)


# ---------------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------------
def _bb(P, pad, w=W, h=H):
    P = np.asarray(P, np.float64).reshape(-1, 2)
    x0 = int(max(0, math.floor(P[:, 0].min() - pad)))
    y0 = int(max(0, math.floor(P[:, 1].min() - pad)))
    x1 = int(min(w, math.ceil(P[:, 0].max() + pad) + 1))
    y1 = int(min(h, math.ceil(P[:, 1].max() + pad) + 1))
    return x0, y0, x1, y1


def _shift(m, dx, dy):
    """uint8 마스크를 정수 픽셀만큼 이동 (빈 곳 0)."""
    out = np.zeros_like(m)
    h, w = m.shape
    dx, dy = int(round(dx)), int(round(dy))
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = m[ys0:ys1, xs0:xs1]
    return out


def cel(img, P, base, shade=None, shade_P=None, hi=None, hi_P=None, rim=None, rim_dir=(1, -1),
        rim_w=4.0, line_w=3.0, alpha=1.0, sil=None, line=INK, closed=True, light=None,
        light_col=None, press=None):
    """셀 한 장 (이미 화면 좌표로 변환된 다각형 P).

    shade_P / hi_P : 딱딱한 그림자/하이라이트 다각형 (P 안으로 클리핑).
    rim            : 림라이트 색 — rim_dir 방향 가장자리에 rim_w 폭의 딱딱한 띠.
    sil            : (H,W) float 실루엣 마스크에 누적 (잔상/그림자용).
    light          : HDR 라이트 버퍼 — light_col 이면 이 셀 전체를 발광체로 가산.
    """
    P = np.asarray(P, np.float64)
    h, w = img.shape[:2]
    pad = 6 + line_w + rim_w
    x0, y0, x1, y1 = _bb(P, pad, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [_fix(P, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    _blend_region(img, m, x0, y0, base, alpha, "over")
    for col, sp in ((shade, shade_P), (hi, hi_P)):
        if col is None or sp is None:
            continue
        sm = np.zeros_like(m)
        sps = sp if isinstance(sp, list) else [sp]
        cv2.fillPoly(sm, [_fix(np.asarray(s, np.float64), x0, y0) for s in sps], 255, cv2.LINE_AA, _SHIFT)
        sm = cv2.min(sm, m)
        _blend_region(img, sm, x0, y0, col, alpha, "over")
    if rim is not None:
        d = np.asarray(rim_dir, np.float64)
        d = d / (np.linalg.norm(d) + 1e-9)
        rm = cv2.subtract(m, _shift(m, -d[0] * rim_w, -d[1] * rim_w))
        _blend_region(img, rm, x0, y0, rim, alpha, "over")
    if light is not None and light_col is not None:
        _blend_region(light, m, x0, y0, light_col, alpha, "add")
    if sil is not None:
        reg = sil[y0:y1, x0:x1]
        np.maximum(reg, m.astype(np.float32) * (alpha / 255.0), out=reg)
    if line is not None and line_w > 0:
        if press is None:
            press = 0.75 + 0.25 * np.sin(np.linspace(0, 6.28 * 2.5, len(P)) + len(P))
        ink(img, P, line_w, line, (0, 0), alpha=alpha, closed=closed, pressure=press)


def poly_fill(img, P, col, alpha=1.0, mode="over"):
    P = np.asarray(P, np.float64)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _bb(P, 2, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [_fix(P, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    _blend_region(img, m, x0, y0, col, alpha, mode)


def mask_poly(mask, P, val=1.0, mode="max"):
    """float 마스크(H,W)에 다각형을 max/add 로 그린다."""
    P = np.asarray(P, np.float64)
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = _bb(P, 2, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [_fix(P, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    reg = mask[y0:y1, x0:x1]
    mf = m.astype(np.float32) * (val / 255.0)
    if mode == "add":
        reg += mf
    else:
        np.maximum(reg, mf, out=reg)


def circ(cx, cy, r, n=40, a0=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + a0
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)


def ellip(cx, cy, rx, ry, rot=0.0, n=40):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(th), ry * np.sin(th)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)


def capsule(p0, p1, r0, r1, n=10):
    """두 점을 잇는 끝이 둥근 테이퍼 팔다리 다각형."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    d = p1 - p0
    L = np.hypot(*d) + 1e-9
    a = math.atan2(d[1], d[0])
    pts = []
    for th in np.linspace(-np.pi / 2, np.pi / 2, n):     # p1 쪽 반원
        pts.append(p1 + r1 * np.array([math.cos(a + th), math.sin(a + th)]))
    for th in np.linspace(np.pi / 2, 3 * np.pi / 2, n):  # p0 쪽 반원
        pts.append(p0 + r0 * np.array([math.cos(a + th), math.sin(a + th)]))
    return np.array(pts)


def limb(p0, p1, r0, r1, bulge=0.0, side=1, n=10):
    """테이퍼 + 한쪽 근육 볼록한 팔다리 다각형."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    d = p1 - p0
    L = np.hypot(*d) + 1e-9
    nrm = np.array([-d[1], d[0]]) / L
    t = np.linspace(0, 1, n)[:, None]
    c = p0 + d * t
    base = r0 + (r1 - r0) * t
    b = bulge * np.sin(np.pi * t) ** 1.2
    left = c + nrm * (base + (b if side > 0 else 0))
    right = c - nrm * (base + (b if side < 0 else 0))
    a = math.atan2(d[1], d[0])
    cap1 = [p1 + r1 * np.array([math.cos(a + th), math.sin(a + th)]) for th in np.linspace(np.pi / 2, -np.pi / 2, 7)[1:-1]]
    cap0 = [p0 + r0 * np.array([math.cos(a + th), math.sin(a + th)]) for th in np.linspace(-np.pi / 2, -3 * np.pi / 2, 7)[1:-1]]
    return np.r_[left, np.array(cap1), right[::-1], np.array(cap0)]


def smooth(P, n=6, closed=True):
    return catmull(P, n, closed)


def lerp_pts(a, b, k):
    return np.asarray(a, np.float64) * (1 - k) + np.asarray(b, np.float64) * k


def _area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ink_mask_outline(img, m, x0, y0, width, color=INK, alpha=1.0, min_len=12):
    """uint8 마스크 외곽선을 붓펜 선으로."""
    cs, _ = cv2.findContours((m > 110).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        if len(c) < min_len:
            continue
        c = c[:, 0, :].astype(np.float64)
        k = 5
        cp = np.r_[c[-k:], c, c[:k]]
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        c = np.stack([np.convolve(cp[:, j], ker, "same")[k:-k] for j in range(2)], 1)
        c = c[::3] + (x0 + 0.5, y0 + 0.5)
        if len(c) < 4:
            continue
        ink(img, c, width, color, (0, 0), alpha=alpha, closed=True,
            pressure=0.75 + 0.25 * np.sin(np.linspace(0, 18, len(c))))


def extrude(img, P3, outline, w_far, w_near, face, wall_lit, wall_dark, wall_mid=None, seed=0,
            boil=0.0, shade=None, shade_uv=None, hi=None, hi_uv=None, rim=None, rim_dir=(-0.4, -1),
            rim_w=4.0, line_w=4.0, inner_w=2.2, sil=None, special=None, alpha=1.0):
    """프로파일(u,v; v 위, 반시계) 을 w_far..w_near 로 압출한 셀 솔리드.

    가까운 면(face) + 카메라를 향한 옆벽(위쪽=wall_lit, 아래쪽=wall_dark) + 외곽 잉크.
    special(i, nu, nv) -> color|None : 특정 벽 색 지정(테일램프 단면 등).
    """
    O = np.asarray(outline, np.float64)
    n = len(O)
    N = P3(np.c_[O, np.full(n, float(w_near))], seed, boil)
    F = P3(np.c_[O, np.full(n, float(w_far))], seed + 1, boil)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _bb(np.r_[N, F], 8 + line_w + rim_w, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    wall_mid = mix(wall_lit, wall_dark, 0.5) if wall_mid is None else wall_mid
    sgn = np.sign(_area(N))
    U = np.zeros((y1 - y0, x1 - x0), np.uint8)
    e = np.roll(O, -1, 0) - O
    nrm = np.stack([e[:, 1], -e[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    groups = {}
    for i in range(n):
        j = (i + 1) % n
        q = np.array([N[i], N[j], F[j], F[i]])
        a = _area(q)
        if a * sgn >= -0.5:          # 뒷면 → 컬링
            continue
        col = None
        if special is not None:
            col = special(i, nrm[i, 0], nrm[i, 1])
        if col is None:
            nv = nrm[i, 1]
            col = wall_lit if nv > 0.35 else (wall_dark if nv < -0.25 else wall_mid)
        groups.setdefault(id(col), (col, []))[1].append(_fix(q, x0, y0))
    for col, qs in groups.values():
        m = np.zeros_like(U)
        cv2.fillPoly(m, qs, 255, cv2.LINE_8, _SHIFT)
        m = cv2.dilate(m, np.ones((2, 2), np.uint8))
        _blend_region(img, m, x0, y0, col, alpha, "over")
        U = cv2.max(U, m)
    fm = np.zeros_like(U)
    cv2.fillPoly(fm, [_fix(N, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    _blend_region(img, fm, x0, y0, face, alpha, "over")
    for col, uv in ((shade, shade_uv), (hi, hi_uv)):
        if col is None or uv is None:
            continue
        uvs = uv if isinstance(uv, list) else [uv]
        sm = np.zeros_like(U)
        for k_, a_ in enumerate(uvs):
            a_ = np.asarray(a_, np.float64)
            S = P3(np.c_[a_, np.full(len(a_), float(w_near))], seed + 3 + k_, boil * 0.6)
            cv2.fillPoly(sm, [_fix(S, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
        sm = cv2.min(sm, fm)
        _blend_region(img, sm, x0, y0, col, alpha, "over")
    U = cv2.max(U, fm)
    if rim is not None:
        d = np.asarray(rim_dir, np.float64)
        d = d / (np.linalg.norm(d) + 1e-9)
        rm = cv2.subtract(U, _shift(U, -d[0] * rim_w, -d[1] * rim_w))
        _blend_region(img, rm, x0, y0, rim, alpha, "over")
    if sil is not None:
        reg = sil[y0:y1, x0:x1]
        np.maximum(reg, U.astype(np.float32) * (alpha / 255.0), out=reg)
    if inner_w > 0:
        ink(img, N, inner_w, INK, (0, 0), alpha=alpha, closed=True)
    if line_w > 0:
        _ink_mask_outline(img, U, x0, y0, line_w, INK, alpha)
    return N


# ---------------------------------------------------------------------------
# 바이크 (사이드 뷰) — 로컬 좌표 u(앞)+, v(위)+, w(카메라 쪽)+ ; 1 unit = 1/400 m
# ---------------------------------------------------------------------------
SU = 1.0 / 400.0
FW_C = (290.0, 132.0)
RW_C = (-290.0, 132.0)
R_TYRE = 132.0
R_DISC = 98.0

FAIRING = [(458, 298), (448, 318), (418, 338), (372, 356), (330, 370), (250, 388), (170, 402),
           (90, 410), (10, 404), (-50, 386), (-80, 376), (-200, 370), (-240, 378), (-300, 398),
           (-360, 420), (-410, 431), (-440, 427), (-454, 411), (-450, 390), (-416, 377), (-360, 359),
           (-300, 336), (-250, 312), (-205, 285), (-175, 250), (-150, 205), (-125, 162), (-80, 130),
           (20, 118), (110, 124), (160, 146), (185, 182), (196, 228), (214, 266), (254, 290),
           (310, 300), (370, 298), (420, 290), (450, 290)]
FAIR_SHADE = [(462, 296), (380, 318), (260, 330), (120, 336), (-40, 330), (-180, 332), (-300, 350),
              (-420, 384), (-470, 396), (-470, 290), (-200, 60), (200, 60), (480, 200)]
FAIR_HI = [[(446, 320), (410, 340), (362, 357), (334, 366), (338, 374), (366, 366), (414, 349), (452, 326)],
           [(170, 396), (90, 405), (20, 400), (90, 399), (170, 390)],
           [(-298, 401), (-360, 423), (-410, 433), (-406, 425), (-360, 415), (-296, 394)]]
STRIPE = [(452, 310), (360, 334), (240, 348), (100, 354), (-40, 350), (-180, 352), (-300, 372),
          (-432, 404)]
SCREEN = [(330, 370), (312, 410), (284, 440), (240, 456), (208, 450), (200, 402), (250, 388)]
SCREEN_HI = [(322, 377), (304, 412), (312, 410), (328, 378)]
SEAT = [(-50, 386), (-80, 399), (-212, 395), (-252, 380), (-200, 369), (-80, 375)]
COWL_LOW = [(196, 204), (166, 108), (-70, 98), (-152, 150), (-150, 214), (40, 228)]
SWINGARM = [(-290, 150), (-290, 114), (-118, 136), (-110, 176)]
EXHAUST = [(-160, 178), (-372, 258), (-380, 234), (-170, 152)]
FORK = [(282, 125), (300, 135), (262, 262), (240, 254)]
TAIL_LAMP = [(-456, 410), (-452, 389), (-441, 390), (-444, 411)]
SIDE_LAMP = [(-436, 417), (-362, 413), (-360, 406), (-432, 408)]
HEAD_LAMP = [(461, 304), (449, 314), (408, 312), (422, 302)]
PEG = [(-152, 186), (-104, 186), (-104, 172), (-152, 172)]

# 라이더 포즈: 0 = 턱(질주), 1 = 정지·한발 짚기
TORSO = {
    0: [(-215, 380), (-222, 412), (-160, 445), (-80, 470), (-10, 485), (40, 482), (75, 465),
        (70, 430), (10, 410), (-80, 400), (-150, 384)],
    1: [(-205, 380), (-222, 415), (-190, 470), (-120, 525), (-50, 570), (10, 585), (55, 575),
        (55, 540), (0, 505), (-70, 450), (-130, 392)],
}
HELMET_C = {0: (102, 486, 0.22), 1: (74, 626, -0.08)}       # (u, v, 기울기 rad)
LEG = {0: [(-160, 395), (20, 312), (-112, 196), (-52, 178)],   # 엉덩이, 무릎, 발목, 발끝
       1: [(-140, 395), (-60, 225), (-25, 45), (40, 15)]}
ARM = {0: [(25, 462), (95, 372), (205, 410)],                  # 어깨, 팔꿈치, 손목
       1: [(20, 555), (115, 480), (205, 410)]}
HELMET = [(-62, 0), (-55, 35), (-25, 58), (15, 60), (48, 42), (66, 14), (70, -12), (62, -38),
          (40, -52), (0, -56), (-35, -48), (-58, -28)]
VISOR = {0: [(28, 50), (55, 32), (68, 8), (68, -16), (56, -24), (30, -18), (18, 8), (16, 34)],
         1: [(-12, 48), (40, 46), (66, 22), (70, -8), (55, -24), (0, -24), (-22, 0), (-24, 30)]}
VISOR_HI = {0: [(34, 46), (58, 26), (62, 18), (32, 38)],
            1: [(-8, 42), (50, 38), (52, 30), (-10, 34)]}


def _wheel(img, P3, cxy, spin, speed, seed, sil, boil, rim_dir, wn=40.0, wf=-40.0):
    """디스크 휠: 압출 타이어 + 디스크 커버 + 허브 + 슬롯(고속이면 동심원 스피드 아크)."""
    cx, cy = cxy
    tyre = circ(cx, cy, R_TYRE, 44)[::-1]          # 반시계(v 위 기준)
    extrude(img, P3, tyre, wf, wn, TYRE, hexc("#3a3848"), TYRE, hexc("#26242f"), seed=seed,
            boil=boil, rim=DARK_HI, rim_dir=rim_dir, rim_w=3.5, line_w=4.0, inner_w=2.0, sil=sil)
    wd = wn - 6
    disc = np.c_[circ(cx, cy, R_DISC, 40), np.full(40, wd)]
    sh = np.c_[circ(cx + 28, cy - 36, R_DISC * 1.05, 32), np.full(32, wd)]
    hi = np.c_[circ(cx - 36, cy + 40, 24, 16), np.full(16, wd)]
    cel(img, P3(disc, seed + 2, boil), DISC, shade=DISC_SH, shade_P=P3(sh, seed + 3, 0),
        hi=DARK_HI * 1.15, hi_P=P3(hi, seed + 4, 0), line_w=2.6)
    if speed > 0.35:
        for i, rr in enumerate((34, 52, 70, 86)):
            a0 = spin * 0.37 + i * 1.7
            arc = np.linspace(a0, a0 + 2.6, 18)
            pts = np.c_[cx + rr * np.cos(arc), cy + rr * np.sin(arc), np.full(18, wd)]
            ink(img, P3(pts, 0, 0), 2.4, mix(DISC, (0.8, 0.82, 0.95), 0.55), (0.4, 0.4), alpha=0.85)
    else:
        for k in range(3):
            a = spin + k * 2 * np.pi / 3
            arc = np.linspace(a, a + 0.62, 8)
            slot = np.r_[np.c_[cx + 54 * np.cos(arc), cy + 54 * np.sin(arc)],
                         np.c_[cx + 76 * np.cos(arc[::-1]), cy + 76 * np.sin(arc[::-1])]]
            cel(img, P3(np.c_[slot, np.full(len(slot), wd)], seed + 10 + k, boil * 0.5), DISC_SH,
                line_w=1.6)
    ring = np.c_[circ(cx, cy, 82, 40), np.full(40, wd)]
    ink(img, P3(ring, seed + 6, boil * 0.5), 3.0, RED * 0.9, (0, 0), closed=True)
    hub = np.c_[circ(cx, cy, 25, 20), np.full(20, wd)]
    cel(img, P3(hub, seed + 5, 0), RED, shade=RED_SH,
        shade_P=P3(np.c_[circ(cx + 10, cy - 10, 23, 16), np.full(16, wd)], 0, 0), line_w=2.2)


@lru_cache(maxsize=1)
def _decal_tex():
    tw, th = 260, 76
    rgb = np.zeros((th, tw, 3), np.float32)
    rgb[:] = WHITE
    a = np.zeros((th, tw), np.float32)
    cv2.rectangle(a, (2, 2), (tw - 3, th - 3), 1.0, -1, cv2.LINE_AA)
    m = place_mask(text_mask("疾風", 54, stroke=1), 78, th / 2, tw, th)
    rgb = rgb * (1 - m[..., None]) + m[..., None] * RED
    m2 = place_mask(text_mask("RS-7", 30, font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), 196, th / 2, tw, th)
    rgb = rgb * (1 - m2[..., None]) + m2[..., None] * hexc("#1a1a2a")
    cv2.rectangle(rgb, (148, 10), (151, th - 10), tuple(float(c) for c in INK), -1)
    return rgb, a


def _decal(img, rgb, a, quad):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _bb(quad, 2, w, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return
    th, tw = a.shape
    M = cv2.getPerspectiveTransform(np.float32([(0, 0), (tw, 0), (tw, th), (0, th)]), np.float32(quad - (x0, y0)))
    r = cv2.warpPerspective(np.dstack([rgb * a[..., None], a]), M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    al = r[..., 3:4]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al) + r[..., :3]


def bike_side(img, light, proj, pose=0.0, look=0.0, seed=0, spin=0.0, speed=1.0, sil=None,
              boil=1.1, rim_dir=(-0.45, -1.0), rim=None, glint=0.0):
    """사이드 뷰 바이크+라이더. proj: (N,3) 로컬(u,v,w) → (N,2) 화면. 테일램프 화면 좌표 반환."""
    rim = CYAN * 1.15 if rim is None else rim
    base_seed = seed * 131

    def P3(U, s, b):
        U = np.asarray(U, np.float64)
        if U.shape[1] == 2:
            U = np.c_[U, np.zeros(len(U))]
        S = proj(U)
        return jitter(S, b, base_seed + s) if b > 0 else S

    def P2(pts, w, s, b=boil, n=0):
        pts = smooth(pts, n) if n else np.asarray(pts, np.float64)
        return P3(np.c_[pts, np.full(len(pts), float(w))], s, b)

    k = float(clamp(pose))
    # --- 뒤쪽 레이어 (먼 쪽 팔)
    arm = lerp_pts(ARM[0], ARM[1], k)
    sh_, el, wr = arm
    cel(img, P2(capsule(sh_ + (10, 4), el + (16, 6), 30, 24, 8), -110, 60), JACKET_SH, line_w=3.0, sil=sil)
    cel(img, P2(capsule(el + (16, 6), wr + (14, 4), 23, 19, 8), -110, 61), JACKET_SH, line_w=3.0, sil=sil)
    cel(img, P2(SWINGARM, -20, 1), DARK, rim=DARK_HI, rim_dir=rim_dir, rim_w=3, line_w=2.8, sil=sil)
    _wheel(img, P3, RW_C, spin, speed, 200, sil, boil, rim_dir)
    _wheel(img, P3, FW_C, spin, speed, 300, sil, boil, rim_dir)
    cel(img, P2(FORK, 30, 3), DARK, hi=DARK_HI,
        hi_P=P2([(292, 128), (300, 132), (258, 258), (250, 256)], 30, 0, 0), line_w=2.4, sil=sil)
    cel(img, P2(COWL_LOW, 20, 4, n=4), DARK, shade=DARK_SH,
        shade_P=P2([(220, 160), (-180, 160), (-180, 60), (220, 60)], 20, 5, 0), rim=DARK_HI,
        rim_dir=rim_dir, rim_w=3, line_w=3.0, sil=sil)
    cel(img, P2(EXHAUST, 70, 6, n=3), DARK_HI * 0.8, shade=DARK,
        shade_P=P2([(-150, 165), (-390, 245), (-390, 150), (-150, 120)], 70, 0, 0),
        hi=WHITE * 0.85, hi_P=P2([(-176, 170), (-360, 244), (-362, 239), (-178, 164)], 70, 0, 0),
        line_w=2.6, sil=sil)
    tip = P2([(-374, 258), (-382, 234), (-376, 232), (-368, 256)], 70, 0, 0)
    poly_fill(img, tip, hexc("#ff7a30") * 1.6)
    poly_fill(light, tip, hexc("#ff6a20") * 1.2, mode="add")

    # --- 메인 카울 (압출 솔리드)
    fo = smooth(FAIRING, 4)
    nf = len(fo)
    tail_i0 = int(np.argmin(np.abs(fo[:, 0] + 452) + np.abs(fo[:, 1] - 400) * 0.5))

    def special(i, nu, nv):
        if nu < -0.6 and fo[i, 0] < -430 and 386 < fo[i, 1] < 414:
            return TAIL_HDR
        return None

    extrude(img, P3, fo, -62, 62, RED, RED_HI * 0.85, RED_SH * 0.7, RED_SH, seed=8, boil=boil,
            shade=RED_SH, shade_uv=FAIR_SHADE, hi=RED_HI, hi_uv=FAIR_HI, rim=rim, rim_dir=rim_dir,
            rim_w=4.5, line_w=4.8, inner_w=2.4, sil=sil, special=special)
    # 테일 단면 램프 → 라이트 버퍼
    ink(img, P2(STRIPE, 62, 11, n=4), 11, WHITE, (0.05, 0.25))
    ink(img, P2([(p[0], p[1] - 14) for p in STRIPE[:6]], 62, 12, n=4), 3.0, INK, (0.1, 0.4))
    ink(img, P2([(194, 210), (226, 300), (236, 378)], 62, 13, n=4), 2.2, INK, (0.2, 0.2))
    ink(img, P2([(-205, 290), (-226, 336), (-236, 372)], 62, 14, n=4), 2.0, INK, (0.2, 0.2))
    ink(img, P2([(-80, 132), (-56, 200), (-10, 250)], 62, 15, n=4), 1.8, INK, (0.3, 0.3))
    cel(img, P2([(170, 266), (90, 272), (60, 256), (150, 250)], 62, 16), DARK_SH, line_w=2.0)
    cel(img, P2([(160, 240), (100, 244), (84, 234), (150, 230)], 62, 17), DARK_SH, line_w=1.6)
    # 일본어 데칼 (오리지널)
    dq = P2([(-150, 320), (-20, 320), (-20, 282), (-150, 282)], 63, 0, 0)
    _decal(img, *_decal_tex(), dq)
    tl = P2(TAIL_LAMP, 62, 18, 0.4)
    cel(img, tl, TAIL_HDR, line_w=2.0, light=light, light_col=TAIL_HDR * 0.9)
    sl = P2(SIDE_LAMP, 62, 19, 0.4)
    poly_fill(img, sl, TAIL_HDR * 0.8)
    poly_fill(light, sl, TAIL_HDR * 0.5, mode="add")
    hl = P2(HEAD_LAMP, 62, 20, 0.4)
    cel(img, hl, HEAD_HDR, line_w=1.8, light=light, light_col=HEAD_HDR * 0.8)
    cel(img, P2(SCREEN, 40, 21, n=3), hexc("#1a2238"), hi=hexc("#9ff6ff") * 1.3,
        hi_P=P2(SCREEN_HI, 40, 22, 0), line_w=2.8, alpha=0.93)
    cel(img, P2(SEAT, 60, 23, n=3), hexc("#151320"), rim=DARK_HI, rim_dir=rim_dir, rim_w=3,
        line_w=2.6, sil=sil)
    cel(img, P2(PEG, 110, 24), DARK_HI, line_w=2.0, sil=sil)

    # --- 라이더
    torso = lerp_pts(TORSO[0], TORSO[1], k)
    hx, hy, ht = lerp_pts(HELMET_C[0], HELMET_C[1], k)
    leg = lerp_pts(LEG[0], LEG[1], k)
    tsm = smooth(torso, 6)
    sh_line = lerp_pts([(-260, 392), (-120, 402), (0, 425), (90, 445), (90, 300), (-260, 300)],
                       [(-260, 392), (-130, 430), (-50, 500), (60, 560), (90, 300), (-260, 300)], k)
    cel(img, P2(tsm, 0, 30), JACKET, shade=JACKET_SH, shade_P=P2(sh_line, 0, 31, 0.6),
        hi=JACKET_HI * 0.8, hi_P=P2(lerp_pts([(-150, 446), (-60, 474), (20, 488), (-60, 466), (-150, 436)],
                                             [(-196, 474), (-136, 532), (-70, 570), (-124, 520), (-186, 462)], k), 0, 0, 0),
        rim=rim, rim_dir=rim_dir, rim_w=4.5, line_w=4.2, sil=sil)
    # 재킷 디테일: 줄무늬, 밑단, 허리 주름, 지퍼선
    ink(img, P2(lerp_pts([(-160, 440), (-80, 466), (-10, 481)], [(-190, 466), (-120, 522), (-50, 566)], k),
                5, 32, n=3), 6.0, CYAN * 0.9, (0.2, 0.2))
    ink(img, P2(lerp_pts([(-212, 414), (-182, 398), (-150, 386)], [(-222, 430), (-180, 410), (-138, 396)], k), 4, 33, n=3),
        3.0, INK, (0.1, 0.1))
    ink(img, P2(lerp_pts([(-120, 446), (-104, 430)], [(-150, 500), (-128, 488)], k), 4, 34), 2.0, INK, (0.3, 0.3))
    ink(img, P2(lerp_pts([(-60, 458), (-44, 440)], [(-96, 540), (-76, 528)], k), 4, 35), 2.0, INK, (0.3, 0.3))
    # 헬멧
    ca, sa = math.cos(ht), math.sin(ht)

    def hp(pts):
        pts = np.asarray(pts, np.float64)
        return np.c_[hx + pts[:, 0] * ca + pts[:, 1] * sa, hy - pts[:, 0] * sa + pts[:, 1] * ca]

    hsm = smooth(HELMET, 4)
    cel(img, P2(hp(hsm), 20, 36), HELM, shade=HELM_SH,
        shade_P=P2(hp(np.array([(-80, -8), (10, -18), (80, -4), (80, -80), (-80, -80)])), 20, 37, 0.5),
        hi=hexc("#8a8fb8"), hi_P=P2(hp([(-40, 48), (-8, 58), (20, 56), (-6, 50), (-36, 42)]), 20, 0, 0),
        rim=rim, rim_dir=rim_dir, rim_w=5, line_w=4.4, sil=sil)
    ink(img, P2(hp([(-60, 14), (-30, 50), (12, 60)]), 22, 38, n=4), 8, RED * 1.1, (0.3, 0.3))
    ink(img, P2(hp([(30, -50), (44, -36)]), 22, 39), 2.0, INK, (0.2, 0.2))
    ink(img, P2(hp([(20, -52), (34, -38)]), 22, 40), 2.0, INK, (0.2, 0.2))
    vis = smooth(lerp_pts(VISOR[0], VISOR[1], look), 4)
    vhi = lerp_pts(VISOR_HI[0], VISOR_HI[1], look)
    vtop = np.array([(-40, 60), (90, 60), (90, 14), (-40, 22)])
    cel(img, P2(hp(vis), 24, 41), hexc("#101a3e"), shade=hexc("#3a5ab8"),
        shade_P=P2(hp(vtop), 24, 0, 0), hi=hexc("#a8f8ff") * 1.7, hi_P=P2(hp(vhi), 24, 0, 0),
        line_w=2.8)
    if glint > 0:
        gc = P2(hp([lerp_pts([(46, 30)], [(24, 36)], look)[0]]), 24, 0, 0)[0]
        for a, L in ((0, 70), (np.pi / 2, 70), (np.pi / 4, 26), (3 * np.pi / 4, 26)):
            d = np.array([math.cos(a), math.sin(a)]) * L * glint
            pts = np.array([gc - d, gc, gc + d])
            ink(img, pts, 6 * glint, (3.0, 3.2, 3.6), (0.9, 0.9))
            ink(light, pts, 9 * glint, (2.0, 2.4, 3.0), (0.9, 0.9))
    # 다리 (가까운 쪽): 허벅지(윗면 볼록), 정강이(종아리 볼록), 롱부츠
    hip, knee, ank, toe = leg
    thigh = limb(hip, knee, 46, 34, bulge=9, side=1 if knee[0] > hip[0] else -1)
    cel(img, P2(thigh, 120, 42), PANTS, shade=JACKET_SH,
        shade_P=P2(limb(hip + (6, -22), knee + (4, -20), 40, 28, 0), 120, 43, 0), rim=rim,
        rim_dir=rim_dir, rim_w=3.5, line_w=3.8, sil=sil)
    dsh = ank - knee
    calf_side = -1 if dsh[1] < 0 and dsh[0] < 0 else 1
    shin = limb(knee, ank, 33, 22, bulge=10, side=calf_side)
    cel(img, P2(shin, 125, 44), PANTS, shade=JACKET_SH,
        shade_P=P2(limb(knee + (12, -6), ank + (10, -4), 24, 16, 0), 125, 45, 0), line_w=3.4, sil=sil)
    # 무릎 주름
    kd = (knee - hip) / (np.hypot(*(knee - hip)) + 1e-9)
    kn = np.array([-kd[1], kd[0]])
    for o in (0, 12):
        a0 = knee - kd * (30 + o) - kn * 24
        ink(img, P2([a0, a0 + kd * 12 + kn * 8, a0 + kd * 18 + kn * 20], 126, 46 + o), 2.0, INK, (0.3, 0.3))
    # 부츠: 정강이 아래 40% + 발
    d = toe - ank
    nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
    bt = knee + (ank - knee) * 0.5
    sn = np.array([-(ank - knee)[1], (ank - knee)[0]]) / (np.hypot(*(ank - knee)) + 1e-9)
    boot = [bt + sn * 27, bt - sn * 27, ank - nrm * 26 - d * 0.3, toe - nrm * 18 + d * 0.15,
            toe + nrm * 16 + d * 0.22, ank + nrm * 34 + d * 0.2]
    cel(img, P2(boot, 130, 48, n=2), BOOT,
        hi=RED, hi_P=P2([bt + sn * 30 + (ank - knee) * 0.12, bt - sn * 30 + (ank - knee) * 0.12,
                         bt - sn * 30 + (ank - knee) * 0.22, bt + sn * 30 + (ank - knee) * 0.22], 130, 0, 0),
        rim=DARK_HI, rim_dir=rim_dir, rim_w=3, line_w=3.4, sil=sil)
    # 팔 (가까운 쪽): 어깨 아머, 상완, 전완, 장갑
    sh_, el, wr = arm
    cel(img, P2(limb(sh_, el, 33, 25, bulge=7, side=-1), 135, 50), JACKET, shade=JACKET_SH,
        shade_P=P2(limb(sh_ + (8, -16), el + (8, -14), 26, 18, 0), 135, 51, 0), rim=rim,
        rim_dir=rim_dir, rim_w=4, line_w=3.8, sil=sil)
    cel(img, P2(limb(el, wr, 25, 19, bulge=6, side=1), 135, 52), JACKET, shade=JACKET_SH,
        shade_P=P2(limb(el + (4, -12), wr + (4, -10), 19, 13, 0), 135, 53, 0), line_w=3.4, sil=sil)
    ed = (wr - el) / (np.hypot(*(wr - el)) + 1e-9)
    en = np.array([-ed[1], ed[0]])
    ink(img, P2([el + ed * 10 - en * 20, el + ed * 22 - en * 12], 136, 54), 2.0, INK, (0.3, 0.3))
    ink(img, P2([el + ed * 20 - en * 22, el + ed * 32 - en * 14], 136, 55), 2.0, INK, (0.3, 0.3))
    ad = (el - sh_) / (np.hypot(*(el - sh_)) + 1e-9)
    an = np.array([-ad[1], ad[0]])
    pc = sh_ + ad * 22
    pad = [pc - ad * 20 + an * 22, pc + ad * 24 + an * 26, pc + ad * 34, pc + ad * 22 - an * 26,
           pc - ad * 20 - an * 20, pc - ad * 28]
    cel(img, P2(pad, 137, 56, n=3), RED, shade=RED_SH, shade_P=P2([pc, pc + ad * 50 - an * 50, pc - ad * 50 - an * 50], 137, 0, 0),
        hi=RED_HI, hi_P=P2([pc - ad * 14 + an * 16, pc + ad * 14 + an * 19, pc + ad * 7 + an * 11, pc - ad * 12 + an * 10], 137, 0, 0),
        rim=rim, rim_dir=rim_dir, rim_w=3.5, line_w=3.4, sil=sil)
    cuff = [wr - ed * 26 + en * 24, wr - ed * 26 - en * 24, wr - ed * 6 - en * 22, wr - ed * 6 + en * 22]
    cel(img, P2(cuff, 139, 57), DARK, line_w=2.6, sil=sil)
    glove = [wr - ed * 8 + en * 21, wr + ed * 22 + en * 22, wr + ed * 38 + en * 6, wr + ed * 34 - en * 18,
             wr - ed * 8 - en * 20]
    cel(img, P2(glove, 140, 58, n=2), BOOT, hi=DARK_HI, hi_P=P2([wr + ed * 4 + en * 18, wr + ed * 26 + en * 18, wr + ed * 24 + en * 10, wr + ed * 4 + en * 10], 140, 0, 0),
        rim=DARK_HI, rim_dir=rim_dir, rim_w=3, line_w=3.0, sil=sil)
    return P2([(-452, 400)], 62, 0, 0)[0]


# ---------------------------------------------------------------------------
# 바이크 (후방 뷰) — 컷 A. 로컬 u(오른쪽)+, v(위)+, 원점 = 타이어 접지점. 1 unit = 1/375 m
# ---------------------------------------------------------------------------
RU = 1.0 / 375.0


def _mir(pts):
    return [(-x, y) for x, y in pts][::-1]


RV_FLANK = [(92, 180), (128, 205), (150, 256), (144, 302), (114, 324), (86, 302), (78, 230)]
RV_TYRE = [(-30, 178), (-34, 62), (-29, 20), (-17, 4), (0, 0), (17, 4), (29, 20), (34, 62), (30, 178)]
RV_HUG = [(-46, 150), (46, 150), (54, 197), (-54, 197)]
RV_TAIL = [(-58, 188), (-92, 212), (-104, 252), (-92, 284), (-56, 300), (56, 300), (92, 284),
          (104, 252), (92, 212), (58, 188)]
RV_LAMP = [(-91, 236), (91, 236), (87, 257), (-87, 257)]
RV_TORSO = [(-60, 318), (-90, 352), (-104, 392), (-96, 420), (-70, 438), (-34, 449), (0, 452),
            (34, 449), (70, 438), (96, 420), (104, 392), (90, 352), (60, 318)]
RV_SCREEN = [(-68, 452), (-50, 494), (50, 494), (68, 452)]


def bike_rear(img, light, X, sil=None, seed=0, boil=1.1, pursuer=0.0, spin=0.0):
    """후방 뷰 바이크. X: (N,2) 로컬 → (N,2) 화면 변환. 테일램프 중심 화면 좌표 반환."""
    bs = seed * 97

    def T(pts, s, b=boil, n=0):
        pts = smooth(pts, n) if n else np.asarray(pts, np.float64)
        S = X(pts)
        return jitter(S, b, bs + s) if b > 0 else S

    warm = mix(WHITE, hexc("#ffd8a8"), 0.5)
    rimc = CYAN * 1.2
    rd = (1.0, -0.35)              # 림: 오른쪽 위 (네온)
    hiw = 0.25 + 0.75 * pursuer    # 추격자 헤드라이트 하이라이트 세기
    # 앞 카울 플랭크 (양쪽, 다리 뒤)
    for sgn, s0 in ((1, 0), (-1, 10)):
        fl = RV_FLANK if sgn > 0 else _mir(RV_FLANK)
        cel(img, T(fl, 1 + s0, n=3), RED, shade=RED_SH,
            shade_P=T([(sgn * 70, 170), (sgn * 160, 170), (sgn * 160, 262), (sgn * 70, 236)], 2 + s0, 0),
            rim=rimc if sgn > 0 else None, rim_dir=rd, rim_w=4, line_w=3.6, sil=sil)
        # 미러
        mx = sgn * 150
        cel(img, T([(mx - 18 * sgn, 352), (mx + 16 * sgn, 360), (mx + 18 * sgn, 344), (mx - 14 * sgn, 338)], 3 + s0),
            DARK, line_w=2.4, sil=sil)
    # 배기구 (양쪽)
    for sgn in (1, -1):
        cx = sgn * 68
        can = [(cx - 17, 92), (cx + 17, 92), (cx + 19, 140), (cx - 19, 140)]
        cel(img, T(can, 20 + sgn, n=0), DARK_HI * 0.9, shade=DARK,
            shade_P=T([(cx, 80), (cx + 30, 80), (cx + 30, 150), (cx, 150)], 0, 0), line_w=2.6, sil=sil)
        tip = T(ellip(cx, 112, 10, 8, 0, 16), 22 + sgn, 0.4)
        cel(img, tip, hexc("#2a0c10"), line_w=2.0)
        tip2 = T(ellip(cx, 110, 6, 4, 0, 12), 0, 0)
        poly_fill(img, tip2, hexc("#ff4a18") * 1.1)
        poly_fill(light, tip2, hexc("#ff4a18") * 0.4, mode="add")
    # 타이어
    cel(img, T(RV_TYRE, 30, n=3), TYRE, hi=DARK_HI * (0.8 + 0.6 * hiw),
        hi_P=T([(-27, 172), (-30, 62), (-24, 24), (-15, 24), (-19, 62), (-17, 172)], 31, 0),
        rim=DARK_HI * 1.3, rim_dir=rd, rim_w=3.5, line_w=3.6, sil=sil)
    # 트레드 홈 (회전감)
    for k in range(4):
        yy = 30 + ((k * 40 + spin * 60) % 150)
        ink(img, T([(-22, yy), (0, yy - 6), (22, yy)], 0, 0), 2.0, DARK_HI, (0.3, 0.3), alpha=0.7)
    cel(img, T(RV_HUG, 32), DARK, line_w=2.6, sil=sil)
    plate = T([(-26, 160), (26, 160), (24, 188), (-24, 188)], 33, 0.5)
    cel(img, plate, hexc("#b8c0c8"), line_w=2.0)
    for yy in (168, 179):
        ink(img, T([(-16, yy), (16, yy)], 0, 0), 2.2, INK, (0.1, 0.1))
    # 다리 (양쪽)
    for sgn, s0 in ((1, 40), (-1, 50)):
        knee = (sgn * 128, 282)
        ank = (sgn * 94, 196)
        hip = (sgn * 66, 330)
        cel(img, T(capsule(hip, knee, 32, 26, 8), s0), PANTS, shade=JACKET_SH,
            shade_P=T(capsule((hip[0] + sgn * 12, hip[1]), (knee[0] + sgn * 12, knee[1]), 26, 20, 6), s0 + 1, 0),
            hi=mix(PANTS, warm, 0.35 * hiw) if sgn < 0 else None,
            hi_P=T(capsule((hip[0] - sgn * 14, hip[1]), (knee[0] - sgn * 14, knee[1]), 20, 14, 6), s0 + 2, 0) if sgn < 0 else None,
            rim=rimc if sgn > 0 else None, rim_dir=rd, rim_w=3.5, line_w=3.4, sil=sil)
        cel(img, T(capsule(knee, ank, 25, 20, 8), s0 + 3), PANTS, shade=JACKET_SH,
            shade_P=T(capsule((knee[0] + sgn * 10, knee[1]), (ank[0] + sgn * 10, ank[1]), 18, 14, 6), s0 + 4, 0),
            line_w=3.2, sil=sil)
        boot = [(sgn * 76, 148), (sgn * 112, 150), (sgn * 116, 200), (sgn * 78, 206)]
        cel(img, T(boot, s0 + 5, n=2), BOOT, hi=RED, hi_P=T([(sgn * 80, 176), (sgn * 114, 176), (sgn * 114, 186), (sgn * 80, 186)], 0, 0),
            line_w=3.0, sil=sil)
        cel(img, T([(sgn * 70, 146), (sgn * 120, 146), (sgn * 120, 156), (sgn * 70, 156)], s0 + 6), DARK_HI, line_w=1.8)
    # 테일 카울
    cel(img, T(RV_TAIL, 60, n=3), RED,
        shade=RED_SH, shade_P=[T([(-120, 150), (120, 150), (120, 222), (40, 214), (-120, 222)], 61, 0.6),
                               T([(50, 222), (120, 222), (120, 320), (70, 320)], 62, 0.6)],
        hi=mix(RED_HI, warm, 0.3 * hiw), hi_P=T([(-86, 268), (-60, 292), (10, 296), (-50, 286), (-80, 262)], 63, 0.4),
        rim=rimc, rim_dir=rd, rim_w=4, line_w=4.2, sil=sil)
    lamp = T(RV_LAMP, 64, 0.5)
    cel(img, lamp, TAIL_HDR, line_w=2.4, light=light, light_col=TAIL_HDR * 0.9)
    poly_fill(img, T([(-70, 242), (70, 242), (68, 250), (-68, 250)], 0, 0), TAIL_CORE)
    for xx in (-30, 30):
        ink(img, T([(xx, 238), (xx, 255)], 0, 0), 2.4, INK, (0, 0))
    # 엉덩이/골반
    cel(img, T(ellip(0, 322, 74, 34, 0, 28), 70), PANTS, shade=JACKET_SH,
        shade_P=T(ellip(30, 310, 70, 30, 0, 20), 71, 0), line_w=3.2, sil=sil)
    # 윈드스크린 끝 (헬멧 뒤로 살짝)
    cel(img, T(RV_SCREEN, 72, n=2), hexc("#1d2a44"), hi=hexc("#9ff6ff") * 1.3,
        hi_P=T([(30, 456), (44, 490), (52, 490), (40, 456)], 0, 0), line_w=2.6, alpha=0.92)
    # 팔 (양쪽)
    for sgn, s0 in ((1, 80), (-1, 90)):
        sh = (sgn * 94, 418)
        el = (sgn * 136, 370)
        wr = (sgn * 116, 338)
        cel(img, T(capsule(sh, el, 25, 21, 8), s0), JACKET, shade=JACKET_SH,
            shade_P=T(capsule((sh[0] + sgn * 9, sh[1] - 7), (el[0] + sgn * 9, el[1] - 7), 20, 15, 6), s0 + 1, 0),
            rim=rimc if sgn > 0 else None, rim_dir=rd, rim_w=4, line_w=3.4, sil=sil)
        cel(img, T(capsule(el, wr, 20, 17, 8), s0 + 2), JACKET, shade=JACKET_SH,
            shade_P=T(capsule((el[0] + sgn * 7, el[1]), (wr[0] + sgn * 7, wr[1]), 15, 12, 6), s0 + 3, 0),
            line_w=3.2, sil=sil)
    # 등
    cel(img, T(RV_TORSO, 100, n=4), JACKET,
        shade=JACKET_SH, shade_P=T([(30, 300), (140, 300), (140, 460), (60, 460), (40, 400)], 101, 0.6),
        hi=mix(JACKET, JACKET_HI, 0.4 + 0.5 * hiw), hi_P=T([(-118, 392), (-100, 360), (-70, 330), (-84, 372), (-96, 420)], 102, 0.5),
        rim=rimc, rim_dir=rd, rim_w=4.5, line_w=4.2, sil=sil)
    ink(img, T([(-92, 416), (-46, 430), (0, 434), (46, 430), (92, 416)], 103, n=3), 7, CYAN * 0.95, (0.15, 0.15))
    for sgn, s0 in ((1, 120), (-1, 125)):
        pad = [(sgn * 70, 432), (sgn * 104, 440), (sgn * 124, 420), (sgn * 120, 392), (sgn * 96, 384), (sgn * 76, 404)]
        cel(img, T(pad, s0, n=2), RED, shade=RED_SH,
            shade_P=T([(sgn * 100, 380), (sgn * 140, 380), (sgn * 140, 446), (sgn * 108, 420)], s0 + 1, 0),
            hi=mix(RED_HI, warm, 0.3 * hiw) if sgn < 0 else RED_HI * 0.9,
            hi_P=T([(sgn * 78, 428), (sgn * 102, 436), (sgn * 108, 428), (sgn * 84, 420)], s0 + 2, 0),
            rim=rimc if sgn > 0 else None, rim_dir=rd, rim_w=3.5, line_w=3.2, sil=sil)
    ink(img, T([(0, 330), (2, 380), (0, 425)], 104, n=3), 2.2, INK, (0.3, 0.3))
    ink(img, T([(-60, 345), (-40, 352)], 105), 2.0, INK, (0.3, 0.3))
    ink(img, T([(56, 350), (36, 356)], 106), 2.0, INK, (0.3, 0.3))
    # 헬멧
    cel(img, T(ellip(0, 458, 58, 52, 0, 36), 110), HELM, shade=HELM_SH,
        shade_P=T(ellip(26, 440, 56, 50, 0, 28), 111, 0.5),
        hi=hexc("#9aa0c8"), hi_P=T([(-44, 478), (-30, 496), (-8, 504), (-28, 492), (-40, 474)], 112, 0.4),
        rim=rimc, rim_dir=rd, rim_w=5, line_w=4.2, sil=sil)
    for xo in (-9, 9):
        ink(img, T([(xo * 0.6, 508), (xo, 470), (xo * 0.8, 418)], 113 + xo, n=3), 5, RED * 1.1, (0.08, 0.08))
    return T([(0, 247)], 0, 0)[0]


# ---------------------------------------------------------------------------
# 도시 배경 페인터 (캐시 레이어용)
# ---------------------------------------------------------------------------
WIN_COLS = np.array([hexc("#ffb050"), hexc("#ffd890"), hexc("#ff9a48"), hexc("#ffcf80"),
                     hexc("#fff0c8"), hexc("#ffc070"), hexc("#8fe8ff"), hexc("#ffb8a0")], np.float32)
NEON_WORDS = ["夜光", "電影", "珈琲", "酒場", "ネオン", "龍門", "カラオケ", "未来", "東邦", "麻雀",
              "薬", "愛", "旅館", "銀河", "質屋", "映画"]
NEON_COLS = [hexc("#ff3fb4"), hexc("#3ff0ff"), hexc("#ff3a3a"), hexc("#ffb03a"), hexc("#7cff9a"),
             hexc("#b46aff")]


def _building(rgb, alpha, x0, x1, ytop, ybase, rng, body, haze, haze_col, lit=0.45, cw=9.0,
              ch=12.0, rim_col=None, rim_side=1, glow_bottom=0.5, hdr=1.0):
    h_, w_ = alpha.shape
    X0, X1 = int(max(0, x0)), int(min(w_, x1))
    Y0, Y1 = int(max(0, ytop)), int(min(h_, ybase))
    if X1 - X0 < 2 or Y1 - Y0 < 2:
        return
    hh, ww = Y1 - Y0, X1 - X0
    g = np.clip((np.arange(hh, dtype=np.float32) + (Y0 - ytop)) / max(1.0, ybase - ytop), 0, 1)[:, None, None]
    col = body[None, None, :] * (0.75 + glow_bottom * g ** 2.2)
    col = col * (1 - haze) + haze_col[None, None, :] * haze
    reg = np.broadcast_to(col, (hh, ww, 3)).copy()
    # 창문 그리드
    xs = np.arange(ww, dtype=np.float32) + (X0 - x0)
    ys = np.arange(hh, dtype=np.float32) + (Y0 - ytop)
    cx = np.floor((xs - 3) / cw).astype(np.int32)
    cy = np.floor((ys - 6) / ch).astype(np.int32)
    fx = ((xs - 3) / cw) % 1.0
    fy = ((ys - 6) / ch) % 1.0
    nx, ny = int(cx.max()) + 2, int(cy.max()) + 2
    R = rng.random((ny, nx))
    C = rng.integers(0, len(WIN_COLS), (ny, nx))
    B = (rng.uniform(0.25, 1.0, (ny, nx)) ** 1.6 * 1.3) * hdr
    fd = rng.random(ny) < 0.18
    cd = rng.random(nx) < 0.12
    # 큰 블록 단위로 꺼진 영역
    blk = rng.random((ny // 4 + 2, nx // 3 + 2)) < 0.25
    on = (R < lit) & ~fd[:, None] & ~cd[None, :] & ~blk[np.arange(ny)[:, None] // 4, np.arange(nx)[None, :] // 3]
    cxx, cyy = np.clip(cx, 0, nx - 1), np.clip(cy, 0, ny - 1)
    mx = (fx > 0.18) & (fx < 0.78) & (xs > 3) & (xs < (x1 - x0) - 3)
    my = (fy > 0.22) & (fy < 0.72) & (ys > 6)
    m = on[cyy][:, cxx] & my[:, None] & mx[None, :]
    wc = WIN_COLS[C[cyy][:, cxx]] * B[cyy][:, cxx][..., None]
    wc = wc * (1 - haze * 0.7) + haze_col * haze * 0.7
    reg[m] = wc[m]
    if rim_col is not None:
        rw = max(2, int(ww * 0.06))
        if rim_side > 0:
            reg[:, -rw:] = reg[:, -rw:] * 0.5 + rim_col * 0.5
        else:
            reg[:, :rw] = reg[:, :rw] * 0.5 + rim_col * 0.5
    rgb[Y0:Y1, X0:X1] = reg
    alpha[Y0:Y1, X0:X1] = 1.0


def _neon_sign(rgb, alpha, light, x, y, size, rng, vertical=True, word=None, col=None):
    word = word or NEON_WORDS[int(rng.integers(len(NEON_WORDS)))]
    col = col if col is not None else NEON_COLS[int(rng.integers(len(NEON_COLS)))]
    m = text_mask(word, size, stroke=1, vertical=vertical)
    mh, mw = m.shape
    pad = size * 0.25
    bx0, by0 = x - mw / 2 - pad, y - mh / 2 - pad
    bx1, by1 = x + mw / 2 + pad, y + mh / 2 + pad
    box = np.array([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)])
    poly_fill(rgb, box, hexc("#150a1c"))
    mask_poly(alpha, box, 1.0)
    ink(rgb, np.r_[box, box[:1]], max(2.0, size * 0.06), col * 1.6, (0, 0))
    ink(light, np.r_[box, box[:1]], max(2.0, size * 0.06), col * 0.6, (0, 0))
    h_, w_ = alpha.shape
    X0, Y0 = int(max(0, math.floor(bx0))), int(max(0, math.floor(by0)))
    X1, Y1 = int(min(w_, math.ceil(bx1) + 1)), int(min(h_, math.ceil(by1) + 1))
    if X1 - X0 < 2 or Y1 - Y0 < 2:
        return
    pm = place_mask(m, x - X0, y - Y0, X1 - X0, Y1 - Y0)
    rgb[Y0:Y1, X0:X1] += pm[..., None] * col * 2.6
    light[Y0:Y1, X0:X1] += pm[..., None] * col * 1.4


@lru_cache(maxsize=1)
def _layers_A():
    """컷 A: (원경 불투명 RGB, 중경 premult RGB, 중경 alpha). VP 기준 확대해서 쓴다."""
    rng = np.random.default_rng(2207)
    vpx, vpy = CXA, CYA
    sky = vgradient([(0, hexc("#04051c")), (0.35, hexc("#100d38")), (0.60, hexc("#2c1650")),
                     (0.80, hexc("#6a2464")), (0.93, hexc("#a8386a")), (1.0, hexc("#e8607a"))],
                    y0=0, y1=vpy + 10)
    # 지평선 아래(도로에 가려지지만 빈틈용)
    sky[int(vpy) + 10:] = hexc("#e8607a") * 0.8
    # 도시 빛 돔 (에어브러시)
    d = np.sqrt(((np.arange(W, dtype=np.float32)[None, :] - vpx) / 1.6) ** 2 +
                (np.arange(H, dtype=np.float32)[:, None] - vpy) ** 2)
    sky += (np.exp(-(d / 200.0) ** 2) * 0.5 + np.exp(-(d / 520.0) ** 2) * 0.18)[..., None] * hexc("#ff6a78")
    painted_texture(sky, 0.07, seed=31, scale=4.0, stretch=8.0)
    # 별 몇 개
    for _ in range(70):
        x, y = rng.uniform(0, W), rng.uniform(0, vpy * 0.45)
        r = rng.uniform(0.8, 1.8)
        poly_fill(sky, circ(x, y, r, 8), np.array([0.9, 0.9, 1.0]) * rng.uniform(0.4, 1.0))
    light = np.zeros_like(sky)
    a_dummy = np.zeros((H, W), np.float32)
    haze_col = hexc("#d0587a")
    # 원경 스카이라인 (지평선 위 작은 빌딩들, 두 겹)
    for row, (hmin, hmax, hz, body) in enumerate(((30, 120, 0.62, hexc("#3a2050")),
                                                  (50, 190, 0.42, hexc("#24183c")))):
        x = -40.0
        while x < W + 40:
            bw = rng.uniform(14, 46) * (1 + abs(x - vpx) / 900)
            dist = abs(x + bw / 2 - vpx) / 960
            bh = rng.uniform(hmin, hmax) * (0.6 + 0.9 * dist)
            _building(sky, a_dummy, x, x + bw, vpy - bh, vpy + 6, rng, body, hz, haze_col,
                      lit=0.35, cw=4.0 + dist * 3, ch=5.0 + dist * 3, glow_bottom=0.9, hdr=1.1)
            if rng.random() < 0.25:
                ax = x + bw * rng.uniform(0.3, 0.7)
                ink(sky, np.array([(ax, vpy - bh), (ax, vpy - bh - rng.uniform(8, 30))]), 1.6, body * 1.3, (0, 0))
            x += bw + rng.uniform(-4, 6)
    # 서치라이트 원천 아래 붉은 항공등
    for _ in range(18):
        x = rng.uniform(0, W)
        y = vpy - rng.uniform(40, 200) * (0.6 + abs(x - vpx) / 900)
        poly_fill(sky, circ(x, y, 2.2, 8), np.array([3.0, 0.3, 0.2]))
        poly_fill(light, circ(x, y, 3.0, 8), np.array([1.5, 0.15, 0.1]), mode="add")
    sky += blur(light, 3) * 0.8 + blur(light, 12) * 0.5

    # 중경: 양옆 고층 타워 (premult)
    mid = np.zeros_like(sky)
    ma = np.zeros((H, W), np.float32)
    ml = np.zeros_like(sky)
    towers = []
    for side in (-1, 1):
        x = vpx + side * 230
        while 0 - 300 < x < W + 300:
            bw = rng.uniform(70, 150) * (1 + abs(x - vpx) / 600)
            dist = abs(x - vpx) / 960
            bh = rng.uniform(150, 260) + dist * rng.uniform(250, 560)
            x0 = x if side > 0 else x - bw
            towers.append((dist, x0, bw, bh, side))
            x += side * (bw + rng.uniform(10, 60) * (1 + dist))
    towers.sort(key=lambda t: -t[0])          # 먼 것 (VP 가까운) 먼저? 가까운 게 크다 → 큰 것 뒤에
    towers.sort(key=lambda t: t[0])
    for dist, x0, bw, bh, side in towers:
        body = mix(hexc("#22193e"), hexc("#0e0e26"), min(1, dist * 1.3))
        hz = 0.30 * (1 - dist) ** 2 + 0.03
        top = vpy - bh
        _building(mid, ma, x0, x0 + bw, top, vpy + 8, rng, body, hz, haze_col, lit=0.5,
                  cw=5 + dist * 6, ch=7 + dist * 8, rim_col=hexc("#ff8a6a") * 0.6,
                  rim_side=-side, glow_bottom=0.7, hdr=1.15)
        # 왕관/안테나
        if rng.random() < 0.5:
            cx_ = x0 + bw / 2
            cw_ = bw * rng.uniform(0.3, 0.6)
            ch_ = rng.uniform(20, 60) * (1 + dist)
            crown = np.array([(cx_ - cw_ / 2, top + 1), (cx_ - cw_ / 2, top - ch_), (cx_ + cw_ / 2, top - ch_), (cx_ + cw_ / 2, top + 1)])
            poly_fill(mid, crown, body * 0.9)
            mask_poly(ma, crown, 1.0)
            ax = cx_ + rng.uniform(-0.2, 0.2) * cw_
            ah = rng.uniform(30, 90) * (1 + dist)
            ink(mid, np.array([(ax, top - ch_), (ax, top - ch_ - ah)]), 2.0 + dist * 2, body * 1.4, (0, 0.2))
            mask_poly(ma, np.array([(ax - 1.5, top - ch_), (ax + 1.5, top - ch_), (ax + 1.5, top - ch_ - ah), (ax - 1.5, top - ch_ - ah)]), 1.0)
            b = circ(ax, top - ch_ - ah, 3 + dist * 2, 10)
            poly_fill(mid, b, np.array([3.2, 0.35, 0.25]))
            poly_fill(ml, b, np.array([2.0, 0.2, 0.1]), mode="add")
        # 네온 간판
        if dist > 0.25 and rng.random() < 0.75:
            sz = int(18 + dist * 34)
            nxp = x0 + (bw * 0.18 if side < 0 else bw * 0.82)
            nyp = top + bh * rng.uniform(0.2, 0.5)
            _neon_sign(mid, ma, ml, nxp, nyp, sz, rng)
    mid += blur(ml, 4) * 0.9 + blur(ml, 16) * 0.6 + blur(ml, 40) * 0.4
    painted_texture(mid, 0.05, seed=44, scale=3.0, stretch=1.0, angle=1.57)
    mid *= ma[..., None] + (1 - ma[..., None]) * 0.0
    mid += (blur(ml, 16) * 0.5 + blur(ml, 40) * 0.4) * (1 - ma[..., None])
    return sky.astype(np.float32), mid.astype(np.float32), ma.astype(np.float32)


# ---------------------------------------------------------------------------
# 컷 A — 후방 추격
# ---------------------------------------------------------------------------
FA, CXA, CYA, HCA = 900.0, 960.0, 450.0, 0.95
VA = 62.0                    # m/s
ZBK = 2.5                    # 카메라→바이크 거리
SHUT = 1.0 / 48.0
LAMP_SP = 24.0               # 가로등 간격
GANTRY_Z0 = 46.0
TRAIL_V = 9.0                # 잔상 리본이 카메라 쪽으로 흘러오는 속도(연출값)


def _xb(t):
    return 0.95 * math.sin(2 * math.pi * 0.62 * t + 0.25) + 0.32 * math.sin(2 * math.pi * 1.45 * t + 1.1)


def _xc(t):
    return 0.6 * _xb(t - 0.14)


def _lean_A(t):
    h = 1.0 / 48
    a = (_xb(t + h) - 2 * _xb(t) + _xb(t - h)) / (h * h)
    th = 0.55 * math.atan(a / 9.8)
    return float(np.clip(th, -0.42, 0.42))


def _pursuer_k(lf):
    return float(smoothstep(12, 26, lf))


class CamA:
    def __init__(self, t):
        self.t = t
        self.xc = _xc(t)
        self.z = VA * t

    def p(self, X, Y, Z):
        X, Y, Z = np.asarray(X, np.float64), np.asarray(Y, np.float64), np.asarray(Z, np.float64)
        Z = np.maximum(Z, 0.05)
        return np.stack([CXA + FA * (X - self.xc) / Z, CYA + FA * (HCA - Y) / Z], -1)


@lru_cache(maxsize=1)
def _grid_A():
    ys = np.arange(360, H, 2, dtype=np.float32) + 0.5
    xs = np.arange(0, W, 2, dtype=np.float32) + 0.5
    return xs[None, :], ys[:, None]


@lru_cache(maxsize=1)
def _sign_tex():
    tw, th = 960, 300
    tex = np.zeros((th, tw, 3), np.float32)
    tex[:] = hexc("#17603f")
    painted_texture(tex, 0.06, seed=5, scale=1.0)
    tex[:8] = tex[-8:] = WHITE
    tex[:, :8] = tex[:, -8:] = WHITE
    tex[:, 610:616] = WHITE
    big = text_mask("第七環状", 150, stroke=2)
    sm = text_mask("新港・臨海", 64, stroke=1)
    ex = text_mask("出口 7", 70, stroke=1)
    km = text_mask("2 km", 58, font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    for m, x, y in ((big, 310, 120), (sm, 300, 240), (ex, 785, 95), (km, 785, 200)):
        pm = place_mask(m, x, y, tw, th)
        tex = tex * (1 - pm[..., None]) + pm[..., None] * WHITE
    # 화살표
    arr = np.array([(785, 268), (745, 236), (770, 236), (770, 225), (800, 225), (800, 236), (825, 236)])
    poly_fill(tex, arr, WHITE)
    return tex


@lru_cache(maxsize=1)
def _sign2_tex():
    tw, th = 300, 300
    tex = np.zeros((th, tw, 3), np.float32)
    tex[:] = hexc("#1b3a8a")
    tex[:7] = tex[-7:] = WHITE
    tex[:, :7] = tex[:, -7:] = WHITE
    ring = np.zeros((th, tw), np.float32)
    cv2.circle(ring, (150, 150), 110, 1.0, -1, cv2.LINE_AA)
    tex = tex * (1 - ring[..., None]) + ring[..., None] * hexc("#e8202a")
    inner = np.zeros((th, tw), np.float32)
    cv2.circle(inner, (150, 150), 84, 1.0, -1, cv2.LINE_AA)
    tex = tex * (1 - inner[..., None]) + inner[..., None] * WHITE
    m = place_mask(text_mask("80", 110, font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), 150, 152, tw, th)
    tex = tex * (1 - m[..., None]) + m[..., None] * hexc("#1b3a8a")
    return tex


def _warp_quad(img, tex, quad, alpha=1.0, gain=1.0):
    """텍스처를 화면 사각형(quad: TL,TR,BR,BL)에 원근 투영 합성 (bbox 국소)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _bb(quad, 2, w, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    th, tw = tex.shape[:2]
    src = np.float32([(0, 0), (tw, 0), (tw, th), (0, th)])
    dst = np.float32(quad - (x0, y0))
    M = cv2.getPerspectiveTransform(src, dst)
    reg = cv2.warpPerspective(tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [_fix(quad, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    a = (m.astype(np.float32) / 255.0 * alpha)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + reg * gain * a


def _lamps_A(cam):
    """보이는 가로등 목록: (Z, X_pole, side)."""
    out = []
    for side, X, off in ((1, 6.3, 0.0), (-1, -6.3, LAMP_SP / 2)):
        base = math.floor((cam.z - off) / LAMP_SP)
        for k in range(base - 1, base + 13):
            Zw = k * LAMP_SP + off
            Z = Zw - cam.z + 0.0
            if 0.4 < Z < 300:
                out.append((Z, X, side))
    return out


def _road_A(cam, t, lf, bike_x, purs):
    """반해상도 도로/벽 레이어 → (rgb, alpha) 풀해상도."""
    xs, ys = _grid_A()
    xc = cam.xc
    dy = ys - CYA
    dyp = np.maximum(dy, 0.6)
    Zg = FA * HCA / dyp
    Xg = xc + (xs - CXA) * Zg / FA
    gnd = dy > 0.6
    rgb = np.zeros((ys.shape[0], xs.shape[1], 3), np.float32)
    alpha = np.zeros((ys.shape[0], xs.shape[1]), np.float32)
    lamps = _lamps_A(cam)
    sod = np.array([1.0, 0.50, 0.17], np.float32)

    # --- 노면 (우리 쪽 + 반대 차선)
    road = gnd & (Xg > -18.5) & (Xg < 6.0)
    Zr = np.where(road, Zg, 1.0)
    Xr = np.where(road, Xg, 0.0)
    tex = noise_lookup(Xr * 3.0, (Zr + cam.z) * 0.35, 11)
    tex2 = noise_lookup(Xr * 0.6, (Zr + cam.z) * 0.05, 12)
    pool = np.zeros_like(Zr)
    for Z, X, side in lamps:
        lx = X - side * 2.4
        pool += 1.0 / (1.0 + ((Xr - lx) / 3.4) ** 2 + ((Zr - Z) / 8.0) ** 2) ** 2
        if side < 0:     # 중앙분리대 가로등은 반대 차선도 비춤
            pool += 0.8 / (1.0 + ((Xr - (X - 2.4)) / 3.4) ** 2 + ((Zr - Z) / 8.0) ** 2) ** 2
    fog = np.clip(1 - Zr / 260.0, 0, 1)
    base = hexc("#17182a") * 1.0
    lum = (0.5 + 0.35 * tex + 0.25 * tex2)
    col = base[None, None, :] * lum[..., None]
    col += pool[..., None] * sod * (0.36 + 0.16 * tex[..., None])
    # 바이크 헤드라이트 (앞쪽 노면)
    dz = Zr - ZBK
    hb = np.exp(-((Xr - bike_x) / (0.9 + 0.09 * np.maximum(dz, 0))) ** 2) * smoothstep(1.0, 5.0, dz) * np.exp(-np.maximum(dz, 0) / 26.0)
    col += hb[..., None] * np.array([0.75, 0.95, 1.15], np.float32) * 1.35
    # 추격자 헤드라이트 + 바이크 그림자
    if purs > 0:
        xp = bike_x - 1.9
        lp = np.exp(-((Xr - xp - (Zr + 4) * 0.1) / (1.2 + Zr * 0.1)) ** 2) * np.exp(-Zr / 12.0) * smoothstep(0.0, 1.5, Zr)
        dvec = np.array([bike_x - xp, ZBK + 6.0])
        dvec /= np.linalg.norm(dvec)
        px_, pz_ = Xr - bike_x, Zr - ZBK
        s_ = px_ * dvec[0] + pz_ * dvec[1]
        lat = np.abs(px_ * dvec[1] - pz_ * dvec[0])
        shad = (smoothstep(-0.3, 0.2, s_) * (1 - smoothstep(9.0, 13.0, s_)) *
                (1 - smoothstep(0.26 + 0.05 * s_, 0.42 + 0.07 * s_, lat)))
        col += (lp * (1 - 0.95 * shad))[..., None] * np.array([0.85, 0.95, 1.1], np.float32) * 0.55 * purs
        col *= (1 - 0.5 * shad * purs)[..., None]
    onc = road & (Xg < -6.2)
    col[onc] *= 0.8
    col = col * fog[..., None] + hexc("#e0607a")[None, None, :] * (1 - fog[..., None]) * 0.6
    rgb[road] = col[road]
    alpha[road] = 1.0

    # --- 우측 방호벽 (X=6.0, 높이 1.2)
    xr = xs - CXA
    with np.errstate(divide="ignore", invalid="ignore"):
        Zw = np.where(xr > 1, FA * (6.0 - xc) / np.maximum(xr, 1e-3), 1e9)
    hw = HCA - dy * Zw / FA
    wallR = (xr > 1) & (Zw > 0.2) & (Zw < 400) & (hw >= 0) & (hw <= 1.2) & ~road
    if wallR.any():
        wl = np.zeros_like(Zw)
        for Z, X, side in lamps:
            if side > 0:
                wl += 1.0 / (1.0 + ((Zw - Z) / 5.0) ** 2)
        joint = (((Zw + cam.z) / 3.5) % 1.0) < 0.05
        joint &= Zw > 6
        wc = hexc("#2e2a46")[None, None, :] * (0.6 + 0.25 * noise_lookup(Zw * 0.3 + cam.z * 0.3, hw * 4, 13))[..., None]
        wc = wc + wl[..., None] * sod * 0.33 * (0.5 + 0.5 * (hw / 1.2))[..., None]
        wc = np.where(joint[..., None], wc * 0.55, wc)
        rail = (hw > 1.02)
        wc = np.where(rail[..., None], wc * 0.6 + sod * 0.4 * (0.5 + wl[..., None]) + 0.08, wc)
        f2 = np.clip(1 - np.broadcast_to(Zw, hw.shape) / 260.0, 0, 1)[..., None]
        wc = wc * f2 + hexc("#e0607a") * (1 - f2) * 0.6
        rgb[wallR] = wc[wallR]
        alpha[wallR] = 1.0
    # --- 중앙분리대 (X=-6.2, 높이 0.55) + 반대편 벽 (X=-18.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        Zm = np.where(xr < -1, FA * (-6.2 - xc) / np.minimum(xr, -1e-3), 1e9)
        Zf = np.where(xr < -1, FA * (-18.5 - xc) / np.minimum(xr, -1e-3), 1e9)
    hm = HCA - dy * Zm / FA
    med = (xr < -1) & (Zm > 0.2) & (hm >= 0) & (hm <= 0.55) & (gnd & (Xg <= -6.2) | (dy <= 0.6))
    med &= hm >= 0
    hf = HCA - dy * Zf / FA
    farw = (xr < -1) & (Zf > 0.2) & (hf >= 0) & (hf <= 1.2) & ~road & ~med
    if med.any():
        ml = np.zeros_like(Zm)
        for Z, X, side in lamps:
            if side < 0:
                ml += 1.0 / (1.0 + ((Zm - Z) / 4.0) ** 2)
        mc = hexc("#4a4262")[None, None, :] * (0.55 + 0.2 * (hm / 0.55))[..., None] + ml[..., None] * sod * 0.6
        mc = np.where((hm > 0.47)[..., None], mc * 0.5 + 0.35 * sod * (0.5 + ml[..., None]), mc)
        f2 = np.clip(1 - np.broadcast_to(Zm, hm.shape) / 260.0, 0, 1)[..., None]
        mc = mc * f2 + hexc("#e0607a") * (1 - f2) * 0.6
        rgb[med] = mc[med]
        alpha[med] = 1.0
    if farw.any():
        fc = hexc("#2a2440")[None, None, :] * np.ones_like(hf)[..., None]
        f2 = np.clip(1 - np.broadcast_to(Zf, hf.shape) / 260.0, 0, 1)[..., None]
        fc = fc * f2 + hexc("#e0607a") * (1 - f2) * 0.6
        rgb[farw] = fc[farw]
        alpha[farw] = 1.0
    # 풀해상도 업샘플 (상단 360px 은 비어 있음)
    full = np.zeros((H, W, 3), np.float32)
    fa = np.zeros((H, W), np.float32)
    full[360:] = cv2.resize(rgb, (W, H - 360), interpolation=cv2.INTER_LINEAR)
    fa[360:] = cv2.resize(alpha, (W, H - 360), interpolation=cv2.INTER_LINEAR)
    return full, fa


def noise_lookup(u, v, seed):
    """월드 좌표 (u,v) 로 타일 노이즈 샘플 (0..1)."""
    from ..core import noise_tex
    tex = noise_tex(seed, 1024, 1.8)
    mu = (u.astype(np.float32) * 37.0) % 1024.0
    mv = (v.astype(np.float32) * 37.0) % 1024.0
    if mu.shape != mv.shape:
        mu, mv = np.broadcast_arrays(mu, mv)
        mu, mv = np.ascontiguousarray(mu), np.ascontiguousarray(mv)
    return cv2.remap(tex, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _smear_hull(img, light, polyA, polyB, col, alpha, light_gain=0.0, lcol=None):
    """두 시점 다각형의 convex hull 로 모션 스미어."""
    pts = np.r_[polyA, polyB].astype(np.float32)
    hull = cv2.convexHull(pts)[:, 0, :].astype(np.float64)
    a = float(np.clip(alpha, 0.05, 1.0))
    poly_fill(img, hull, col, alpha=a)
    if light is not None and light_gain > 0:
        poly_fill(light, hull, (lcol if lcol is not None else col) * light_gain * a, mode="add")
    return hull


def _quad(cam, X0, X1, Y0, Y1, Z):
    return cam.p([X0, X1, X1, X0], [Y1, Y1, Y0, Y0], [Z, Z, Z, Z])


def _ground_quad(cam, X0, X1, Z0, Z1):
    return cam.p([X0, X1, X1, X0], [0, 0, 0, 0], [Z0, Z0, Z1, Z1])


def _pool_at(lamps, X, Z):
    s = 0.0
    for Zl, Xp, side in lamps:
        lx = Xp - side * 2.4
        s += 1.0 / (1.0 + ((X - lx) / 3.4) ** 2 + ((Z - Zl) / 8.0) ** 2) ** 2
    return s


def _draw_pole(img, light, cam, Z, X, side):
    sm = VA * SHUT
    sod = np.array([1.0, 0.5, 0.17], np.float32)
    w_px = FA * 0.26 / Z
    dx = abs(cam.p(X, 0, Z)[0] - cam.p(X, 0, Z + sm)[0])
    a = w_px / (w_px + dx)
    pc = hexc("#15131f") + sod * 0.10
    qa = _quad(cam, X - 0.13, X + 0.13, 0, 9.0, Z)
    qb = _quad(cam, X - 0.13, X + 0.13, 0, 9.0, Z + sm)
    _smear_hull(img, None, qa, qb, pc, a)
    # 하이라이트 엣지 (도로 쪽)
    xe = X - side * 0.1
    ea = _quad(cam, xe - 0.03, xe + 0.03, 0.2, 8.8, Z)
    eb = _quad(cam, xe - 0.03, xe + 0.03, 0.2, 8.8, Z + sm)
    _smear_hull(img, None, ea, eb, sod * 0.9, a * 0.9)
    # 암
    xa = X - side * 2.55
    aa = cam.p([X, xa, xa, X], [8.95, 8.7, 8.55, 8.8], [Z] * 4)
    ab = cam.p([X, xa, xa, X], [8.95, 8.7, 8.55, 8.8], [Z + sm] * 4)
    _smear_hull(img, None, aa, ab, pc, a)
    # 램프 헤드 (발광 하단)
    ha = cam.p([xa - 0.5, xa + 0.5, xa + 0.45, xa - 0.45], [8.75, 8.75, 8.42, 8.42], [Z] * 4)
    hb = cam.p([xa - 0.5, xa + 0.5, xa + 0.45, xa - 0.45], [8.75, 8.75, 8.42, 8.42], [Z + sm] * 4)
    _smear_hull(img, None, ha, hb, hexc("#1a1622"), a)
    la = cam.p([xa - 0.42, xa + 0.42, xa + 0.38, xa - 0.38], [8.5, 8.5, 8.38, 8.38], [Z] * 4)
    lb = cam.p([xa - 0.42, xa + 0.42, xa + 0.38, xa - 0.38], [8.5, 8.5, 8.38, 8.38], [Z + sm] * 4)
    hw = FA * 0.8 / Z
    dl = np.linalg.norm(cam.p(xa, 8.45, Z) - cam.p(xa, 8.45, Z + sm))
    al = hw / (hw + dl)
    _smear_hull(img, light, la, lb, np.array([3.4, 2.1, 0.9], np.float32), max(al, 0.35), light_gain=1.4,
                lcol=np.array([2.0, 1.0, 0.35], np.float32))
    return cam.p(xa, 8.4, Z)


def _cones_A(cam, lamps):
    """가로등 아래 연무 속 빛 원뿔 (1/4 해상도에서 블러)."""
    q = np.zeros((H // 4, W // 4), np.float32)
    for Z, X, side in lamps:
        if Z < 3.0 or Z > 140:
            continue
        xa = X - side * 2.55
        top = cam.p([xa - 0.4, xa + 0.4], [8.4, 8.4], [Z, Z]) / 4
        bot = cam.p([xa - 3.0, xa + 3.0], [0, 0], [Z, Z]) / 4
        poly = np.array([top[0], top[1], bot[1], bot[0]])
        mask_poly(q, poly, float(np.clip(6.0 / Z, 0.05, 0.6)), mode="add")
    q = cv2.GaussianBlur(q, (0, 0), 3.0)
    return cv2.resize(q, (W, H), interpolation=cv2.INTER_LINEAR)


def _gantry(img, light, cam, Zg):
    if Zg < 0.6 or Zg > 300:
        return
    sm = VA * SHUT
    dark = hexc("#1b1926")
    sod = np.array([1.0, 0.5, 0.17], np.float32)
    for X in (-7.0, 7.0):
        w_px = FA * 0.4 / Zg
        dx = abs(cam.p(X, 0, Zg)[0] - cam.p(X, 0, Zg + sm)[0])
        a = w_px / (w_px + dx)
        _smear_hull(img, None, _quad(cam, X - 0.2, X + 0.2, 0, 7.4, Zg), _quad(cam, X - 0.2, X + 0.2, 0, 7.4, Zg + sm), dark, a)
    dy_ = abs(cam.p(0, 7.0, Zg)[1] - cam.p(0, 7.0, Zg + sm)[1])
    bh = FA * 0.8 / Zg
    ab = bh / (bh + dy_)
    beam_a, beam_b = _quad(cam, -7.2, 7.2, 6.6, 7.4, Zg), _quad(cam, -7.2, 7.2, 6.6, 7.4, Zg + sm)
    _smear_hull(img, None, beam_a, beam_b, dark, ab)
    if ab > 0.35:
        # 트러스 사선
        for i in range(12):
            x0 = -7.0 + i * 1.2
            pts = cam.p([x0, x0 + 0.6, x0 + 1.2], [6.65, 7.35, 6.65], [Zg] * 3)
            ink(img, pts, max(1.5, FA * 0.06 / Zg), sod * 0.35, (0, 0), alpha=ab)
        ink(img, cam.p([-7.2, 7.2], [7.38, 7.38], [Zg, Zg]), max(1.5, FA * 0.05 / Zg), sod * 0.7, (0, 0), alpha=ab)
    # 표지판
    for (X0, X1, tex) in ((-1.8, 4.6, _sign_tex()), (-5.6, -3.6, _sign2_tex())):
        n = 3 if Zg < 14 else 1
        for k in range(n):
            Zs = Zg + sm * (k / max(1, n - 1) if n > 1 else 0.0)
            quad = _quad(cam, X0, X1, 4.6, 6.6, Zs)
            _warp_quad(img, tex, quad, alpha=(1.0 if n == 1 else 0.5 if k == 0 else 0.35),
                       gain=0.95 + 0.25 * float(np.exp(-Zg / 20)))
        quad = _quad(cam, X0, X1, 4.6, 6.6, Zg)
        if ab > 0.3:
            ink(img, np.r_[quad, quad[:1]], max(1.5, FA * 0.05 / Zg), INK, (0, 0), alpha=ab)
        # 조명 (아래 램프)
        for xx in np.linspace(X0 + 0.5, X1 - 0.5, 3):
            pl = cam.p(xx, 4.5, Zg)
            r = max(1.5, FA * 0.12 / Zg)
            poly_fill(img, ellip(pl[0], pl[1], r * 1.6, r * 0.6, 0, 12), np.array([3.0, 3.0, 2.6]))
            poly_fill(light, ellip(pl[0], pl[1], r * 2, r, 0, 12), np.array([1.4, 1.3, 1.0]), mode="add")


def _car(img, light, cam, X, Z, body=None, sil=None, seed=0, boil=1.0):
    """앞서 가는 차 (후면+측면 박스 셀). 가까우면 상대속도 스미어."""
    if Z < 0.8 or Z > 400:
        return
    if Z < 7.0:
        tmp = img.copy()
        _car_body(tmp, light, cam, X, Z + 0.55, body, seed, boil)
        _car_body(img, light, cam, X, Z, body, seed, boil)
        img[:] = img * 0.6 + tmp * 0.4
        return
    _car_body(img, light, cam, X, Z, body, seed, boil)


def _car_body(img, light, cam, X, Z, body=None, seed=0, boil=1.0):
    body = hexc("#2a2c40") if body is None else body
    sod = np.array([1.0, 0.5, 0.17], np.float32)
    L = 4.4
    xl, xr = X - 0.9, X + 0.9
    side_x = xl if X > cam.xc else xr
    b = boil if Z < 25 else 0.0
    lw = float(np.clip(FA * 0.012 / Z, 1.2, 4.5))
    sd = int(seed)
    # 측면
    side = cam.p([side_x] * 4, [0.32, 1.0, 1.0, 0.32], [Z, Z, Z + L, Z + L])
    cel(img, jitter(side, b, sd + 1) if b else side, body * 0.75, hi=mix(body, sod, 0.55),
        hi_P=cam.p([side_x] * 4, [0.9, 1.0, 1.0, 0.9], [Z, Z, Z + L, Z + L]), shade=body * 0.45,
        shade_P=cam.p([side_x] * 4, [0.32, 0.5, 0.5, 0.32], [Z, Z, Z + L, Z + L]), line_w=lw)
    sg = 1 if X > cam.xc else -1
    cx0, cx1 = side_x + sg * 0.2, side_x + sg * 0.32
    cabs = cam.p([cx0, cx1, cx1, cx0], [1.0, 1.42, 1.42, 1.0], [Z + 0.05, Z + 0.75, Z + 3.0, Z + 3.6])
    cel(img, jitter(cabs, b, sd + 2) if b else cabs, hexc("#121626"), hi=sod * 0.6,
        hi_P=cam.p([cx1, cx1, cx1, cx1], [1.37, 1.42, 1.42, 1.37], [Z + 0.72, Z + 0.75, Z + 3.0, Z + 3.04]), line_w=lw)
    # 후면
    rear = _quad(cam, xl, xr, 0.32, 1.0, Z)
    cel(img, jitter(rear, b, sd + 3) if b else rear, body, hi=mix(body, sod, 0.4),
        hi_P=_quad(cam, xl, xr, 0.92, 1.0, Z), line_w=lw)
    win = cam.p([xl + 0.2, xr - 0.2, xr - 0.32, xl + 0.32], [1.0, 1.0, 1.4, 1.4], [Z + 0.05, Z + 0.05, Z + 0.75, Z + 0.75])
    cel(img, jitter(win, b, sd + 4) if b else win, hexc("#0f1424"), hi=hexc("#6a8ac8"),
        hi_P=cam.p([xl + 0.3, xl + 0.6, xl + 0.7, xl + 0.4], [1.02, 1.02, 1.38, 1.38], [Z + 0.07, Z + 0.07, Z + 0.73, Z + 0.73]),
        line_w=lw)
    # 타이어
    for xx in (xl + 0.12, xr - 0.42):
        tq = _quad(cam, xx, xx + 0.3, 0.0, 0.34, Z + 0.4)
        cel(img, tq, TYRE, line_w=lw * 0.8)
    # 테일램프 (발광)
    for x0_, x1_ in ((xl + 0.06, xl + 0.6), (xr - 0.6, xr - 0.06)):
        tq = _quad(cam, x0_, x1_, 0.74, 0.88, Z - 0.01)
        poly_fill(img, tq, TAIL_HDR * 0.9)
        poly_fill(light, tq, TAIL_HDR * 0.7, mode="add")


def _smooth_path(P, k=3):
    """폴리라인 이동평균 스무딩 (끝점 유지)."""
    P = np.asarray(P, np.float64)
    if len(P) < 2 * k + 2:
        return P
    pad = np.r_[np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)]
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    out = np.stack([np.convolve(pad[:, j], ker, "valid") for j in range(2)], 1)
    out[0] = P[0]
    return out


def _ribbon(mask, pts, widths, alphas, levels=12):
    """폭/알파가 변하는 리본을 float 마스크에 (알파 단계별 묶음 fillPoly)."""
    P = np.asarray(pts, np.float64)
    if len(P) < 2:
        return
    tang = np.gradient(P, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    Lp = P + nrm * (widths[:, None] / 2)
    Rp = P - nrm * (widths[:, None] / 2)
    h, w = mask.shape
    x0, y0, x1, y1 = _bb(np.r_[Lp, Rp], 4, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    q = np.clip(np.round(np.asarray(alphas) * levels), 0, levels).astype(int)
    for lv in range(1, levels + 1):
        idx = np.nonzero(q[:-1] == lv)[0]
        if len(idx) == 0:
            continue
        polys = [_fix(np.array([Lp[i], Lp[i + 1], Rp[i + 1], Rp[i]]), x0, y0) for i in idx]
        tmp = np.zeros_like(m)
        cv2.fillPoly(tmp, polys, int(255 * lv / levels), cv2.LINE_AA, _SHIFT)
        m = cv2.max(m, tmp)
    reg = mask[y0:y1, x0:x1]
    np.maximum(reg, m.astype(np.float32) / 255.0, out=reg)


def _star(img, light, c, size, col, n_long=2, gain=1.0):
    """4방향 크로스 스타 글린트."""
    for k, a in enumerate((0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4)):
        L = size * (1.0 if k < n_long else 0.35)
        d = np.array([math.cos(a), math.sin(a)]) * L
        pts = np.array([c - d, c, c + d])
        ink(light, pts, max(2.0, size * 0.02), col * gain, (0.95, 0.95))


def render_A(ctx: Ctx) -> np.ndarray:
    lf = ctx.lframe
    t = ctx.lt
    cam = CamA(t)
    purs = _pursuer_k(lf)
    bx = _xb(t)
    sky, mid, ma = _layers_A()
    light = np.zeros((H, W, 3), np.float32)
    # 원경/중경 (VP 기준 확대 = 전진)
    def zoom(src, s, dx=0.0):
        M = np.array([[s, 0, CXA * (1 - s) + dx], [0, s, CYA * (1 - s)]], np.float32)
        return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img = zoom(sky, 1.0 + 0.04 * t, -cam.xc * 3)
    img += _searchlights(lf / 24.0 + 1.0, ((260, 0.4), (1700, 2.1), (620, 3.3)))[..., None] * np.array([0.55, 0.6, 0.85], np.float32) * 0.3
    s_mid = 1.0 + 0.30 * t
    midw = zoom(np.dstack([mid, ma]), s_mid, -cam.xc * 14)
    img = img * (1 - midw[..., 3:4]) + midw[..., :3]
    # 유선 (VP 방사)
    sl = _radial_streaks(lf)
    img += sl[..., None] * np.array([0.55, 0.75, 1.0], np.float32) * 0.12
    # 도로/벽
    rgb, ra = _road_A(cam, t, lf, bx, purs)
    img = img * (1 - ra[..., None]) + rgb * ra[..., None]
    lamps = _lamps_A(cam)
    sod = np.array([1.0, 0.5, 0.17], np.float32)
    # 차선
    sm = VA * SHUT
    for X in (-5.75, 5.75):
        q = _ground_quad(cam, X - 0.08, X + 0.08, 0.7, 320)
        poly_fill(img, q, mix(WHITE, sod, 0.35) * 0.75)
    for X, dim in ((-2.0, 1.0), (2.0, 1.0), (-10.3, 0.6), (-14.3, 0.6)):
        ph = cam.z % 12.0
        for k in range(0, 26):
            Z0 = k * 12.0 - ph
            Z1 = Z0 + 4.0 + sm
            if Z1 < 0.7:
                continue
            Z0 = max(Z0, 0.7)
            pz = _pool_at(lamps, X, (Z0 + Z1) / 2)
            c = (mix(WHITE, sod, 0.4) * (0.45 + 0.9 * min(pz, 1.2))) * dim
            q = _ground_quad(cam, X - 0.08, X + 0.08, Z0, Z1)
            poly_fill(img, q, c, alpha=float(np.clip(4.0 / (Z1 - Z0), 0.35, 0.95)))
    # 방호벽 상단 레일 라인 (크리스프)
    poly_fill(img, cam.p([6.0, 6.0, 6.0, 6.0], [1.2, 1.2, 1.12, 1.12], [0.6, 320, 320, 0.6]), sod * 0.85 + 0.1)
    poly_fill(img, cam.p([-6.2, -6.2, -6.2, -6.2], [0.55, 0.55, 0.5, 0.5], [0.6, 320, 320, 0.6]), sod * 0.7 + 0.08)
    # 연무 원뿔
    cones = _cones_A(cam, lamps)
    img += cones[..., None] * np.array([1.0, 0.55, 0.22], np.float32) * 0.55
    # 바이크 헤드라이트 빔 (앞쪽 연무)
    bsx = CXA + FA * (bx - cam.xc) / ZBK
    bsy = CYA + FA * HCA / ZBK
    vp = np.array([CXA + FA * (bx - cam.xc) / 60.0, CYA + FA * (HCA - 0.7) / 60.0])
    beam = np.zeros((H // 4, W // 4), np.float32)
    bp = np.array([(bsx - 60, bsy - 200), (bsx + 60, bsy - 200), vp + (90, 6), vp + (-90, 6)]) / 4
    mask_poly(beam, bp, 0.5)
    beam = cv2.resize(cv2.GaussianBlur(beam, (0, 0), 6), (W, H))
    img += beam[..., None] * np.array([0.6, 0.85, 1.0], np.float32) * 0.5

    # --- 깊이 정렬 오브젝트
    objs = []
    for Z, X, side in lamps:
        objs.append((Z, "pole", (Z, X, side)))
    Zg = GANTRY_Z0 - VA * t
    objs.append((Zg, "gantry", Zg))
    carA_Z = 16.0 - 26.0 * t
    objs.append((carA_Z, "car", (-4.1, carA_Z, hexc("#3c4866"), 1)))
    carB_Z = 70.0 - 17.0 * t
    objs.append((carB_Z, "car", (4.0, carB_Z, hexc("#4a3044"), 2)))
    objs.append((ZBK, "bike", None))
    objs.sort(key=lambda o: -o[0])
    tail_pt = None
    sil = np.zeros((H, W), np.float32)
    for Z, kind, data in objs:
        if kind == "pole":
            _draw_pole(img, light, cam, *data)
        elif kind == "gantry":
            _gantry(img, light, cam, data)
        elif kind == "car":
            _car(img, light, cam, data[0], data[1], data[2], seed=data[3] * 50 + ctx.drawing())
        elif kind == "bike":
            _oncoming(img, light, cam, t)
            # 접지 그림자 + 테일램프 노면 반사
            poly_fill(img, ellip(bsx, bsy + 4, 120, 16, 0, 24), (0.0, 0.0, 0.0), alpha=0.55)
            refl = np.zeros((H // 4, W // 4), np.float32)
            mask_poly(refl, ellip(bsx / 4, (bsy + 22) / 4, 26, 6, 0, 20), 1.0)
            refl = cv2.resize(cv2.GaussianBlur(refl, (0, 0), 3), (W, H))
            img += refl[..., None] * np.array([1.0, 0.1, 0.08], np.float32) * 0.45
            lb = np.zeros_like(light)
            tail_pt = _draw_bike_A(img, lb, ctx, cam, bx, purs, sil)
            light *= (1 - sil)[..., None]
            light += lb
    # 잔상 리본 (테일라이트 트레일)
    _trails_A(img, light, ctx, cam)
    # 추격자
    if purs > 0:
        _pursuer_A(img, light, ctx, purs)
    # 투과광 글로우
    add_glow(img, light, sigmas=(5, 18, 54), weights=(0.75, 0.5, 0.35))
    img += light * 0.25
    # 카메라 롤 + 진동 (1코마)
    roll = -_lean_A(t) * 9.0
    shake_y = 2.5 * wobble(t * 4.0, 7) + 1.5 * (hash01(91, lf) - 0.5)
    shake_x = 2.0 * wobble(t * 3.3, 8)
    img = cv2.warpAffine(img, _roll_M(roll, 1.05, shake_x, shake_y), (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return img


def _roll_M(deg, scale, dx, dy):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def _radial_streaks(lf):
    from ..core import radial_lines_mask
    q = radial_lines_mask(CXA / 4, CYA / 4, n=70, seed=1000 + lf, r_in=40, r_jit=1.5,
                          width=(0.0015, 0.005), w=W // 4, h=H // 4)
    return cv2.resize(q, (W, H), interpolation=cv2.INTER_LINEAR)


def _oncoming(img, light, cam, t):
    vrel = VA + 26.0
    sm = vrel * SHUT
    for k in range(18):
        lane = (-8.4, -12.3, -16.2)[k % 3]
        Z0 = hash01(500, k) * 280.0
        Z = (Z0 - vrel * t) % 280.0 + 2.0
        if Z < 4.0:
            continue
        for dx in (-0.72, 0.72):
            X = lane + dx
            a = cam.p(X, 0.62, Z)
            b = cam.p(X, 0.62, Z + sm)
            wpx = float(np.clip(FA * 0.16 / Z, 1.6, 12.0))
            col = np.array([2.6, 2.8, 3.2], np.float32) * min(1.0, 30.0 / Z + 0.35) * float(smoothstep(4.0, 14.0, Z))
            ink(img, np.array([a, (a + b) / 2, b]), wpx, col, (0, 0))
            ink(light, np.array([a, (a + b) / 2, b]), wpx * 1.4, col * 0.6, (0, 0))
            # 노면 반사
            ra = cam.p(X, -0.05, Z)
            rb = cam.p(X, -0.05, Z + sm * 3)
            ink(img, np.array([ra, rb]), wpx * 0.8, col * 0.25, (0.2, 0.5))


def _bike_screen_A(ctx, cam, bx):
    t2 = ctx.on(2)
    th = _lean_A(t2)
    if ctx.lframe >= 28:     # 컷 끝: 슬라이드 진입 준비로 크게 기울임
        th = th * (1 - smoothstep(28, 33, ctx.lframe)) + 0.38 * smoothstep(28, 33, ctx.lframe)
    sx = CXA + FA * (bx - cam.xc) / ZBK
    sy = CYA + FA * HCA / ZBK + 2.0 * math.sin(ctx.drawing() * 2.1)
    k = FA / ZBK * RU
    return sx, sy, k, th


def _draw_bike_A(img, light, ctx, cam, bx, purs, sil):
    sx, sy, k, th = _bike_screen_A(ctx, cam, bx)
    c, s = math.cos(th), math.sin(th)

    def X(P):
        P = np.asarray(P, np.float64)
        return np.c_[sx + k * (P[:, 0] * c + P[:, 1] * s), sy - k * (P[:, 1] * c - P[:, 0] * s)]

    return bike_rear(img, light, X, sil=sil, seed=ctx.drawing(), pursuer=purs, spin=ctx.lt * 7)


def _trails_A(img, light, ctx, cam):
    t = ctx.lt
    m_core = np.zeros((H, W), np.float32)
    m_wide = np.zeros((H, W), np.float32)
    n = 70
    T = 0.62
    taus = np.linspace(0, T, n)
    Zs = ZBK - TRAIL_V * taus
    ok = Zs > 0.45
    taus, Zs = taus[ok], Zs[ok]
    for off in (-0.2, 0.2):
        Xs = np.array([_xb(t - tau) for tau in taus]) + off
        lean = np.array([_lean_A(t - tau) for tau in taus])
        Ys = 0.66 + 0.0 * lean
        Xs = Xs + np.sin(lean) * 0.66 * 0.6
        P = cam.p(Xs, Ys, Zs)
        P = _smooth_path(P, 3)
        age = taus / T
        wd = np.minimum(FA * 0.05 / Zs, 46.0)
        al = (1 - age) ** 1.6
        _ribbon(m_core, P, np.maximum(wd * 0.2, 2.0), al)
        _ribbon(m_wide, P, wd * 1.2, al * 0.32)
    img += m_wide[..., None] * TAIL_HDR * 0.45
    img += m_core[..., None] * TRAIL_CORE * 0.85
    light += m_wide[..., None] * TAIL_HDR * 0.9 + m_core[..., None] * TAIL_HDR * 0.6


def _pursuer_A(img, light, ctx, purs):
    lf = ctx.lframe
    u = smoothstep(21, 33, lf + 0.0)
    q = np.zeros((H // 4, W // 4), np.float32)
    if u <= 0:
        # 화면 밖 — 빛만 좌하단에서 번짐
        mask_poly(q, ellip(-30 / 4, 1150 / 4, 90, 50, 0, 20), 1.0)
        q = cv2.resize(cv2.GaussianBlur(q, (0, 0), 16), (W, H))
        img += q[..., None] * np.array([0.8, 0.9, 1.0], np.float32) * 0.9 * purs
        return
    ex = ease_out(u, 2.2)
    cx = lerp(-330, 210, ex) + 12 * wobble(ctx.on(2) * 2, 41)
    cy = lerp(1230, 960, ex) + 5 * wobble(ctx.on(2) * 2.6, 42)
    sc = 1.15
    b = 1.1
    sd = ctx.drawing() * 7
    P = lambda pts: np.asarray(pts, np.float64) * sc + (cx, cy)
    # 추격 바이크 앞부분: 각진 다크 카울 + 이중 헤드라이트 + 앞바퀴
    tyre = P([(-40, 60), (40, 60), (46, 200), (-46, 200)])
    cel(img, jitter(tyre, b, sd + 9), TYRE, line_w=4)
    nose = P([(-210, 40), (-150, -60), (-40, -120), (60, -130), (170, -110), (260, -50), (300, 30), (230, 110), (-170, 120)])
    cel(img, jitter(smooth(nose, 4), b, sd + 1), hexc("#3a3e58"), shade=hexc("#181a2a"),
        shade_P=jitter(P([(-260, 50), (340, 30), (340, 220), (-260, 220)]), b, sd + 2),
        hi=hexc("#8a90b8"), hi_P=jitter(P([(-140, -58), (-40, -112), (60, -122), (-30, -98), (-130, -46)]), b, sd + 5),
        rim=CYAN * 0.9, rim_dir=(0.3, -1), rim_w=5, line_w=5.0)
    ink(img, jitter(P([(-190, 60), (40, 70), (280, 40)]), b, sd + 6), 3.0, INK, (0.2, 0.2))
    scr = P([(-40, -116), (20, -200), (150, -205), (190, -108), (60, -126)])
    cel(img, jitter(scr, b, sd + 3), hexc("#1a2440"), hi=hexc("#9ff6ff") * 1.2,
        hi_P=P([(40, -196), (70, -196), (40, -128), (20, -124)]), line_w=3.5, alpha=0.95)
    for hx_ in (-70, 150):
        hl = P(ellip(hx_, 10, 44, 22, 0, 20))
        cel(img, jitter(hl, 0.6, sd + 4 + hx_), np.array([4.0, 4.2, 4.6], np.float32), line_w=3.0,
            light=light, light_col=np.array([2.6, 2.8, 3.2], np.float32) * purs)
    c = P([(40, 10)])[0]
    _star(img, light, c, 520 * purs, np.array([1.3, 1.5, 1.9], np.float32), n_long=1, gain=1.0)
    # 헥사 고스트 (렌즈 플레어) — 광원→화면중심 선상
    cen = np.array([W / 2, H / 2])
    for f_, r_, col in ((0.55, 34, hexc("#3ff0ff")), (1.25, 60, hexc("#b46aff")), (1.7, 22, hexc("#ff8a2a"))):
        pp = c + (cen - c) * f_
        poly_fill(light, circ(pp[0], pp[1], r_, 6, 0.3), col * 0.18 * purs, mode="add")
    mask_poly(q, ellip(c[0] / 4, c[1] / 4, 55, 26, 0, 20), 1.0)
    q = cv2.resize(cv2.GaussianBlur(q, (0, 0), 14), (W, H))
    img += q[..., None] * np.array([0.8, 0.9, 1.05], np.float32) * 0.8 * purs


# ---------------------------------------------------------------------------
# 컷 B — 사이드 뷰 파워 슬라이드
# ---------------------------------------------------------------------------
FB, CXB, CYB, HCB = 1500.0, 960.0, 520.0, 1.0
ZBB = 3.7                   # 바이크 차선 깊이
Z_FAR = 10.9                # 건너편 방호벽
V0B = 40.0
FB_T1 = (40 - CUT) / 24.0   # 슬라이드 시작 (컷 B 내부 초)
FB_T2 = (STOP - CUT) / 24.0  # 정지
LAMP_SP_B = 24.0
STRIP_W = W + 1500
DEPTH_B = {"far": 420.0, "mid": 110.0, "near": 42.0}


@lru_cache(maxsize=1)
def _kin_B():
    """바이크 X(τ) 적분 테이블 (τ: 컷 B 내부 초, 과거 −3s 까지)."""
    dt = 1.0 / 480
    taus = np.arange(-3.0, 2.0, dt)
    u = np.clip((taus - FB_T1) / (FB_T2 - FB_T1), 0, 1)
    v = V0B * (1 - u) ** 1.35
    X = np.cumsum(v) * dt
    X -= np.interp(0.0, taus, X)
    return taus, X, v


def _XbB(tau):
    taus, X, v = _kin_B()
    return np.interp(tau, taus, X)


def _VbB(tau):
    taus, X, v = _kin_B()
    return float(np.interp(tau, taus, v))


def _off_B(tau):
    return lerp(-170.0, 80.0, float(smoothstep(FB_T1 - 0.05, FB_T2 + 0.05, tau)))


def _XcB(tau):
    return _XbB(tau) - _off_B(tau) * ZBB / FB


def _pose_B(lf):
    """(yaw, lean, rider pose, look, bank_deg) — 작화 프레임 기준."""
    f = float(lf)
    yaw = (0.52 * ease_out(remap(f, 40, 45), 2.0) - 0.10 * remap(f, 45, 52)
           - 0.30 * ease_in_out(remap(f, 52, 58)))
    lean = (0.44 * ease_out(remap(f, 40, 45), 2.0) - 0.06 * remap(f, 45, 52)
            - 0.34 * ease_in_out(remap(f, 52, 58)))
    pose = ease_in_out(remap(f, 56, 62))
    look = ease_in_out(remap(f, 62, 66))
    pitch = (0.11 * ease_out(remap(f, 41, 47)) - 0.11 * ease_in_out(remap(f, 52, 58))
             - 0.035 * math.sin(math.pi * remap(f, 58, 62)))
    return yaw, lean, pose, look, pitch


def _bike_xform(u, v, w, yaw, lean, pitch):
    """로컬(m) → 바이크 기준 월드 오프셋 (dX, Y, dZ_toward_cam). pitch>0 = 뒤가 들림(앞바퀴 접지점 기준)."""
    fu = 290 * SU
    cp, sp = math.cos(pitch), math.sin(pitch)
    du = u - fu
    u0 = fu + du * cp + v * sp
    v0 = -du * sp + v * cp
    cl, sl = math.cos(lean), math.sin(lean)
    w1 = w * cl - v0 * sl
    v1 = v0 * cl + w * sl
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    u2 = u0 * cy_ + w1 * sy_
    w2 = w1 * cy_ - u0 * sy_
    return u2, v1, w2


def _proj_B(Xb, Xc, yaw, lean, pitch=0.0, shake=(0.0, 0.0)):
    def proj(U):
        U = np.asarray(U, np.float64)
        u2, v1, w2 = _bike_xform(U[:, 0] * SU, U[:, 1] * SU, U[:, 2] * SU, yaw, lean, pitch)
        X = Xb + u2
        Z = ZBB - w2
        return np.c_[CXB + FB * (X - Xc) / Z + shake[0], CYB + FB * (HCB - v1) / Z + shake[1]]
    return proj


def _world_B(Xc, X, Y, Z, shake=(0.0, 0.0)):
    X, Y, Z = np.asarray(X, np.float64), np.asarray(Y, np.float64), np.asarray(Z, np.float64)
    return np.stack([CXB + FB * (X - Xc) / Z + shake[0], CYB + FB * (HCB - Y) / Z + shake[1]], -1)


def _lamp_world_B(tau, dv=0.0):
    """테일램프 월드 좌표."""
    lf = tau * 24 + CUT
    yaw, lean, _, _, pitch = _pose_B(lf)
    u2, v1, w2 = _bike_xform(-452 * SU, (400 + dv) * SU, 0.0, yaw, lean, pitch)
    return _XbB(tau) + u2, v1, ZBB - w2


@lru_cache(maxsize=1)
def _layers_B():
    rng = np.random.default_rng(4401)
    sky = vgradient([(0, hexc("#04051c")), (0.35, hexc("#0e0c34")), (0.62, hexc("#2a1550")),
                     (0.82, hexc("#6a2464")), (0.95, hexc("#b0406a")), (1.0, hexc("#e8607a"))],
                    y0=0, y1=CYB)
    sky[int(CYB):] = hexc("#e8607a")
    d = np.sqrt(((np.arange(W, dtype=np.float32)[None, :] - 1250) / 2.2) ** 2 +
                (np.arange(H, dtype=np.float32)[:, None] - CYB) ** 2)
    sky += (np.exp(-(d / 260.0) ** 2) * 0.35)[..., None] * hexc("#ff7a5a")
    painted_texture(sky, 0.07, seed=77, scale=4.0, stretch=8.0)
    for _ in range(80):
        x, y = rng.uniform(0, W), rng.uniform(0, CYB * 0.5)
        poly_fill(sky, circ(x, y, rng.uniform(0.8, 1.7), 8), np.array([0.9, 0.9, 1.0]) * rng.uniform(0.3, 0.9))
    haze_col = hexc("#d0587a")
    strips = {}
    for name, (hmin, hmax, bwmin, bwmax, hz, body, cw, ch, neon_p, nsz) in {
        "far": (30, 170, 26, 70, 0.62, hexc("#3a2250"), 4, 5, 0.0, 0),
        "mid": (80, 290, 60, 140, 0.36, hexc("#1e1838"), 6, 8, 0.4, 28),
        "near": (180, 440, 150, 280, 0.10, hexc("#120f24"), 11, 15, 0.9, 50),
    }.items():
        rgb = np.zeros((H, STRIP_W, 3), np.float32)
        a = np.zeros((H, STRIP_W), np.float32)
        lt = np.zeros_like(rgb)
        x = -50.0
        base = 500.0
        while x < STRIP_W + 50:
            bw = rng.uniform(bwmin, bwmax)
            bh = rng.uniform(hmin, hmax)
            top = base - bh
            _building(rgb, a, x, x + bw, top, base + 30, rng, body * rng.uniform(0.85, 1.15), hz,
                      haze_col, lit=0.32 if name == "near" else 0.4, cw=cw, ch=ch, rim_col=hexc("#ff8a6a") * 0.55,
                      rim_side=1, glow_bottom=0.8, hdr=1.1)
            if rng.random() < 0.45:
                cx_ = x + bw / 2
                cw_ = bw * rng.uniform(0.3, 0.6)
                ch_ = rng.uniform(15, 50) * (bh / 300)
                crown = np.array([(cx_ - cw_ / 2, top + 1), (cx_ - cw_ / 2, top - ch_), (cx_ + cw_ / 2, top - ch_), (cx_ + cw_ / 2, top + 1)])
                poly_fill(rgb, crown, body * 0.9)
                mask_poly(a, crown, 1.0)
                ah = rng.uniform(20, 80) * (bh / 300)
                ax = cx_
                ink(rgb, np.array([(ax, top - ch_), (ax, top - ch_ - ah)]), 2.5, body * 1.4, (0, 0.2))
                mask_poly(a, np.array([(ax - 1.5, top - ch_), (ax + 1.5, top - ch_), (ax + 1.5, top - ch_ - ah), (ax - 1.5, top - ch_ - ah)]), 1.0)
                b_ = circ(ax, top - ch_ - ah, 3.5, 10)
                poly_fill(rgb, b_, np.array([3.2, 0.35, 0.25]))
                poly_fill(lt, b_, np.array([2.0, 0.2, 0.1]), mode="add")
            if rng.random() < neon_p:
                sz = int(nsz * rng.uniform(0.8, 1.2))
                vert = rng.random() < 0.75
                nx_ = x + bw * rng.uniform(0.25, 0.75)
                ny_ = top + bh * rng.uniform(0.15, 0.45)
                _neon_sign(rgb, a, lt, nx_, ny_, sz, rng, vertical=vert)
            gap = {"far": (-6, 10), "mid": (0, 60), "near": (90, 380)}[name]
            x += bw + rng.uniform(*gap)
        b16 = blur(lt, 16)
        rgb += blur(lt, 4) * 0.9 + b16 * 0.5
        painted_texture(rgb, 0.05, seed={"far": 61, "mid": 62, "near": 63}[name], scale=3.0, stretch=1.0, angle=1.57)
        rgb *= a[..., None]
        rgb += (b16 * 0.45 + blur(lt, 40) * 0.35) * (1 - a[..., None])
        strips[name] = (rgb.astype(np.float32), a.astype(np.float32))
    return sky.astype(np.float32), strips


def _strip_view(name, Xc, v, shake):
    sky, strips = _layers_B()
    rgb, a = strips[name]
    s = FB / DEPTH_B[name]
    off = Xc * s + 300.0
    o = int(math.floor(off))
    fr = off - o
    o = int(np.clip(o, 0, STRIP_W - W - 2))
    crop = np.dstack([rgb[:, o:o + W + 1], a[:, o:o + W + 1]])
    M = np.float32([[1, 0, -fr + shake[0]], [0, 1, shake[1]]])
    crop = cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    L = int(round(v * s * SHUT))
    if L > 2:
        crop = cv2.blur(crop, (L, 1), borderType=cv2.BORDER_REPLICATE)
    return crop[..., :3], crop[..., 3]


def _searchlights(t, cx_list=((300, 0.0), (1500, 1.3), (900, 2.4)), base_y=520.0):
    q = np.zeros((H // 4, W // 4), np.float32)
    for bx_, ph in cx_list:
        ang = -math.pi / 2 + 0.35 * math.sin(t * 0.7 + ph)
        L = 1400
        d = np.array([math.cos(ang), math.sin(ang)])
        n = np.array([-d[1], d[0]])
        b0 = np.array([bx_, base_y])
        tip = b0 + d * L
        poly = np.array([b0 - n * 6, b0 + n * 6, tip + n * 120, tip - n * 120]) / 4
        mask_poly(q, poly, 0.5, mode="add")
    q = cv2.GaussianBlur(q, (0, 0), 3)
    q *= np.clip((np.arange(H // 4)[:, None] / (H / 4)) * 1.6, 0, 1)
    return cv2.resize(q, (W, H))


def _barrier_B(img, light, Xc, v, shake, tau):
    """건너편 방호벽 + 가로등 기둥 (1D 열 프로파일 + 모션 블러)."""
    sod = np.array([1.0, 0.5, 0.17], np.float32)
    xs = np.arange(W, dtype=np.float32) - shake[0]
    Xw = Xc + (xs - CXB) * Z_FAR / FB
    sm = v * SHUT
    # 가로등 위치 → 조명 프로파일
    lampX = (np.floor(Xw / LAMP_SP_B)[None, :] + np.array([-1, 0, 1, 2])[:, None]) * LAMP_SP_B
    lit = np.sum(1.0 / (1.0 + ((Xw[None, :] - lampX) / 5.0) ** 2), 0)
    # 이음매 (모션 블러: 셔터 동안 평균)
    jt = np.zeros(W, np.float32)
    pole = np.zeros(W, np.float32)
    ns = 6
    for k in range(ns):
        Xs = Xw - sm * k / ns
        jt += ((Xs / 3.5) % 1.0 < 0.035).astype(np.float32)
        pole += ((Xs % LAMP_SP_B) < 0.2).astype(np.float32)
    refl = np.zeros(W, np.float32)
    for k in range(ns):
        refl += (((Xw - sm * k / ns) % 4.0) < 0.1).astype(np.float32)
    jt /= ns
    pole /= ns
    refl /= ns
    if sm * FB / Z_FAR > 6:
        k_ = int(min(301, sm * FB / Z_FAR)) | 1
        jt = cv2.blur(jt[None, :], (k_, 1))[0]
        pole = cv2.blur(pole[None, :], (k_, 1))[0]
    y_top = CYB + FB * (HCB - 1.2) / Z_FAR + shake[1]
    y_rail = CYB + FB * (HCB - 1.08) / Z_FAR + shake[1]
    y_bot = CYB + FB * HCB / Z_FAR + shake[1]
    r0, r1 = int(max(0, math.floor(y_top) - 1)), int(min(H, math.ceil(y_bot) + 2))
    ys = np.arange(r0, r1, dtype=np.float32)[:, None]
    tex = noise_lookup(np.broadcast_to(Xw[None, :] * 0.8, (r1 - r0, W)), np.broadcast_to(ys * 0.02, (r1 - r0, W)), 21)
    g = np.clip((ys - y_top) / (y_bot - y_top), 0, 1)
    wc = hexc("#3a3552")[None, None, :] * (0.55 + 0.25 * tex[..., None]) * (0.8 + 0.4 * g[..., None])
    wc = wc + (lit[None, :, None] * sod * 0.5) * (0.6 + 0.4 * g[..., None])
    wc = wc * (1 - 0.45 * jt[None, :, None])
    rail = (ys >= y_top) & (ys < y_rail)
    wc = np.where(rail[..., None], wc * 0.5 + sod * 0.45 * (0.5 + lit[None, :, None]) + 0.1, wc)
    m = np.clip(np.minimum(ys - y_top + 0.5, y_bot - ys + 0.5), 0, 1)
    # 반사판 (캣츠아이) — 고속이면 주황 스트릭
    y_ref = CYB + FB * (HCB - 0.75) / Z_FAR + shake[1]
    rr = np.clip(1.6 - np.abs(ys - y_ref) / 2.2, 0, 1)
    if sm * FB / Z_FAR > 6:
        refl = cv2.blur(refl[None, :], (int(min(301, sm * FB / Z_FAR)) | 1, 1))[0] * 2.5
    wc = wc + rr[..., None] * refl[None, :, None] * np.array([2.4, 1.0, 0.25], np.float32) * (0.4 + lit[None, :, None])
    reg = img[r0:r1]
    reg[:] = reg * (1 - m[..., None]) + wc * m[..., None]
    # 기둥 (위로 화면 밖까지) — 기둥이 있는 열만
    cols = np.nonzero(pole > 1e-3)[0]
    if len(cols):
        c0, c1 = cols.min(), cols.max() + 1
        yb = int(min(H, math.ceil(y_bot)))
        pm = pole[None, c0:c1, None]
        pc = hexc("#16131f") + sod * 0.08
        img[:yb, c0:c1] = img[:yb, c0:c1] * (1 - pm) + pc * pm
        ph = np.clip(np.roll(pole, 3) - pole, 0, 1)
        img[:yb] += ph[None, :, None] * sod * 0.4


def _road_B(img, Xc, v, shake, tau, skid_pts=None):
    """노면: 행별 깊이, 차선 대시, 나트륨 풀, 모션 블러."""
    y0 = int(CYB + FB * HCB / Z_FAR + shake[1])
    y0 = max(0, y0)
    ys = np.arange(y0, H, dtype=np.float32)[:, None] + 0.5 - shake[1]
    xs = np.arange(W, dtype=np.float32)[None, :] + 0.5 - shake[0]
    Z = FB * HCB / np.maximum(ys - CYB, 1.0)
    X = Xc + (xs - CXB) * Z / FB
    sm = v * SHUT
    ns = 4 if sm > 0.05 else 1
    tex = np.zeros(np.broadcast(X, Z).shape, np.float32)
    dash = np.zeros_like(tex)
    for k in range(ns):
        Xs = X - sm * k / ns
        tex += noise_lookup(Xs * 2.5, np.broadcast_to(Z * 3.0, Xs.shape), 31)
        dash += (((Xs / 12.0) % 1.0) < 0.3).astype(np.float32)
    tex /= ns
    dash /= ns
    if sm * FB / ZBB > 8:
        # 가까운 행일수록 더 길게 번진 노면 (행 밴드별 박스 블러)
        tex = cv2.blur(tex, (int(min(201, sm * FB / 6.0)) | 1, 1))
    sod = np.array([1.0, 0.5, 0.17], np.float32)
    lampX = (np.floor(X / LAMP_SP_B) + np.array([0, 1])[:, None, None]) * LAMP_SP_B
    pool = np.sum(1.0 / (1.0 + ((X[None] - lampX) / 4.5) ** 2 + ((Z[None] - 8.4) / 3.0) ** 2) ** 2, 0)
    col = hexc("#1b1c2c")[None, None, :] * (0.55 + 0.45 * tex[..., None])
    col = col + pool[..., None] * sod * 0.6
    # 차선 라인: Z=6.2 대시, Z=10.3 실선, Z=2.4 실선(화면 밖)
    lane = np.exp(-((Z - 6.2) / 0.075) ** 2) * dash
    edge = np.exp(-((Z - 10.35) / 0.07) ** 2)
    lc = mix(WHITE, sod, 0.35) * 0.85
    col = col + (lane + edge)[..., None] * (lc - col) * 0.9
    # 바이크/테일램프의 붉은 반사광은 렌더 후단에서
    img[y0:] = col
    return y0


def _trail_B(img, light, tau, Xc, shake, sil=None):
    """월드 고정 테일라이트 잔상 리본."""
    T = 2.2
    n = 160
    taus = tau - np.linspace(0, T, n) ** 1.6 / T ** 0.6
    pts, ages = [], []
    pts2 = []
    for tt in taus:
        X, Y, Z = _lamp_world_B(tt)
        pts.append((X, Y, Z))
        pts2.append(_lamp_world_B(tt, dv=-26))
        ages.append(tau - tt)
    pts2 = np.array(pts2)
    pts = np.array(pts)
    ages = np.array(ages)
    P = _world_B(Xc, pts[:, 0], pts[:, 1], pts[:, 2], shake)
    vis = (P[:, 0] > -400) & (P[:, 0] < W + 400)
    if vis.sum() < 2:
        return
    idx = np.nonzero(vis)[0]
    i0, i1 = max(0, idx[0] - 1), min(n, idx[-1] + 2)
    P, ages, Zs = P[i0:i1], ages[i0:i1], pts[i0:i1, 2]
    Pb = _world_B(Xc, pts2[i0:i1, 0], pts2[i0:i1, 1], pts2[i0:i1, 2], shake)
    al = np.clip(1 - ages / T, 0, 1) ** 1.4
    wd = FB * 0.06 / Zs
    m_core = np.zeros((H, W), np.float32)
    m_wide = np.zeros((H, W), np.float32)
    _ribbon(m_core, P, np.maximum(wd * 0.35, 2.0), al)
    _ribbon(m_wide, P, wd * 1.25, al * 0.5)
    # 두 번째 가는 리본 (사이드 램프)
    _ribbon(m_core, Pb, np.maximum(wd * 0.14, 1.2), al * 0.7)
    _ribbon(m_wide, Pb, wd * 0.6, al * 0.3)
    img += m_wide[..., None] * TAIL_HDR * 0.6 + m_core[..., None] * TRAIL_CORE * 0.95
    light += m_wide[..., None] * TAIL_HDR * 0.9 + m_core[..., None] * TAIL_HDR * 0.5


def _sparks_B(img, light, tau, Xc, shake):
    """발판 긁힘 불꽃 (분석적 궤적, 결정적)."""
    m_hot = np.zeros((H, W), np.float32)
    m_old = np.zeros((H, W), np.float32)
    f_now = tau * 24 + CUT
    g = 9.8
    for fe in range(42, 56):
        if fe > f_now:
            break
        te = (fe - CUT) / 24.0
        rng = np.random.default_rng([9090, fe])
        n = 34 if fe < 51 else 16
        yaw, lean, _, _, _ = _pose_B(fe)
        # 긁히는 지점: 바이크 하부 뒤쪽 (먼 쪽 발판)
        ex = _XbB(te) - 0.30 * math.cos(yaw)
        ez = ZBB + 0.30 * math.sin(yaw) - 0.12
        vb = _VbB(te)
        sub = rng.uniform(0, 1, n)            # 프레임 내 방출 시점
        vx = vb * rng.uniform(0.15, 0.75, n) + rng.normal(0, 2.0, n)
        vy = rng.uniform(1.0, 6.5, n)
        vz = rng.uniform(-5.5, 1.5, n)
        life = rng.uniform(0.16, 0.42, n)
        bright = rng.uniform(0.6, 1.0, n)
        for i in range(n):
            t0 = te + sub[i] / 24.0
            a1 = tau - t0
            if a1 <= 0 or a1 > life[i]:
                continue
            a0 = max(0.0, a1 - 1.0 / 40.0)
            pts = []
            for a in (a0, (a0 + a1) / 2, a1):
                x = ex + vx[i] * (1 - math.exp(-3.5 * a)) / 3.5
                y = 0.04 + vy[i] * a - 0.5 * g * a * a
                if y < 0:
                    y = -y * 0.25
                z = ez + vz[i] * a
                pts.append((x, y, z))
            pts = np.array(pts)
            P = _world_B(Xc, pts[:, 0], pts[:, 1], np.maximum(pts[:, 2], 1.0), shake)
            age = a1 / life[i]
            tgt = m_hot if age < 0.45 else m_old
            wpx = max(1, int(round(FB * 0.012 / pts[-1, 2] * (1.3 - age))))
            cv2.polylines(tgt, [np.round(P * 16).astype(np.int32)], False,
                          float(bright[i] * (1 - age * 0.6)), wpx, cv2.LINE_AA, 4)
    img += m_hot[..., None] * np.array([3.6, 3.0, 1.6], np.float32) + m_old[..., None] * np.array([3.0, 1.2, 0.3], np.float32)
    light += m_hot[..., None] * np.array([2.4, 1.6, 0.6], np.float32) + m_old[..., None] * np.array([1.6, 0.5, 0.1], np.float32)


def _smoke_puffs(tau_draw):
    """타이어 연기 퍼프 목록: (X, Y, Z, r, alpha, seed) — 작화 시간 기준."""
    out = []
    f_now = tau_draw * 24 + CUT
    for fe in range(41, 60):
        if fe > f_now:
            break
        te = (fe - CUT) / 24.0
        rng = np.random.default_rng([7070, fe])
        vb = _VbB(te)
        burst = fe == STOP
        n = 1 if fe < STOP else (5 if burst else 1)
        yaw, lean, _, _, _ = _pose_B(fe)
        rx = _XbB(te) - 0.72 * math.cos(yaw)
        rz = ZBB + 0.72 * math.sin(yaw)
        for i in range(n):
            vals = rng.random(9)
            life = 0.75 + 0.5 * vals[0]
            age = tau_draw - te
            if age <= 0 or age > life:
                continue
            if burst:
                ang = -0.3 - 2.5 * (i / max(1, n - 1))
                vx = 3.2 * math.cos(ang)
                vy = 0.6 + 0.8 * vals[1]
            else:
                vx = vb * (0.35 + 0.3 * vals[1])
                vy = 0.35 + 0.4 * vals[2]
            dx = vx * (1 - math.exp(-age * 2.8)) / 2.8
            x = rx + dx + (vals[3] - 0.5) * 0.2
            y = 0.12 + vy * (1 - math.exp(-age * 1.5)) / 1.5
            z = rz + 0.25 + age * (0.4 + 0.8 * vals[4])
            r = (0.10 + 0.34 * math.sqrt(age)) * (0.85 + 0.3 * vals[5]) * (1.25 if burst else 1.0)
            a = (1 - (age / life) ** 1.3) * 0.8
            out.append((x, y, z, r, a, fe * 31 + i))
    return out


def _draw_puff(img, cx, cy, r, alpha, seed, rim_side=-1):
    rng = np.random.default_rng(seed)
    n = 14
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = int(rng.integers(3, 5))
    rad = r * (1 + 0.14 * np.sin(th * k + rng.uniform(0, 6)))
    P = smooth(np.c_[cx + rad * np.cos(th), cy + rad * 0.85 * np.sin(th)], 4)
    S = smooth(np.c_[cx + r * 0.28 + rad * 0.92 * np.cos(th), cy + r * 0.34 + rad * 0.8 * np.sin(th)], 3)
    Hh = smooth(np.c_[cx - r * 0.3 + rad * 0.5 * np.cos(th), cy - r * 0.32 + rad * 0.42 * np.sin(th)], 3)
    cel(img, P, hexc("#b8a8c8"), shade=hexc("#6e5c88"), shade_P=S, hi=hexc("#ffc8b8"), hi_P=Hh,
        line_w=max(1.5, r * 0.03), line=hexc("#4a3a64"), alpha=float(np.clip(alpha, 0, 1)))


def _ghosts_B(img, light, sil, k, seed=0):
    """잔상(残像): 실루엣에서 뒤로 뻗는 붉은 투과광 스미어 + 에코 윤곽 하나."""
    if k <= 0:
        return
    ys, xs = np.nonzero(sil[::4, ::4] > 0.5)
    if len(xs) == 0:
        return
    y0, y1 = max(0, ys.min() * 4 - 8), min(H, ys.max() * 4 + 12)
    x1 = min(W, xs.max() * 4 + 8)
    x0 = max(0, xs.min() * 4 - 900)
    reg = sil[y0:y1, x0:x1]
    acc = reg.copy()
    s_ = 1
    while s_ < 1024:
        sh = np.zeros_like(acc)
        sh[:, :-s_] = acc[:, s_:]
        np.maximum(acc, sh * math.exp(-s_ / 260.0), out=acc)
        s_ *= 2
    rows = np.random.default_rng(seed).random(y1 - y0).astype(np.float32)
    rows = cv2.GaussianBlur(rows[:, None], (0, 0), 1.5)[:, 0]
    rows = 0.35 + 0.9 * np.clip((rows - 0.3) * 2.0, 0, 1)
    m = np.clip(acc - reg, 0, 1) * rows[:, None] * k
    col = np.array([1.9, 0.14, 0.2], np.float32)
    img[y0:y1, x0:x1] += m[..., None] * col * 0.8
    light[y0:y1, x0:x1] += m[..., None] * col * 0.35
    # 에코 윤곽 (한 장 뒤의 그림)
    d = 240
    sh = np.zeros_like(reg)
    sh[:, :-d] = reg[:, d:]
    edge = np.clip(sh - cv2.erode(sh, np.ones((7, 7), np.uint8)), 0, 1) * np.clip(sh - reg, 0, 1)
    img[y0:y1, x0:x1] += edge[..., None] * np.array([2.6, 0.5, 0.45], np.float32) * 0.8 * k


def _near_pole_B(img, tau, Xc, v, shake):
    """카메라 바로 앞을 스치는 가로등 기둥: 불투명 실루엣 + 뒤로 늘어진 스미어 꼬리."""
    for fpass in (37.0, 45.0):
        Xp = _XcB((fpass - CUT) / 24.0)
        Zp = 2.2
        x_now = CXB + FB * (Xp - Xc) / Zp + shake[0]
        x_prev = CXB + FB * (Xp - Xc + v * SHUT * 0.8) / Zp + shake[0]
        wpx = FB * 0.16 / Zp
        xa, xb_ = x_now - wpx / 2, max(x_prev, x_now) + wpx / 2
        if xb_ < 0 or xa > W:
            continue
        c0, c1 = int(max(0, math.floor(xa))), int(min(W, math.ceil(xb_)))
        xs = np.arange(c0, c1, dtype=np.float32) + 0.5
        core = np.clip(np.minimum(xs - xa, x_now + wpx / 2 - xs) + 0.5, 0, 1)
        tail = np.clip(1 - (xs - (x_now + wpx / 2)) / max(1.0, xb_ - x_now - wpx / 2), 0, 1) ** 1.5 * 0.6
        tail *= xs > x_now + wpx / 2
        a = np.maximum(core, tail)[None, :, None]
        img[:, c0:c1] = img[:, c0:c1] * (1 - a) + hexc("#0a0812") * a
        # 도로 쪽 나트륨 림
        e = np.clip(1 - np.abs(xs - (xa + 6)) / 5, 0, 1)[None, :, None] * core[None, :, None]
        img[:, c0:c1] += e * np.array([1.0, 0.5, 0.17], np.float32) * 0.7


def render_B(ctx: Ctx) -> np.ndarray:
    lf = ctx.lframe
    tau = (lf - CUT) / 24.0
    tdraw = (ctx.on(2) * 24 - CUT) / 24.0 if lf >= CUT + 1 else tau
    Xb = _XbB(tau)
    Xc = _XcB(tau)
    v = _VbB(tau)
    # 셰이크 (정지 임팩트)
    shake = (0.0, 0.0)
    if lf >= STOP:
        a = 22.0 * math.exp(-(lf - STOP) / 2.5)
        shake = (a * math.sin((lf - STOP) * 2.9 + 0.5), a * 0.8 * math.cos((lf - STOP) * 3.7))
    elif lf < 42:
        shake = (1.5 * wobble(tau * 6, 51), 2.5 * wobble(tau * 7, 52))
    sky, strips = _layers_B()
    light = np.zeros((H, W, 3), np.float32)
    img = sky.copy()
    if shake != (0.0, 0.0):
        img = cv2.warpAffine(img, np.float32([[1, 0, shake[0] * 0.3], [0, 1, shake[1] * 0.3]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    img += _searchlights(lf / 24.0 + 3)[..., None] * np.array([0.5, 0.55, 0.75], np.float32) * 0.35
    for name in ("far", "mid", "near"):
        rgb, a = _strip_view(name, Xc, v, shake)
        img = img * (1 - a[..., None]) + rgb
    _barrier_B(img, light, Xc, v, shake, tau)
    y0r = _road_B(img, Xc, v, shake, tau)
    # 젖은 노면: 방호벽 위 밝은 것들의 세로 반사
    hr = min(360, H - y0r, y0r)
    src = img[y0r - hr:y0r][::-1]
    br = np.clip(src - 0.5, 0, None)
    br = cv2.blur(br, (5, 71))
    fall = np.linspace(0.9, 0.15, hr, dtype=np.float32)[:, None, None]
    img[y0r:y0r + hr] += br * fall * 0.55
    # 유선 (가로) — 속도에 비례
    sp = v / V0B
    if sp > 0.05:
        from ..core import streak_lines_mask
        sl = streak_lines_mask(0.0, n=80, seed=3000 + lf, phase=lf * 400.0, length=(300, 1300),
                               width=(1.0, 4.5), w=W, h=H)
        sl[int(CYB + 140):] *= 0.6
        img += sl[..., None] * np.array([0.8, 0.85, 1.0], np.float32) * 0.5 * sp

    # 스키드 마크 (월드 고정, 검은 타이어 자국 + 뜨거운 긁힘)
    _skid_B(img, light, tau, Xc, shake)
    yaw, lean, pose, look, pitch = _pose_B(ctx.drawing() * 2) if lf >= 40 else _pose_B(lf)
    proj = _proj_B(Xb, Xc, yaw, lean, pitch, shake)
    # 바이크 밑 그림자 + 붉은 노면 반사
    cpts = proj(np.array([[-290, 0, 0], [290, 0, 0]], np.float64))
    cx_, cy_ = cpts.mean(0)
    poly_fill(img, ellip(cx_, cy_ + 6, abs(cpts[1, 0] - cpts[0, 0]) * 0.62, 22, 0, 30), (0, 0, 0), alpha=0.5)
    lw = _lamp_world_B(tau)
    lp = _world_B(Xc, lw[0], 0.0, lw[2], shake)
    q = np.zeros((H // 4, W // 4), np.float32)
    mask_poly(q, ellip(lp[0] / 4 - 20, lp[1] / 4 + 4, 70, 10, 0, 24), 1.0)
    q = cv2.resize(cv2.GaussianBlur(q, (0, 0), 5), (W, H))
    img += q[..., None] * np.array([1.0, 0.1, 0.08], np.float32) * 0.7

    puffs = _smoke_puffs(tdraw)
    back = [p for p in puffs if p[2] >= ZBB]
    front = [p for p in puffs if p[2] < ZBB]
    for (x, y, z, r, a, sd) in sorted(back, key=lambda p: -p[2]):
        c = _world_B(Xc, x, y, z, shake)
        _draw_puff(img, c[0], c[1], FB * r / z, a, sd)
    # 트레일 (바이크 뒤)
    _trail_B(img, light, tau, Xc, shake)
    sil = np.zeros((H, W), np.float32)
    lb = np.zeros_like(light)
    tmp = img.copy()
    spin = Xb / (R_TYRE * SU)
    bike_side(tmp, lb, proj, pose=pose, look=look, seed=ctx.drawing(), spin=spin,
              speed=min(1.0, v / 12.0), sil=sil, glint=_glint(lf))
    ghost_k = 1.0 - smoothstep(40, 47, lf)
    _ghosts_B(img, light, sil, ghost_k, seed=ctx.drawing())
    img = img * (1 - sil[..., None]) + tmp * sil[..., None]
    light *= (1 - sil)[..., None]
    light += lb
    # 불꽃 / 앞쪽 연기
    _sparks_B(img, light, tau, Xc, shake)
    for (x, y, z, r, a, sd) in sorted(front, key=lambda p: -p[2]):
        c = _world_B(Xc, x, y, z, shake)
        _draw_puff(img, c[0], c[1], FB * r / z, a, sd)
    _near_pole_B(img, tau, Xc, v, shake)
    add_glow(img, light, sigmas=(5, 18, 54), weights=(0.75, 0.5, 0.35))
    img += light * 0.25
    # 정지 임팩트: 집중선 + 플래시
    if STOP <= lf <= STOP + 3:
        from ..core import radial_lines_mask
        k = 1.0 - (lf - STOP) / 4.0
        rl = radial_lines_mask(cx_ / 4, (cy_ - 180) / 4, n=90, seed=600 + lf, r_in=520 / 4, r_jit=0.6,
                               width=(0.0015, 0.005), w=W // 4, h=H // 4)
        rl = cv2.resize(rl, (W, H))
        img = img * (1 - rl[..., None] * 0.6 * k) + rl[..., None] * np.array([1.0, 0.95, 0.9]) * 0.6 * k
    if lf == STOP:
        img = img * 1.25 + np.array([0.22, 0.10, 0.10], np.float32)
    # 홀드: 천천히 T.U. (영웅 포즈 쪽으로)
    if lf > STOP:
        zk = 1.0 + 0.075 * ease_out(remap(lf, STOP, FRAMES - 1), 1.6)
        fx_, fy_ = cx_ + 20, cy_ - 230
        M = np.float32([[zk, 0, fx_ * (1 - zk)], [0, zk, fy_ * (1 - zk)]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img


def _glint(lf):
    return float(max(0.0, 1.0 - abs(lf - 66) / 2.5)) if 63 <= lf <= 69 else 0.0


def _skid_B(img, light, tau, Xc, shake):
    if tau < FB_T1 + 0.04:
        return
    n = 60
    ts = np.linspace(FB_T1 + 0.04, tau, n)
    P, Ph, al = [], [], []
    for tt in ts:
        yaw, lean, _, _, _ = _pose_B(tt * 24 + CUT)
        X = _XbB(tt) - 0.72 * math.cos(yaw)
        Z = ZBB + 0.72 * math.sin(yaw)
        P.append(_world_B(Xc, X, 0.0, Z, shake))
        Xh = _XbB(tt) - 0.28 * math.cos(yaw)
        Zh = ZBB + 0.28 * math.sin(yaw) - 0.12
        Ph.append(_world_B(Xc, Xh, 0.0, Zh, shake))
        al.append(tau - tt)
    P, Ph, al = np.array(P), np.array(Ph), np.array(al)
    m = np.zeros((H, W), np.float32)
    _ribbon(m, P, np.full(n, 26.0), np.full(n, 0.75))
    img *= (1 - m[..., None] * 0.65)
    mh = np.zeros((H, W), np.float32)
    hot = np.clip(1 - al / 0.5, 0, 1) ** 2
    if (tau * 24 + CUT) < 57:
        _ribbon(mh, Ph, np.full(n, 5.0), hot)
    img += mh[..., None] * np.array([3.0, 1.2, 0.3], np.float32)
    light += mh[..., None] * np.array([1.6, 0.5, 0.1], np.float32)


def render(ctx: Ctx) -> np.ndarray:
    if ctx.lframe < CUT:
        return render_A(ctx)
    return render_B(ctx)
