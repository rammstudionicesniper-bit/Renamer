"""거울 / 정체성 붕괴 — 퍼펙트 블루 오마주 (60f, 8.5–11.0s).

레이어 (뒤 → 앞, 기준 좌표 = 카메라 배율 1)
  1. 세트 BG (정적 캐시): 과슈풍 분장실 벽(에어브러시 전구 빛), 벽에 걸린 무대의상, 포스터,
     래커 거울 프레임 + 전구 소켓
  2. 유리: 반사 플레이트(방 안 반사 배경 + 오리지널 아이돌 흉상 셀 + 핑크/블루 조명 젤 + 유리 광택)
     → 금 간 뒤에는 보로노이 파편(각자 어긋난/회전된/색이 다른 반사, 잉크 테두리 + 반짝이는 림)
     → 파편이 빠진 자리에는 거울 뒤의 '빛'(투과광) + TV 노이즈
  3. 세트 FG (정적 캐시): 화장대 상판과 소품, 전경의 소녀 뒷모습 실루엣(림라이트)
  4. 전구 발광(HDR) + 글로우, 먼지, 불꽃, 유리 가루, 날아오는 파편(깊이 정렬), 광선
  5. 화면 공간: 카메라(푸시인/셰이크, 1코마) → 글리치 띠/주사선/노이즈, 자막(RGB 분리),
     반전 임팩트, 흰 플래시(다음 컷 폭발로 연결)
타이밍
  f00–17  온전한 거울. 천천히 T.U. 반사 속 그녀가 혼자 깜빡이고(f8) 카메라를 보고(f10) 웃는다(f14).
  f18     금! (반전 프레임, 셰이크, 전구 파열) — 균열이 충격점에서 1코마로 번진다.
  f20–29  어긋난 파편들: 웃는 얼굴 / 이목구비 없는 얼굴 / 동공이 작아진 얼굴 / 감은 눈 … 조각이 떨어지고
          균열 사이로 빛이 샌다.
  f30     산산조각 (반전 프레임, 줌 스미어, 집중선) → 파편이 회전하며 카메라 쪽으로 (2코마).
  f44     풀 노이즈 + 큰 자막.  f52–56 거울 뒤의 빛이 차오름.  f57–59 흰 플래시.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from ..core import (FONT_JP, H, W, Ctx, _SHIFT, _fix, add_glow, blur, catmull, cel_shape, clamp,
                    ease_in, ease_in_out, ease_out, fill_ellipse, fill_poly, hash01, hexc, ink,
                    jitter, mix, painted_texture, radial_field, radial_lines_mask, smoothstep,
                    text_mask, transform, vgradient, zoom_blur)

NAME = "s4_mirror"
FRAMES = 60

# 사운드 싱크 포인트 (장면 내부 프레임)
EVENTS = [
    (0, "heartbeat"),
    (4, "pulse"),        # 전구 하나가 지직거리며 꺼짐
    (8, "blink"),        # 거울 속 그녀가 혼자 눈을 감았다 뜸
    (12, "glitch"),
    (14, "heartbeat"),   # 반사가 웃기 시작 (짧은 T.U. 스냅)
    (15, "glitch"),
    (18, "impact"),      # 금이 감 (반전 프레임)
    (18, "flash"),
    (19, "spark"),       # 전구 두 개 파열
    (19, "glitch"),
    (24, "glitch"),
    (26, "heartbeat"),   # 균열 사이로 빛이 새기 시작
    (27, "glitch"),
    (30, "shatter"),     # 거울 산산조각 (듀오톤 반전 프레임)
    (30, "flash"),
    (30, "shockwave"),
    (31, "spark"),       # 전구 네 개 파열 + 유리 가루
    (31, "whoosh"),      # 파편이 카메라 쪽으로 날아옴 (줌 스미어 + 집중선)
    (32, "glitch"),
    (34, "passby"),      # 웃는 얼굴의 큰 파편이 전경으로 스침
    (36, "glitch"),
    (39, "glitch"),
    (44, "glitch"),      # 풀 노이즈 + 큰 자막
    (49, "glitch"),
    (52, "swell"),       # 거울 뒤의 빛이 차오름 → 역광 실루엣
    (53, "glitch"),
    (57, "flash"),       # 흰 플래시 → 폭발 컷
]

# ---------------------------------------------------------------------------
# 색
# ---------------------------------------------------------------------------
INK = hexc("#0a0510")
SKIN = hexc("#f2d2c4")
PINK = hexc("#ff4f9a")
BLUE = hexc("#3a6cff")
FLUO = hexc("#b8ffcf")
LIP = hexc("#d4143a")
GOLD = hexc("#ffd36a")
ICE = hexc("#bfe8ff")
BULB = np.array([3.2, 2.45, 1.65], np.float32)          # 전구 코어 (HDR)
BULB_GLOW = np.array([1.0, 0.62, 0.36], np.float32)
WHITE_HOT = np.array([2.6, 2.7, 3.1], np.float32)
RIM_BLUE = hexc("#7ab0ff")
WASH = np.array([1.7, 1.66, 1.75], np.float32)           # 흰 플래시
SIL_COL = np.array([0.10, 0.04, 0.14], np.float32)       # 역광 실루엣

C_SKIN = mix(SKIN, PINK, 0.12)
C_SKIN_SH = hexc("#a2709e")
C_HAIR = hexc("#3d2b62")
C_HAIR_SH = hexc("#1b1236")
C_HAIR_HI = hexc("#b48ae8")
C_DRESS = hexc("#f2509e")
C_DRESS_SH = hexc("#8a2470")
C_FRILL = hexc("#fff2fa")
C_FRILL_SH = hexc("#b9a2e0")
C_BOW = hexc("#e3203c")
C_BOW_SH = hexc("#861034")
C_EYEW = hexc("#f7f2ff")
C_EYEW_SH = hexc("#a9a4dc")
C_IRIS = hexc("#3e2680")
C_IRIS_HI = hexc("#9a78ea")
C_PUPIL = hexc("#120822")
C_BACKING = hexc("#3a3046")

# ---------------------------------------------------------------------------
# 기하 (기준 좌표)
# ---------------------------------------------------------------------------
FRAME_OUT = (300, 40, 1620, 892)
GLASS = (370, 110, 1550, 860)
PX0, PY0, PX1, PY1 = 330, 70, 1590, 900          # 반사 플레이트 영역
PW, PH = PX1 - PX0, PY1 - PY0
P0 = np.array([PX0, PY0], np.float64)
FACE_C = (1000.0, 430.0)                          # 반사 속 얼굴 중심 (기준 좌표)
FS = 1.14                                         # 흉상 배율
NB = -34.0                                        # 목 아래(몸통) 세로 보정
IMPACT = (962.0, 392.0)                           # 금이 시작되는 점 (얼굴 왼쪽 관자놀이)
CAM_C = (985.0, 450.0)
LIGHT = np.array([-0.62, -0.78])                  # 파편 림이 빛나는 방향 (왼쪽 위 전구)

T_CRACK = 18
T_SEP = 20
T_SHATTER = 30
T_WHITE = 57

BULBS = ([(x, 75.0) for x in np.linspace(360, 1560, 11)] +
         [(335.0, y) for y in np.linspace(175, 815, 6)] +
         [(1585.0, y) for y in np.linspace(175, 815, 6)])
BULB_R = 19.0
POP = {5: 19, 19: 19, 2: 31, 9: 31, 12: 31, 21: 31}   # 전구 인덱스 → 파열 프레임


# ---------------------------------------------------------------------------
# 작은 헬퍼
# ---------------------------------------------------------------------------
def _rr(x0, y0, x1, y1, r, n=7):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return np.array(pts, np.float64)


def _ell_pts(cx, cy, rx, ry, n=40, rot=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = math.cos(rot), math.sin(rot)
    x, y = rx * np.cos(th), ry * np.sin(th)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)


def _star_pts(cx, cy, r, rot=0.0, k=5, inner=0.45):
    pts = []
    for i in range(k * 2):
        a = math.radians(rot) - math.pi / 2 + i * math.pi / k
        rr = r if i % 2 == 0 else r * inner
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return np.array(pts)


def _twinkle(img, x, y, size, color, rot=0.0):
    """4방향 십자 반짝임 (80년대 애니의 '키라리')."""
    for k, (ln, wd) in enumerate(((size, size * 0.07), (size * 0.5, size * 0.05))):
        for j in range(2):
            a = math.radians(rot + 45 * k + 90 * j)
            dx, dy = math.cos(a), math.sin(a)
            nx, ny = -dy, dx
            poly = [(x - dx * ln, y - dy * ln), (x + nx * wd, y + ny * wd),
                    (x + dx * ln, y + dy * ln), (x - nx * wd, y - ny * wd)]
            fill_poly(img, poly, color, mode="add")
    fill_ellipse(img, x, y, size * 0.12, size * 0.12, color, mode="add")


def _ring(img, pts, width, color):
    """긴 닫힌 선 (cv2 직접 — 큰 bbox 합성을 피함)."""
    cv2.polylines(img, [_fix(pts)], True, tuple(float(c) for c in color), int(round(width)), cv2.LINE_AA, _SHIFT)
    return img


def _glow2d(m, sig, wts):
    out = np.zeros_like(m)
    for s, w in zip(sig, wts):
        out += blur(m, s) * w
    return out


def _clip_fill(img, base_pts, sh_pts, color, alpha=1.0):
    """base 다각형 안으로 클리핑한 추가 셀 톤."""
    allp = np.vstack([np.asarray(base_pts).reshape(-1, 2), np.asarray(sh_pts).reshape(-1, 2)])
    h, w = img.shape[:2]
    x0 = int(max(0, math.floor(np.asarray(base_pts)[:, 0].min() - 3)))
    y0 = int(max(0, math.floor(np.asarray(base_pts)[:, 1].min() - 3)))
    x1 = int(min(w, math.ceil(np.asarray(base_pts)[:, 0].max() + 4)))
    y1 = int(min(h, math.ceil(np.asarray(base_pts)[:, 1].max() + 4)))
    if x1 <= x0 or y1 <= y0 or len(allp) == 0:
        return img
    a = np.zeros((y1 - y0, x1 - x0), np.uint8)
    b = np.zeros_like(a)
    cv2.fillPoly(a, [_fix(base_pts, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    cv2.fillPoly(b, [_fix(sh_pts, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    m = (cv2.min(a, b).astype(np.float32) / 255.0)[..., None] * alpha
    reg = img[y0:y1, x0:x1]
    reg *= (1 - m)
    reg += m * np.asarray(color, np.float32)
    return img


def _stamp_mask(img, m, cx, cy, color, rot_deg=0.0, alpha=1.0, mode="over", scale=1.0):
    """작은 마스크를 국소 영역에 합성 (전체 화면 warp 없이)."""
    mh, mw = m.shape
    R = int(math.hypot(mw, mh) * scale / 2) + 3
    x0, y0 = int(cx) - R, int(cy) - R
    M = cv2.getRotationMatrix2D((mw / 2, mh / 2), rot_deg, scale)
    M[0, 2] += (cx - x0) - mw / 2
    M[1, 2] += (cy - y0) - mh / 2
    loc = cv2.warpAffine(m, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR)
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(img.shape[1], x0 + 2 * R), min(img.shape[0], y0 + 2 * R)
    if X1 <= X0 or Y1 <= Y0:
        return img
    a = loc[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0][..., None] * alpha
    reg = img[Y0:Y1, X0:X1]
    col = np.asarray(color, np.float32)
    if mode == "add":
        reg += a * col
    else:
        reg *= (1 - a)
        reg += a * col
    return img


def _halfplane(poly, a, b, c):
    """a*x + b*y <= c 쪽만 남기기 (Sutherland–Hodgman)."""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        fp = a * P[0] + b * P[1] - c
        fq = a * Q[0] + b * Q[1] - c
        if fp <= 0:
            out.append(P)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append(P + (Q - P) * t)
    return np.array(out) if out else np.zeros((0, 2))


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _centroid(p):
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = cr.sum() / 2
    if abs(A) < 1e-6:
        return p.mean(0)
    return np.array([((x + xn) * cr).sum() / (6 * A), ((y + yn) * cr).sum() / (6 * A)])


def _hanging_dress(cx, top, s=1.0):
    pts = [(-30, 0), (-80, 16), (-112, 52), (-96, 104), (-66, 96), (-60, 170), (-122, 480), (-60, 470),
           (0, 486), (60, 470), (122, 480), (60, 170), (66, 96), (96, 104), (112, 52), (80, 16), (30, 0),
           (14, 22), (0, 26), (-14, 22)]
    return catmull(np.array(pts) * s + [cx, top], 6, closed=True)


# ---------------------------------------------------------------------------
# 1. 세트 (정적)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _set_back() -> np.ndarray:
    img = vgradient([(0, hexc("#170d2c")), (0.45, hexc("#271336")), (1, hexc("#120a1e"))])
    d = radial_field(960, 470, sx=1.45, sy=1.0)
    img += (np.exp(-(d / 620) ** 2) * 0.50)[..., None] * hexc("#ff8a5a")
    img += (np.exp(-(radial_field(110, 360, sx=1.0, sy=1.6) / 440) ** 2) * 0.26)[..., None] * BLUE
    img += (np.exp(-(radial_field(1850, 380, sx=1.0, sy=1.5) / 380) ** 2) * 0.22)[..., None] * BLUE
    painted_texture(img, 0.10, seed=401, scale=3.0, stretch=5.0, angle=1.35)
    for x in (150, 1770):
        fill_poly(img, [(x - 2, 0), (x + 2, 0), (x + 2, H), (x - 2, H)], hexc("#0c0614"), 0.5)

    # 왼쪽 벽: 옷걸이에 걸린 무대의상
    ink(img, [(190, 92), (200, 98), (198, 112), (190, 118)], 3, INK, (0, 0))
    ink(img, [(110, 168), (190, 122), (270, 168)], 5, hexc("#1a1020"), (0.05, 0.05))
    dress = _hanging_dress(190, 150, 0.95)
    cel_shape(img, dress, hexc("#2a3e78") * 0.85, hexc("#121a40"),
              [(196, 140), (320, 140), (320, 640), (232, 640), (214, 420), (204, 300)], line_w=3.2, seed=3,
              hi=hexc("#4a6ab0") * 0.8, hi_pts=[(90, 150), (150, 150), (120, 300), (60, 300)])
    rng = np.random.default_rng(77)
    for _ in range(30):
        x, y = rng.uniform(110, 270), rng.uniform(320, 600)
        if abs(x - 190) < (y - 300) * 0.2 + 52:
            fill_ellipse(img, x, y, 1.7, 1.7, hexc("#9fd0ff") * rng.uniform(0.6, 1.6), mode="add")
    ink(img, catmull([(78, 606), (130, 596), (190, 612), (250, 596), (302, 606)], 6), 2.4, INK, (0.2, 0.2))

    # 오른쪽 벽: 공연 포스터 (오리지널)
    pc = (1772, 360)
    rot = math.radians(2.5)

    def _pr(x, y):
        x, y = x - pc[0], y - pc[1]
        return (pc[0] + x * math.cos(rot) - y * math.sin(rot), pc[1] + x * math.sin(rot) + y * math.cos(rot))
    poster = [_pr(1660, 150), _pr(1890, 150), _pr(1890, 570), _pr(1660, 570)]
    fill_poly(img, poster, hexc("#5a1e4a") * 0.85)
    fill_poly(img, [_pr(1660, 150), _pr(1890, 150), _pr(1890, 300), _pr(1660, 330)], hexc("#7a2a5e") * 0.85)
    fill_poly(img, _star_pts(*_pr(1775, 300), 74, 8), hexc("#c89a40") * 0.7)
    ink(img, _star_pts(*_pr(1775, 300), 74, 8), 2.5, INK, (0, 0), closed=True)
    _stamp_mask(img, text_mask("ライブ", 46, FONT_JP, stroke=1), *_pr(1775, 470), hexc("#e8d0e0") * 0.65,
                rot_deg=-2.5)
    ink(img, poster, 3, INK, (0, 0), closed=True)
    for p, ox in ((poster[0], 8), (poster[1], -8)):
        fill_ellipse(img, p[0] + ox, p[1] + 8, 5, 5, hexc("#d02040"))

    # 거울 프레임 (래커칠 나무)
    fo = _rr(*FRAME_OUT, 28)
    frame_hi = _rr(FRAME_OUT[0], FRAME_OUT[1], FRAME_OUT[2], FRAME_OUT[1] + 26, 20)
    cel_shape(img, fo, hexc("#4a2046"), hexc("#2a1030"),
              [(1460, 30), (1640, 30), (1640, 900), (1460, 900)], line_w=4.5, seed=11,
              hi=hexc("#8a4058"), hi_pts=frame_hi)
    trim = _rr(GLASS[0] - 9, GLASS[1] - 9, GLASS[2] + 9, GLASS[3] + 9, 14)
    _ring(img, trim, 7, hexc("#b88a48"))
    _ring(img, trim, 2, hexc("#ffe0a0"))
    gx0, gy0, gx1, gy1 = GLASS
    img[gy0 + 4:gy1 - 4, gx0 + 4:gx1 - 4] = hexc("#0a0612")
    _ring(img, _rr(*GLASS, 10), 3, INK)
    for i, (x, y) in enumerate(BULBS):
        fill_ellipse(img, x, y, BULB_R + 8, BULB_R + 8, hexc("#2a2430"))
        fill_ellipse(img, x - 3, y - 3, BULB_R + 6, BULB_R + 6, hexc("#6a6070"), alpha=0.5)
        ink(img, _ell_pts(x, y, BULB_R + 8, BULB_R + 8), 2.5, INK, (0, 0), closed=True)
        fill_ellipse(img, x, y, BULB_R, BULB_R, hexc("#4a3e56"))
        fill_ellipse(img, x - 6, y - 7, 5, 4, hexc("#c8b8d8"), alpha=0.8)
        ink(img, _ell_pts(x, y, BULB_R, BULB_R), 1.8, INK, (0, 0), closed=True)
    return img


@lru_cache(maxsize=1)
def _set_front():
    """화장대 + 소품 + 전경 소녀 뒷모습. (rgb, alpha)"""
    img = np.zeros((H, W, 3), np.float32)
    top = [(0, 884), (W, 884), (W, 968), (0, 968)]
    fill_poly(img, top, hexc("#7a3e5e"))
    fill_poly(img, [(0, 968), (W, 968), (W, H), (0, H)], hexc("#26101e"))
    reg = img[884:968]
    xs = np.arange(W, dtype=np.float32)
    pool = np.exp(-((xs - 960) / 700) ** 2)[None, :, None]
    yy = np.linspace(1.0, 0.55, 84, dtype=np.float32)[:, None, None]
    reg *= 0.6 + 0.8 * pool * yy
    reg += pool * yy * hexc("#ff9a70") * 0.18
    painted_texture(img[884:], 0.08, seed=401, scale=2.0, stretch=8.0, angle=0.0, ox=300, oy=120)
    ink(img, [(0, 968), (W, 968)], 4, INK, (0, 0))
    ink(img, [(0, 972), (W, 972)], 2, hexc("#ffb090") * 0.9, (0, 0))
    ink(img, [(0, 884), (W, 884)], 3, INK, (0, 0))

    # 립스틱 (세운 것 / 눕힌 것)
    tube = [(694, 868), (722, 868), (722, 932), (694, 932)]
    cel_shape(img, tube, hexc("#d8a848"), hexc("#8a6024"), [(712, 860), (730, 860), (730, 940), (712, 940)],
              line_w=2.5, seed=21)
    bullet = [(698, 868), (698, 836), (704, 826), (718, 818), (718, 868)]
    cel_shape(img, bullet, LIP, hexc("#7a0a20"), [(711, 810), (730, 810), (730, 870), (711, 870)], line_w=2.5, seed=22)
    fill_poly(img, [(700, 846), (703, 832), (706, 832), (703, 850)], hexc("#ff9ab0"))
    cel_shape(img, [(752, 934), (836, 926), (838, 946), (754, 954)], hexc("#e0e0f0"), hexc("#8a8aa8"),
              [(752, 945), (840, 937), (840, 960), (752, 960)], line_w=2.5, seed=23)
    cel_shape(img, [(836, 926), (862, 925), (864, 945), (838, 946)], LIP, None, line_w=2.5, seed=24)
    # 콤팩트
    cel_shape(img, _ell_pts(1170, 944, 56, 16), hexc("#e8b0c8"), hexc("#9a5a7a"),
              [(1110, 946), (1240, 946), (1240, 970), (1110, 970)], line_w=2.5, seed=25)
    lid = _ell_pts(1176, 892, 50, 44, rot=math.radians(-6))
    cel_shape(img, lid, hexc("#e8b0c8"), hexc("#9a5a7a"), [(1196, 840), (1240, 840), (1240, 950), (1196, 950)],
              line_w=2.5, seed=26)
    cel_shape(img, _ell_pts(1176, 892, 38, 33, rot=math.radians(-6)), hexc("#8ab0e8"), hexc("#4a5aa8"),
              [(1180, 850), (1230, 850), (1230, 940), (1180, 940)], line_w=2.0, seed=27)
    fill_poly(img, [(1150, 880), (1172, 866), (1178, 872), (1156, 888)], hexc("#ffffff") * 1.2)
    # 향수병
    body = [(1340, 934), (1330, 900), (1338, 856), (1362, 840), (1398, 840), (1422, 856), (1430, 900), (1420, 934)]
    cel_shape(img, body, hexc("#e04a9a"), hexc("#7a1a5a"), [(1392, 830), (1440, 830), (1440, 940), (1392, 940)],
              line_w=3.0, seed=28, hi=hexc("#ff9ad0"),
              hi_pts=[(1338, 856), (1362, 840), (1372, 840), (1352, 930), (1340, 930)])
    fill_poly(img, [(1366, 828), (1394, 828), (1394, 842), (1366, 842)], hexc("#d8a848"))
    ink(img, [(1366, 828), (1394, 828), (1394, 842), (1366, 842)], 2, INK, (0, 0), closed=True)
    cap = _ell_pts(1380, 806, 20, 24)
    cel_shape(img, cap, hexc("#c8e0ff"), hexc("#7a8ac8"), [(1384, 770), (1410, 770), (1410, 840), (1384, 840)],
              line_w=2.5, seed=29)
    for (x0, y0, x1, y1) in ((1350, 860, 1356, 920), (1404, 852, 1410, 872)):
        fill_poly(img, [(x0, y0), (x0 + 3, y0), (x1, y1), (x1 - 3, y1)], hexc("#ffffff") * 1.4)
    # 브러시 컵
    cup = [(1478, 934), (1474, 870), (1530, 870), (1526, 934)]
    cel_shape(img, cup, hexc("#3a5aa8"), hexc("#1a2a60"), [(1508, 860), (1540, 860), (1540, 940), (1508, 940)],
              line_w=2.5, seed=30)
    tips = []
    for k, (bx, ang) in enumerate(((1488, -10), (1502, 3), (1518, 14))):
        a = math.radians(ang)
        tp = (bx + math.sin(a) * 90, 780)
        ink(img, [(bx, 872), tp], 6, hexc("#2a1a20"), (0, 0))
        tc = (tp[0] + math.sin(a) * 14, tp[1] - 14)
        tips.append(_ell_pts(tc[0], tc[1], 9, 18, rot=a))
        cel_shape(img, tips[-1], hexc("#e8c0a0"), hexc("#9a6a50"),
                  [(tc[0], tc[1] - 30), (tc[0] + 30, tc[1] - 30), (tc[0] + 30, tc[1] + 30), (tc[0], tc[1] + 30)],
                  line_w=2.0, seed=31 + k)
    m = np.zeros((H, W), np.uint8)
    m[884:] = 255
    for poly in [tube, bullet, lid, body, cap, cup] + tips:
        cv2.fillPoly(m, [_fix(poly)], 255, cv2.LINE_AA, _SHIFT)
    cv2.rectangle(m, (1362, 826), (1398, 846), 255, -1)
    for bx, ang in ((1488, -10), (1502, 3), (1518, 14)):
        cv2.line(m, (bx, 872), (int(bx + math.sin(math.radians(ang)) * 90), 780), 255, 8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))

    # 전경: 거울 앞에 앉은 소녀의 뒷모습 (역광 실루엣 + 림라이트)
    fg = np.zeros((H, W, 3), np.float32)
    cx = 262.0
    shoulders = catmull([(-80, 1120), (-60, 1010), (30, 968), (130, 950), (190, 946), (340, 946), (400, 950),
                         (500, 968), (600, 1012), (640, 1120)], 8, closed=True)
    puffL = _ell_pts(40, 1000, 112, 70, rot=math.radians(-24))
    puffR = _ell_pts(490, 1000, 110, 68, rot=math.radians(24))
    frill_base = catmull([(110, 944), (186, 966), (262, 972), (338, 966), (414, 944)], 10)
    frill = []
    for i in range(0, len(frill_base) - 1, 2):
        p, q = frill_base[i], frill_base[min(i + 2, len(frill_base) - 1)]
        for t in np.linspace(0, 1, 6, endpoint=False):
            frill.append(p + (q - p) * t + np.array([0, 9 * math.sin(t * math.pi)]))
    frill = np.array(frill + [frill_base[-1], (414, 910), (110, 910)])
    neck = [(236, 900), (288, 900), (294, 950), (230, 950)]
    # 단발 뒷머리: 종 모양, 밑단은 갈라진 붓끝
    side = [(cx, 646), (cx + 58, 652), (cx + 108, 686), (cx + 140, 744), (cx + 154, 816), (cx + 162, 878),
            (cx + 178, 918)]
    # 밑단: 붓끝처럼 휘어진 머리 다발 (길이/폭 제각각)
    strands = [(150, 26, 22, 1), (118, 34, 30, 1), (84, 30, 38, 1), (50, 36, 34, 1), (14, 34, 44, -1),
               (-22, 36, 36, -1), (-58, 32, 42, -1), (-94, 34, 30, -1), (-128, 30, 26, -1), (-158, 24, 18, -1)]
    tips = []
    for xo, wd, ln, dr in strands:
        yb = 906 + 0.0008 * xo * xo * -1 + 14
        x = cx + xo
        tips += [(x + wd / 2, yb), (x + wd * 0.32, yb + ln * 0.55), (x + dr * wd * 0.18, yb + ln),
                 (x - wd * 0.12, yb + ln * 0.5)]
    side_l = [(2 * cx - x, y) for x, y in side[::-1]]
    hair = np.vstack([catmull(side, 6)[:-1], np.array(tips), catmull(side_l, 6)])
    # 리본 (아이돌의 큰 리본 — 실루엣을 소녀로 읽히게)
    bow_c = np.array([cx + 92, 676])
    loopA = catmull(np.array([(0, 0), (30, -40), (70, -52), (86, -20), (64, 10), (24, 10)]) + bow_c, 5, closed=True)
    loopB = catmull(np.array([(0, 0), (-26, -44), (-62, -60), (-80, -30), (-60, 4), (-22, 8)]) + bow_c, 5, closed=True)
    tailA = np.array([(4, 8), (40, 64), (56, 58), (20, 4)]) + bow_c
    tailB = np.array([(-4, 8), (-22, 70), (-6, 74), (6, 6)]) + bow_c
    knot = _ell_pts(bow_c[0], bow_c[1], 14, 13)
    m8 = np.zeros((H, W), np.uint8)
    for poly in (shoulders, puffL, puffR, frill, neck, hair, loopA, loopB, tailA, tailB, knot):
        cv2.fillPoly(m8, [_fix(poly)], 255, cv2.LINE_AA, _SHIFT)
    fa = m8.astype(np.float32) / 255.0
    D, DS = hexc("#2c0c2a"), hexc("#160618")
    cel_shape(fg, shoulders, D, DS, [(-90, 1016), (262, 1016), (262, 1130), (-90, 1130)], line_w=4, seed=41)
    for k, pf in enumerate((puffL, puffR)):
        cel_shape(fg, pf, D * 1.15, DS, [(pf[:, 0].min() - 9, 1012), (pf[:, 0].max() + 9, 1012),
                                          (pf[:, 0].max() + 9, 1100), (pf[:, 0].min() - 9, 1100)],
                  line_w=3.5, seed=44 + k)
    cel_shape(fg, neck, hexc("#4a2434"), hexc("#2a1220"), [(236, 900), (300, 900), (300, 922), (236, 922)],
              line_w=3, seed=45)
    cel_shape(fg, frill, hexc("#2e2242"), hexc("#1a1228"), [(262, 900), (480, 900), (480, 990), (262, 990)],
              line_w=3, seed=42)
    ring = []
    for i, a in enumerate(np.linspace(math.radians(200), math.radians(340), 15)):
        ring.append((cx + math.cos(a) * 140, 800 + math.sin(a) * 118))
    for i, a in enumerate(np.linspace(math.radians(340), math.radians(200), 15)):
        k = 0.86 if i % 2 == 0 else 0.93
        ring.append((cx + math.cos(a) * 140 * (k + 0.04), 800 + math.sin(a) * 118 * k))
    # 역광: 대부분 그늘색, 오른쪽 가장자리만 초승달 모양으로 빛을 받는다
    lit = catmull([(cx + 30, 648), (cx + 104, 682), (cx + 146, 746), (cx + 160, 830), (cx + 168, 900),
                   (cx + 190, 960), (cx + 120, 900), (cx + 112, 800), (cx + 84, 720), (cx + 30, 672)], 5, closed=True)
    cel_shape(fg, hair, hexc("#130d2a"), hexc("#3a2e78"), lit, line_w=4, seed=43,
              hi=hexc("#4a3a8a"), hi_pts=np.array(ring))
    for k, st in enumerate(([(cx - 40, 660), (cx - 66, 800), (cx - 58, 930)], [(cx + 26, 652), (cx + 34, 800), (cx + 30, 930)],
                            [(cx + 100, 690), (cx + 128, 810), (cx + 112, 924)], [(cx - 106, 690), (cx - 134, 810), (cx - 112, 924)],
                            [(cx - 6, 700), (cx - 10, 840), (cx - 2, 930)])):
        ink(fg, catmull(st, 8), 1.8, hexc("#06040e"), (0.3, 0.3))
    rim = catmull([(cx + 14, 649), (cx + 58, 654), (cx + 106, 688), (cx + 137, 745), (cx + 150, 816),
                   (cx + 158, 880)], 10)
    ink(fg, rim + np.array([-3, 3]), 7.5, hexc("#ffa070") * 2.2, (0.08, 0.25))
    for poly in (loopB, loopA, tailA, tailB):
        cel_shape(fg, poly, hexc("#9a1a46"), hexc("#4a0a26"), [(bow_c[0] - 120, bow_c[1] - 4), (bow_c[0] + 120, bow_c[1] - 4),
                                                               (bow_c[0] + 120, bow_c[1] + 90), (bow_c[0] - 120, bow_c[1] + 90)],
                  line_w=3, seed=47)
    cel_shape(fg, knot, hexc("#b8204e"), None, line_w=3, seed=48)
    # 림라이트 (거울 전구에서 오는 따뜻한 역광)
    ink(fg, catmull(np.array([(30, -40), (70, -52), (86, -20)]) + bow_c, 6) + [-2, 3], 3.5, hexc("#ff9aa8") * 2.0, (0.2, 0.2))
    ink(fg, catmull(np.array([(-26, -44), (-62, -60)]) + bow_c, 6) + [0, 3], 2.5, hexc("#ff9aa8") * 1.6, (0.2, 0.2))
    rim2 = catmull([(430, 948), (510, 962), (590, 1006), (630, 1070)], 8)
    ink(fg, rim2 + np.array([-2, 4]), 4.0, hexc("#ff9a8a") * 1.7, (0.15, 0.3))
    rim3 = catmull([(cx - 14, 649), (cx - 58, 654), (cx - 106, 688), (cx - 137, 745), (cx - 150, 810)], 8)
    ink(fg, rim3 + np.array([2, 3]), 2.6, hexc("#7a8aff") * 1.5, (0.1, 0.4))
    al = np.maximum(m.astype(np.float32) / 255.0, fa)
    r0 = 560
    img[r0:] = img[r0:] * (1 - fa[r0:, :, None]) + fg[r0:] * fa[r0:, :, None]
    return img.astype(np.float32), al.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. 반사 플레이트 (플레이트 좌표 = 기준 좌표 - (PX0, PY0))
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _refl_bg() -> np.ndarray:
    ys = np.linspace(0, 1, PH, dtype=np.float32)
    col = np.stack([np.interp(ys, [0, 0.5, 1], [a, b, c]) for a, b, c in
                    zip(hexc("#1c1038"), hexc("#2a1644"), hexc("#140a26"))], 1)
    img = np.broadcast_to(col[:, None, :], (PH, PW, 3)).copy()
    fx, fy = FACE_C[0] - PX0, FACE_C[1] - PY0
    d = radial_field(fx - 40, fy - 30, PW, PH, sx=1.0, sy=1.15)
    img += (np.exp(-(d / 300) ** 2) * 0.62)[..., None] * PINK
    img += (np.exp(-(d / 600) ** 2) * 0.22)[..., None] * hexc("#7a2a8a")
    xs = np.linspace(0, 1, PW, dtype=np.float32)
    img += (smoothstep(0.55, 1.0, xs) * 0.30)[None, :, None] * BLUE
    # 왼쪽: 살짝 열린 문틈의 형광등 빛
    fill_poly(img, [(80, 30), (252, 52), (252, PH), (80, PH)], hexc("#0e0818"), 0.9)
    ink(img, [(80, 30), (252, 52), (252, PH)], 3, INK, (0, 0))
    ink(img, [(96, 60), (236, 76), (236, PH)], 2, hexc("#3a2a50"), (0, 0))
    fill_ellipse(img, 214, 460, 9, 9, hexc("#c8a060"))
    fill_ellipse(img, 211, 457, 3, 3, hexc("#ffffff") * 1.5)
    slit = np.zeros((PH, PW), np.float32)
    fill_poly(slit[..., None], [(254, 54), (262, 55), (262, PH), (254, PH)], (1.0,))
    img += _glow2d(slit, (3, 14, 50), (0.8, 0.5, 0.45))[..., None] * FLUO * 1.2
    img += slit[..., None] * FLUO * 1.8
    fl = np.zeros((PH, PW), np.float32)
    fill_poly(fl[..., None], [(262, 760), (262, PH), (420, PH)], (0.35,))
    img += blur(fl, 6)[..., None] * FLUO
    # 오른쪽: 반대편 벽에 걸린 의상이 비침
    ink(img, [(1080, 118), (1090, 124), (1088, 136), (1080, 142)], 3, INK, (0, 0))
    ink(img, [(1004, 190), (1080, 146), (1156, 190)], 5, hexc("#120a1c"), (0.05, 0.05))
    dress = _hanging_dress(1080, 172, 1.0)
    cel_shape(img, dress, hexc("#26245a"), hexc("#141236"), [(1086, 160), (1220, 160), (1220, 700), (1120, 700),
                                                            (1100, 420), (1094, 300)], line_w=3.0, seed=51,
              hi=hexc("#3a4a90"), hi_pts=[(960, 170), (1030, 170), (1000, 330), (940, 330)])
    # 반대편 거울 전구의 보케
    rng = np.random.default_rng(52)
    bok = np.zeros((PH, PW, 3), np.float32)
    for i in range(9):
        x = 120 + i * 130 + rng.uniform(-30, 30)
        y = 70 + rng.uniform(-14, 14) + (i % 2) * 18
        r = rng.uniform(26, 42)
        fill_ellipse(bok, x, y, r, r, hexc("#ffb27a") * rng.uniform(0.22, 0.4))
        ink(bok, _ell_pts(x, y, r - 2, r - 2), 2.5, hexc("#ffd0a0") * 0.45, (0, 0), closed=True)
    img += blur(bok, 1.5)
    painted_texture(img, 0.08, seed=401, scale=2.5, stretch=4.0, angle=0.6, ox=610, oy=270)
    return img.astype(np.float32)


# 흉상 로컬 좌표 (얼굴 중심 기준, y 아래로)
HAIR_BACK = [(0, -214), (66, -207), (118, -180), (152, -134), (166, -70), (170, 0), (168, 62), (162, 108),
             (156, 140), (146, 160), (124, 166), (108, 152), (60, 150), (0, 152), (-60, 150), (-108, 152),
             (-124, 166), (-146, 160), (-156, 140), (-162, 108), (-168, 62), (-170, 0), (-166, -70),
             (-152, -134), (-118, -180), (-66, -207)]
HAIR_BACK_SH = [(36, -230), (200, -230), (200, 200), (60, 200), (70, 60), (56, -100)]
HAIR_INNER_SH = [(-200, 96), (-128, 84), (-110, 130), (-112, 180), (-200, 180)]
FACE = [(-96, -100), (-101, -40), (-99, 15), (-92, 58), (-78, 94), (-54, 124), (-28, 145), (-10, 154), (0, 156),
        (10, 154), (28, 145), (54, 124), (78, 94), (92, 58), (99, 15), (101, -40), (96, -100), (56, -128), (0, -136),
        (-56, -128)]
FACE_SH = [(30, -150), (54, -70), (64, 4), (60, 62), (42, 112), (12, 160), (150, 175), (150, -150)]
NECK = [(-36, 90), (-40, 230), (40, 230), (36, 90)]
NECK_SH = [(-60, 80), (60, 80), (60, 240), (26, 240), (24, 196), (0, 182), (-28, 176), (-60, 172)]
BODY = [(-44, 222), (-100, 236), (-176, 258), (-236, 296), (-268, 350), (-286, 430), (-296, 560), (296, 560),
        (286, 430), (268, 350), (236, 296), (176, 258), (100, 236), (44, 222)]
BODY_SH = [(70, 210), (320, 210), (320, 570), (140, 570), (110, 420), (90, 300)]
BANG_TIPS = [(150, -60), (128, -46), (116, -36), (102, -66), (86, -28), (70, -58), (52, -22), (34, -54), (14, -18),
             (-6, -50), (-24, -22), (-44, -58), (-62, -26), (-82, -62), (-100, -32), (-116, -58), (-128, -44),
             (-150, -60)]
CROWN = [(-168, -96), (-152, -150), (-112, -190), (-52, -212), (0, -216), (52, -212), (112, -190), (152, -150),
         (168, -96)]
LOCK_L = [(-104, -98), (-114, -20), (-110, 50), (-102, 110), (-94, 150), (-84, 176), (-80, 124), (-82, 62),
          (-86, 0), (-90, -60)]
EYE_W = [(-18, 10), (-24, -2), (-34, -11), (-48, -15), (-62, -12), (-73, -3), (-79, 6), (-70, 15), (-56, 20),
         (-40, 21), (-27, 17)]


def _mirror_x(pts):
    return [(-x, y) for x, y in pts]


VARIANTS = ("neutral", "smile", "blank", "stare", "closed")


def _draw_bust(img, variant="neutral", seed=0, boil=0.0, gaze=(0.0, 0.0), breath=0.0, tilt=0.0, body=True,
               features=True):
    """오리지널 아이돌 흉상 셀 (플레이트 좌표 img 에 그림).

    이목구비(눈/코/입)는 머리카락 위에 그린다 — 애니의 '앞머리 위로 비치는 눈' 관례.
    body/features 를 나눠 그릴 수 있어 표정 변형 플레이트는 몸통을 공유한다."""
    fx, fy = FACE_C[0] - PX0, FACE_C[1] - PY0
    ca, sa = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))

    def T(pts, dy=0.0):
        P = np.asarray(pts, np.float64).reshape(-1, 2)
        x = P[:, 0] * ca - P[:, 1] * sa
        y = P[:, 0] * sa + P[:, 1] * ca + dy
        return np.stack([fx + FS * x, fy + FS * y], 1)

    def C(pts, n=8, closed=True):
        return catmull(np.asarray(pts, np.float64), n, closed=closed)

    def J(pts, k):
        return jitter(pts, boil, sd + k) if boil else pts

    sd = seed * 31
    bd = breath * 3.0 + NB
    LW = 3.4

    if body:
        # 뒷머리
        hb = T(C(HAIR_BACK))
        cel_shape(img, hb, C_HAIR, C_HAIR_SH, T(HAIR_BACK_SH), line_w=LW + 0.6, boil=boil, seed=sd + 1)
        _clip_fill(img, J(hb, 1), T(HAIR_INNER_SH), C_HAIR_SH)
        # 몸통 / 드레스
        cel_shape(img, T(BODY, bd), C_DRESS, C_DRESS_SH, T(BODY_SH, bd), line_w=LW, boil=boil, seed=sd + 2)
        # 퍼프 소매 (어깨선을 따라 기울어진 납작한 부풂)
        for sgn in (-1, 1):
            cx, cy = sgn * 236, 292 + bd
            rot = math.radians(sgn * 30)
            puff = []
            for a in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                rx, ry = 56, 36 + 4 * math.sin(a * 4) ** 2
                x, y = math.cos(a) * rx, math.sin(a) * ry
                puff.append((cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot)))
            if sgn > 0:
                shp = [(cx - 20, cy - 80), (cx + 90, cy - 80), (cx + 90, cy + 80), (cx - 40, cy + 80)]
            else:
                shp = [(cx - 90, cy + 8), (cx + 90, cy + 30), (cx + 90, cy + 80), (cx - 90, cy + 80)]
            cel_shape(img, T(puff), C_DRESS, C_DRESS_SH, T(shp), line_w=LW, boil=boil, seed=sd + 3 + sgn,
                      hi=mix(C_DRESS, np.ones(3), 0.5),
                      hi_pts=T(C([(cx - 30 * sgn - 10, cy - 18 - 10 * sgn), (cx - 4, cy - 30), (cx + 22 * sgn, cy - 24 + 12 * sgn),
                                  (cx - 2, cy - 22)], 4)))
            for k in (-1, 1):
                bx = cx + k * 18
                ink(img, J(T(C([(bx, cy - 30 + sgn * k * 10), (bx - 3 * sgn, cy - 18 + sgn * k * 10)], 4, False)), 9 + k),
                    1.6, INK, (0.3, 0.6))
            # 어깨 림라이트: 소매 윗면 바깥쪽을 따라 (왼쪽 핑크 / 오른쪽 블루)
            a0, a1 = (math.pi * 0.98, math.pi * 1.62) if sgn < 0 else (math.pi * 1.38, math.pi * 2.02)
            arc = []
            for a in np.linspace(a0, a1, 12):
                x, y = math.cos(a) * 51, math.sin(a) * 32
                arc.append((cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot)))
            ink(img, J(T(arc), 93 + sgn), 3.4, PINK * 2.0 if sgn < 0 else RIM_BLUE * 1.8, (0.25, 0.25))
        # 목
        cel_shape(img, T(NECK), C_SKIN, C_SKIN_SH, T(NECK_SH), line_w=LW - 0.6, boil=boil, seed=sd + 4)
        ink(img, T([(-38, 196 + NB * 0.5), (0, 202 + NB * 0.5), (38, 196 + NB * 0.5)]), 7, hexc("#1a0a20"), (0, 0))
        cel_shape(img, T(_star_pts(0, 212 + NB * 0.5, 9, 0)), GOLD * 1.3, None, line_w=1.6, seed=sd + 5)

        # 프릴 칼라 (2겹, 주름 셀)
        for layer, (sc, col, sh) in enumerate(((1.16, hexc("#f4c6e4"), hexc("#a080c0")), (1.0, C_FRILL, C_FRILL_SH))):
            outer = catmull(np.array([(-154, 238), (-104, 282), (0, 310), (104, 282), (154, 238)]) * [sc, 1]
                            + [0, (sc - 1) * 90], 12)
            inner = catmull(np.array([(-46, 222), (0, 250), (46, 222)]), 10)
            frill, segs = [], []
            nb = 13
            idx = np.linspace(0, len(outer) - 1, nb + 1)
            for i in range(nb):
                p = outer[int(idx[i])]
                q = outer[int(idx[i + 1])] if i < nb - 1 else outer[-1]
                dvec = q - p
                nrm = np.array([dvec[1], -dvec[0]]) / (np.hypot(*dvec) + 1e-6)
                if nrm[1] < 0:
                    nrm = -nrm
                for t in np.linspace(0, 1, 6, endpoint=False):
                    frill.append(p + dvec * t + nrm * math.sin(t * math.pi) * 13)
                segs.append((p, q))
            frill.append(outer[-1])
            frill += list(inner[::-1])
            frill = T(np.array(frill), bd)
            cel_shape(img, frill, col, None, line=None, line_w=0, boil=boil, seed=sd + 6 + layer)
            n_in = len(inner)
            for i, (p, q) in enumerate(segs):
                a = inner[int(round(i / max(1, nb - 1) * (n_in - 1)))]
                if i % 2 == 1 or i > nb * 0.62:
                    fill_poly(img, J(T([a, p, q], bd), 60 + i), sh)
                ink(img, J(T([a, p], bd), 80 + i), 1.4, INK, (0.4, 0.1))
            ink(img, J(frill, 6 + layer), LW - 0.9, INK, (0, 0), closed=True)
        # 리본
        for sgn in (-1, 1):
            loop = [(0, 258), (sgn * 20, 236), (sgn * 52, 228), (sgn * 64, 250), (sgn * 54, 278), (sgn * 22, 280)]
            cel_shape(img, T(loop, bd), C_BOW, C_BOW_SH, T([(0, 262), (sgn * 70, 256), (sgn * 70, 290), (0, 290)], bd),
                      line_w=LW - 0.6, boil=boil, seed=sd + 12 + sgn)
            tail = [(sgn * 6, 266), (sgn * 26, 318), (sgn * 12, 330), (sgn * 2, 300)]
            cel_shape(img, T(tail, bd), C_BOW, C_BOW_SH,
                      T([(sgn * 14, 300), (sgn * 40, 300), (sgn * 40, 340), (sgn * 14, 340)], bd),
                      line_w=LW - 0.8, boil=boil, seed=sd + 15 + sgn)
        cel_shape(img, T(_ell_pts(0, 258, 12, 14), bd), C_BOW, C_BOW_SH, T([(0, 240), (20, 240), (20, 280), (0, 280)], bd),
                  line_w=LW - 0.6, boil=boil, seed=sd + 18)
        fill_ellipse(img, *T([(-4, 252)], bd)[0], 3.5, 3, hexc("#ff9aa8"))

        # 귀걸이 (단발 아래로 살짝)
        for sgn in (-1, 1):
            ink(img, T([(sgn * 126, 150), (sgn * 130, 174)]), 1.6, INK, (0, 0))
            cel_shape(img, T(_star_pts(sgn * 131, 186, 12, sgn * 10)), GOLD * 1.5, hexc("#c08a30"),
                      T([(sgn * 131, 170), (sgn * 150, 170), (sgn * 150, 200), (sgn * 131, 200)]), line_w=1.8, seed=sd + 19)

        # 얼굴
        face = T(C(FACE, 6))
        cel_shape(img, face, C_SKIN, C_SKIN_SH, T(FACE_SH), line_w=LW, boil=boil, seed=sd + 20)
        cast = [(x + 7, y + 18) for x, y in BANG_TIPS[2:-2]] + [(130, -150), (-130, -150)]
        _clip_fill(img, J(face, 20), J(T(cast), 21), C_SKIN_SH, 0.85)
        face_m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(face_m, [_fix(J(face, 20))], 255, cv2.LINE_AA, _SHIFT)
        # 볼터치: 에어브러시 + 사선
        for sgn in (-1, 1):
            cxy = T([(sgn * 60, 52)])[0]
            x0, y0 = int(cxy[0] - 70), int(cxy[1] - 50)
            g = np.zeros((100, 140), np.float32)
            fill_ellipse(g[..., None], cxy[0] - x0, cxy[1] - y0, 28 * FS, 13 * FS, (1.0,))
            g = blur(g, 7) * (face_m[y0:y0 + 100, x0:x0 + 140].astype(np.float32) / 255.0)
            img[y0:y0 + 100, x0:x0 + 140] += g[..., None] * (hexc("#ff4f8a") - C_SKIN) * (0.5 if sgn < 0 else 0.3)
            for k in range(3):
                x = sgn * 60 + (k - 1) * 9
                ink(img, J(T([(x + 4, 44), (x - 4, 58)]), 22 + k), 2.0, hexc("#e0407a"), (0.3, 0.3))
        # 옆머리
        for sgn in (-1, 1):
            lk = T(C(LOCK_L if sgn < 0 else _mirror_x(LOCK_L), 5))
            cel_shape(img, lk, C_HAIR, C_HAIR_SH if sgn > 0 else None, lk if sgn > 0 else None,
                      line_w=LW - 0.4, boil=boil, seed=sd + 60 + sgn)
        # 앞머리 + 엔젤링
        bangs = T(np.vstack([C(CROWN, 6, False), np.array(BANG_TIPS)]))
        ring_top, ring_bot = [], []
        for i, a in enumerate(np.linspace(math.radians(206), math.radians(334), 17)):
            ring_top.append((math.cos(a) * 150, -36 + math.sin(a) * 152))
            k = 0.84 if i % 2 == 0 else 0.92
            ring_bot.append((math.cos(a) * 150 * (k + 0.05), -36 + math.sin(a) * 152 * k))
        ring = np.array(ring_top + ring_bot[::-1])
        bang_sh = [(28, -230), (200, -230), (200, -30), (64, -24), (60, -60), (48, -110)]
        cel_shape(img, bangs, C_HAIR, C_HAIR_SH, T(bang_sh), line_w=LW + 0.4, boil=boil, seed=sd + 63,
                  hi=C_HAIR_HI, hi_pts=T(ring))
        _clip_fill(img, J(T(ring), 70), T([(40, -240), (200, -240), (200, -100), (40, -100)]), hexc("#6a7ae8"))
        for k, strand in enumerate(([(-60, -196), (-80, -120), (-92, -70)], [(10, -206), (6, -130), (2, -60)],
                                    [(80, -190), (100, -120), (110, -70)], [(-150, -60), (-160, 40), (-158, 120)],
                                    [(150, -60), (162, 40), (160, 120)], [(-30, -200), (-40, -120), (-44, -64)])):
            ink(img, J(T(C(strand, 6, False)), 72 + k), 1.6, INK, (0.4, 0.4))
        cel_shape(img, T(_star_pts(-96, -118, 16, -12)), GOLD * 1.45, hexc("#c08a30"),
                  T([(-96, -140), (-70, -140), (-70, -96), (-96, -96)]), line_w=2.2, boil=boil, seed=sd + 78)

        # 림라이트 (투과광 선): 왼쪽 핑크, 오른쪽 블루
        hbp = np.array(HAIR_BACK, np.float64)
        left = C(hbp[[0, 25, 24, 23, 22, 21, 20, 19]], 8, False)
        ink(img, J(T(left + [4, 2]), 90), 4.0, PINK * 2.4, (0.1, 0.3))
        right = C(hbp[[1, 2, 3, 4, 5, 6, 7]], 8, False)
        ink(img, J(T(right + [-4, 2]), 91), 3.6, RIM_BLUE * 2.2, (0.15, 0.25))
        ink(img, J(T(C([(98, -40), (100, 15), (92, 58), (78, 94)], 6, False) + [-3, 0]), 92), 2.6,
            RIM_BLUE * 1.8, (0.2, 0.3))
    if features:
        # 눈
        gx, gy = gaze
        for sgn in (-1, 1):
            S = (lambda P: _mirror_x(P)) if sgn > 0 else (lambda P: P)
            if variant == "blank":
                continue
            if variant == "closed":
                lid = S([(-16, 9), (-28, 15), (-48, 18), (-66, 14), (-80, 5)])
                ink(img, J(T(C(lid, 6, False)), 30 + sgn), 5.5, INK, (0.15, 0.1))
                for k in range(3):
                    bx = -54 - k * 9
                    ink(img, J(T(S([(bx, 16), (bx - 5, 25)])), 33 + k), 2.0, INK, (0, 0.5))
                continue
            vs = 1.22 if variant == "stare" else 1.0
            ew = [(x, 4 + (y - 4) * vs) for x, y in EYE_W]
            ew_pts = J(T(C(S(ew), 5)), 34 + sgn)
            x0, y0 = int(ew_pts[:, 0].min() - 6), int(ew_pts[:, 1].min() - 6)
            x1, y1 = int(ew_pts[:, 0].max() + 7), int(ew_pts[:, 1].max() + 7)
            lay = img[y0:y1, x0:x1].copy()
            em = np.zeros(lay.shape[:2], np.uint8)
            cv2.fillPoly(em, [_fix(ew_pts, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
            lay[:] = C_EYEW if sgn < 0 else C_EYEW_SH
            icx, icy = sgn * 47 + gx, 5 + gy
            irx, iry = (9.5, 11.5) if variant == "stare" else (15.0, 18.5)
            c0 = T([(icx, icy)])[0] - (x0, y0)
            fill_ellipse(lay, c0[0], c0[1], irx * FS, iry * FS, C_IRIS)
            c1 = T([(icx, icy + 6)])[0] - (x0, y0)
            fill_ellipse(lay, c1[0], c1[1], irx * FS * 0.8, iry * FS * 0.58, C_IRIS_HI)
            pr = 0.38 if variant != "stare" else 0.3
            fill_ellipse(lay, c0[0], c0[1] + 1, irx * FS * pr * 1.1, iry * FS * pr, C_PUPIL)
            lidsh = T(C(S([(-14, -30), (-80, -30), (-80, -6), (-62, -3), (-48, -6), (-30, -2), (-14, 4)]), 4)) - (x0, y0)
            fill_poly(lay, lidsh, C_PUPIL, 0.45)
            if variant != "stare":
                h0 = T([(icx - 5.5, icy - 7)])[0] - (x0, y0)
                fill_ellipse(lay, h0[0], h0[1], 5.2 * FS, 6.5 * FS, np.array([1.6, 1.6, 1.7]))
                h1 = T([(icx + 6, icy + 8)])[0] - (x0, y0)
                fill_ellipse(lay, h1[0], h1[1], 2.4 * FS, 2.4 * FS, np.array([1.3, 1.3, 1.4]))
                h2 = T([(icx + 4, icy - 9)])[0] - (x0, y0)
                fill_ellipse(lay, h2[0], h2[1], 1.6 * FS, 1.6 * FS, PINK * 1.6)
            else:
                h0 = T([(icx - 3, icy - 4)])[0] - (x0, y0)
                fill_ellipse(lay, h0[0], h0[1], 2.0 * FS, 2.0 * FS, np.array([1.4, 1.4, 1.5]))
            emf = (em.astype(np.float32) / 255.0)[..., None]
            reg = img[y0:y1, x0:x1]
            reg[:] = reg * (1 - emf) + lay * emf
            up = S([(-15, 10), (-22, -2), (-33, -13 * vs), (-48, -18 * vs), (-63, -15 * vs), (-75, -5), (-85, 3)])
            ink(img, J(T(C(up, 6, False)), 40 + sgn), 7.0, INK, (0.05, 0.12),
                pressure=np.array([0.45, 0.6, 0.8, 1.0, 1.0, 0.9, 0.7]))
            for k, (a, b) in enumerate((((-76, -4), (-92, -12)), ((-82, 1), (-97, -1)), ((-68, -11), (-80, -22)))):
                ink(img, J(T(S([a, b])), 42 + k), 3.0, INK, (0, 0.6))
            ink(img, J(T(C(S([(-72, 13), (-56, 21 * vs), (-40, 22 * vs)]), 5, False)), 44), 2.2, INK, (0.3, 0.4))
            ink(img, J(T(C(S([(-28, -24), (-46, -29), (-66, -25)]), 5, False)), 45), 1.7, INK, (0.3, 0.3))

        if variant != "blank":
            fill_poly(img, J(T([(4, 48), (15, 66), (3, 72)]), 50), C_SKIN_SH)
            ink(img, J(T([(8, 54), (12, 66), (5, 71)]), 51), 2.0, INK, (0.2, 0.4))
            if variant == "smile":
                mouth = [(-27, 94), (-13, 102), (0, 104), (13, 102), (27, 94), (16, 110), (0, 115), (-16, 110)]
                fill_poly(img, J(T(C(mouth, 4)), 52), hexc("#3a0816"))
                fill_poly(img, J(T(C([(-19, 98), (0, 104), (19, 98), (0, 107.5)], 4)), 53), hexc("#fff4f4"))
                ink(img, J(T(C([(-30, 91), (-14, 101), (0, 104), (14, 101), (30, 91)], 5, False)), 54),
                    3.0, INK, (0.15, 0.15))
                ink(img, J(T(C([(-14, 111), (0, 116), (14, 111)], 5, False)), 55), 3.2, LIP, (0.3, 0.3))
                for sgn in (-1, 1):
                    ink(img, T([(sgn * 31, 86), (sgn * 34, 94)]), 1.8, INK, (0.2, 0.2))
            else:
                ink(img, J(T(C([(-15, 100), (-4, 98.5), (4, 98.5), (15, 100)], 5, False)), 54), 2.6, INK, (0.2, 0.2))
                fill_poly(img, J(T(C([(-10, 103), (0, 101.5), (10, 103), (0, 109)], 4)), 55), LIP)
    return img


@lru_cache(maxsize=1)
def _light_filter():
    """핑크(왼쪽)/블루(오른쪽) 조명 젤 + 얼굴 스포트 (플레이트 좌표, 곱하기)."""
    xs = np.linspace(0, 1, PW, dtype=np.float32)
    t = smoothstep(0.32, 0.80, xs)[None, :, None]
    left = np.array([1.06, 0.92, 1.00], np.float32)
    right = np.array([0.84, 0.90, 1.14], np.float32)
    g = left * (1 - t) + right * t
    fx, fy = FACE_C[0] - PX0, FACE_C[1] - PY0
    d = radial_field(fx, fy + 40, PW, PH, sx=1.25, sy=1.0)
    spot = (0.74 + 0.30 * np.exp(-(d / 520) ** 2))[..., None]
    return (g * spot).astype(np.float32)


@lru_cache(maxsize=1)
def _sheen():
    """유리 광택: 사선 띠(에어브러시) + 가는 선. 플레이트 좌표 가산 마스크."""
    m = np.zeros((PH, PW), np.float32)
    for (x0, wd, a) in ((170, 150, 0.16), (350, 46, 0.18), (930, 190, 0.08), (1130, 34, 0.20)):
        fill_poly(m[..., None], [(x0, 0), (x0 + wd, 0), (x0 + wd - 560, PH), (x0 - 560, PH)], (a,))
    m = blur(m, 9)
    for x0, a in ((316, 0.30), (1176, 0.22)):
        fill_poly(m[..., None], [(x0, 0), (x0 + 3, 0), (x0 + 3 - 560, PH), (x0 - 560, PH)], (a,), mode="add")
    return m


def _plate(variant, seed=0, boil=0.0, gaze=(0.0, 0.0), breath=0.0, tilt=0.0):
    img = _refl_bg().copy()
    _draw_bust(img, variant, seed, boil, gaze, breath, tilt)
    img *= _light_filter()
    img += _sheen()[..., None] * np.array([0.95, 0.9, 1.05], np.float32)
    return img


@lru_cache(maxsize=1)
def _bust_base():
    img = _refl_bg().copy()
    _draw_bust(img, "blank", seed=7, gaze=(1.5, 0.0), features=False)
    return img


@lru_cache(maxsize=5)
def _plate_static(variant):
    img = _bust_base().copy()
    _draw_bust(img, variant, seed=7, gaze=(1.5, 0.0), body=False)
    img *= _light_filter()
    img += _sheen()[..., None] * np.array([0.95, 0.9, 1.05], np.float32)
    return img


@lru_cache(maxsize=1)
def _glass_mask():
    m = np.zeros((PH, PW), np.uint8)
    cv2.fillPoly(m, [_fix(_rr(*GLASS, 10) - [PX0, PY0])], 255, cv2.LINE_AA, _SHIFT)
    return m.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 3. 파편 (보로노이, 충격점 중심 극좌표 시드 → 거미줄형 균열)
# ---------------------------------------------------------------------------
TINTS = {
    "none": (np.array([1.0, 1.0, 1.0]), np.zeros(3)),
    "pink": (np.array([1.30, 0.70, 1.00]), np.array([0.05, 0.0, 0.03])),
    "blue": (np.array([0.55, 0.80, 1.45]), np.array([0.0, 0.02, 0.06])),
    "fluo": (np.array([0.72, 1.30, 0.86]), np.array([0.0, 0.03, 0.0])),
    "red": (np.array([1.55, 0.42, 0.46]), np.array([0.06, 0.0, 0.0])),
    "neg": (np.array([0.9, 1.05, 1.25]), np.zeros(3)),
    "mono": (np.array([0.85, 0.9, 1.1]), np.zeros(3)),
}


def _clip_rect(poly, x0, y0, x1, y1):
    for (a, b, c) in ((-1, 0, -x0), (1, 0, x1), (0, -1, -y0), (0, 1, y1)):
        poly = _halfplane(poly, a, b, c)
        if len(poly) < 3:
            return poly
    return poly


@lru_cache(maxsize=1)
def _web():
    """거미줄형 균열: 충격점에서 뻗는 방사선 × 엇갈린 동심 고리."""
    rng = np.random.default_rng(4242)
    I = np.array(IMPACT)
    NR = 17
    radii = [0, 34, 84, 152, 240, 350, 490, 660, 860, 1100]
    th = np.array([2 * np.pi * (k + rng.uniform(-0.3, 0.3)) / NR for k in range(NR)])
    pts = np.zeros((NR, len(radii), 2))
    for k in range(NR):
        bend = 0.0
        for j, r in enumerate(radii):
            if j == 0:
                pts[k, j] = I
                continue
            bend += rng.normal(0, 0.045)
            rr = r * rng.uniform(0.88, 1.12)
            a = th[k] + bend
            pts[k, j] = I + np.array([math.cos(a) * rr, math.sin(a) * rr * 0.94])
    gx0, gy0, gx1, gy1 = GLASS
    cells = []
    for j in range(len(radii) - 1):
        k = int(rng.integers(0, NR))
        start = k
        covered = 0
        while covered < NR:
            if j == 0:
                L = 1
            elif j < 3:
                L = int(rng.choice([1, 1, 2]))
            else:
                L = int(rng.choice([1, 2, 2, 3]))
            L = min(L, NR - covered)
            ks = [(k + i) % NR for i in range(L + 1)]
            inner = [I] if j == 0 else [pts[kk, j] for kk in ks]
            outer = [pts[kk, j + 1] for kk in reversed(ks)]
            poly = _clip_rect(np.array(inner + outer), gx0 - 2, gy0 - 2, gx1 + 2, gy1 + 2)
            if len(poly) >= 3 and _area(poly) > 40:
                cells.append(poly)
            k = (k + L) % NR
            covered += L
    return cells


@lru_cache(maxsize=1)
def _crack_edges():
    """유리 테두리 위에 있지 않은 균열 선분들 (중복 제거)."""
    gx0, gy0, gx1, gy1 = GLASS
    seen = set()
    out = []
    for poly in _web():
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            on_b = ((abs(p[0] - q[0]) < 0.5 and (abs(p[0] - gx0 + 2) < 0.5 or abs(p[0] - gx1 - 2) < 0.5)) or
                    (abs(p[1] - q[1]) < 0.5 and (abs(p[1] - gy0 + 2) < 0.5 or abs(p[1] - gy1 - 2) < 0.5)))
            if on_b or math.hypot(*(q - p)) < 1.0:
                continue
            key = tuple(sorted((tuple(np.round(p, 1)), tuple(np.round(q, 1)))))
            if key in seen:
                continue
            seen.add(key)
            out.append((p, q))
    return out


@lru_cache(maxsize=1)
def _shards():
    I = np.array(IMPACT)
    gx0, gy0, gx1, gy1 = GLASS
    out = _web()
    shards = []
    var_cycle = ["blank", "stare", "neutral", "closed", "smile", "blank", "stare"]
    tint_cycle = ["none", "pink", "blue", "none", "fluo", "red", "mono", "neg", "none", "blue"]
    for idx, poly in enumerate(out):
        c = _centroid(poly)
        dv = c - I
        rho = float(np.hypot(*dv))
        dirv = dv / (rho + 1e-6)
        edge = bool(np.any((np.abs(poly[:, 0] - (gx0 - 2)) < 1) | (np.abs(poly[:, 0] - (gx1 + 2)) < 1) |
                           (np.abs(poly[:, 1] - (gy0 - 2)) < 1) | (np.abs(poly[:, 1] - (gy1 + 2)) < 1)))
        r = np.random.default_rng([4243, idx])
        sh = dict(poly=poly, c=c, rho=rho, dir=dirv, area=_area(poly))
        sh["stuck"] = edge and rho > 400 and r.random() < 0.8
        # 균열 단계 (f20–29): 조금씩 어긋남
        sh["crack_off"] = dirv * r.uniform(1.0, 5.0) * (1.2 - rho / 900) + r.normal(0, 1.2, 2)
        sh["crack_rot"] = r.uniform(-1.4, 1.4)
        if rho < 300:
            sh["crack_var"] = var_cycle[idx % len(var_cycle)]
            sh["crack_tint"] = "none"
            sh["crack_cshift"] = r.normal(0, 1, 2) * 10 + dirv * 7
            sh["crack_crot"] = r.uniform(-5, 5)
        else:
            sh["crack_var"] = "smile"
            sh["crack_tint"] = "none" if r.random() < 0.9 else ("blue", "pink")[idx % 2]
            sh["crack_cshift"] = r.normal(0, 1, 2) * 3
            sh["crack_crot"] = r.uniform(-1.2, 1.2)
        # 비행 단계 (f30–)
        near = clamp(1 - rho / 900)
        sh["r_end"] = (70 + 260 * r.random()) * (0.5 + 0.8 * near)
        sh["z_end"] = 0.0 if sh["stuck"] else (0.05 + 0.30 * r.random() * near + 0.05 * near)
        sh["spin"] = r.choice([-1, 1]) * r.uniform(25, 160)
        sh["flip"] = r.choice([-1, 1]) * r.uniform(1.2, 5.0)
        sh["flip_axis"] = r.uniform(0, np.pi)
        sh["glint_at"] = r.uniform(0.3, 1.2) if r.random() < 0.3 else 99.0
        sh["fly_var"] = VARIANTS[int(r.integers(0, 5))]
        sh["fly_tint"] = tint_cycle[int(r.integers(0, len(tint_cycle)))]
        sh["hero"] = False
        sh["cs"] = r.uniform(0.38, 0.62)
        sh["crot"] = r.uniform(-25, 25)
        sh["fall"] = None
        shards.append(sh)
    order = sorted(range(len(shards)), key=lambda i: shards[i]["rho"])
    # 얼굴 위의 두 조각만 다른 색 (균열 단계)
    face_ids = [i for i in order if 90 < shards[i]["rho"] < 260]
    for i, t in zip(face_ids[2::5], ("blue", "fluo", "pink")):
        shards[i]["crack_tint"] = t
    # 얼굴 근처 조각 몇 개는 균열 단계에서 떨어져 나간다
    falls = [i for i in order if 40 < shards[i]["rho"] < 200 and shards[i]["area"] < 9000][:4]
    for k, i in enumerate(falls):
        shards[i]["fall"] = (22, 24, 25, 27)[k]
    # 히어로 파편: 카메라 쪽으로 크게 날아오며 '작은 얼굴 전체'를 담는다
    cand = [i for i in order if 80 < shards[i]["rho"] < 420 and shards[i]["area"] > 5000 and not shards[i]["stuck"]
            and shards[i]["fall"] is None]
    big = max((i for i in cand if shards[i]["rho"] < 300), key=lambda i: shards[i]["area"])
    cand = [big] + [i for i in cand if i != big]
    hero_vars = ["smile", "blank", "stare", "neutral", "closed", "blank", "smile", "stare"]
    hero_tints = ["none", "blue", "pink", "fluo", "neg", "red", "mono", "none"]
    for k, i in enumerate(cand[:8]):
        s = shards[i]
        s["hero"] = True
        s["fly_var"] = hero_vars[k]
        s["fly_tint"] = hero_tints[k]
        s["z_end"] = 0.30 + 0.04 * (k % 4)
        s["r_end"] = 60 + 30 * (k % 3)
        s["flip"] = (0.7 + 0.3 * (k % 3)) * (1 if k % 2 else -1)
        s["spin"] = (12 + 9 * k) * (1 if k % 2 else -1)
        if k == 0:      # 주인공 파편: 가장 크게, 천천히, 웃는 얼굴
            s["z_end"] = 0.52
            s["r_end"] = 70
            s["flip"] = 0.55
            s["spin"] = -14
            s["cs"] = 0.6
            s["crot"] = -8
            s["glint_at"] = 0.42
    return shards


def _apply_tint(reg, tint):
    if tint == "none":
        return reg
    m, a = TINTS[tint]
    if tint == "neg":
        return (1.05 - np.clip(reg, 0, 1.05)) * m.astype(np.float32)
    if tint == "mono":
        L = reg[..., :1] * 0.3 + reg[..., 1:2] * 0.59 + reg[..., 2:3] * 0.11
        return L * m.astype(np.float32)
    return reg * m.astype(np.float32) + a.astype(np.float32)


def _shrink(poly, px):
    c = _centroid(poly)
    v = poly - c
    L = np.hypot(v[:, 0], v[:, 1])[:, None] + 1e-6
    return c + v * np.maximum(0.0, (L - px) / L)


def _draw_shard(comp, emit, poly, A, Cc, plate, tint="none", back=False, glint=0.0, gap=1.4, ink_w=2.2,
                rim=1.0, alpha=1.0, glint_pos=0.5, rim_seed=None, edge_col=True, sil=0.0):
    """파편 하나: A(정지 위치 기준 → 현재 기준 좌표), Cc(플레이트 → 정지 기준 좌표)."""
    P = _shrink(poly, gap)
    dst = P @ A[:, :2].T + A[:, 2]
    h, w = comp.shape[:2]
    x0 = int(max(0, math.floor(dst[:, 0].min()) - 3))
    y0 = int(max(0, math.floor(dst[:, 1].min()) - 3))
    x1 = int(min(w, math.ceil(dst[:, 0].max()) + 4))
    y1 = int(min(h, math.ceil(dst[:, 1].max()) + 4))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    m8 = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m8, [_fix(dst, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    a = (m8.astype(np.float32) / 255.0)[..., None] * alpha
    if back:
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        cc = dst.mean(0)
        side = ((xx - cc[0]) * 0.6 + (yy - cc[1]) * -0.8) > 0
        reg = np.where(side[..., None], C_BACKING * 0.55, C_BACKING).astype(np.float32)
    else:
        A3 = np.vstack([A, [0, 0, 1]])
        C3 = np.vstack([Cc, [0, 0, 1]])
        M = (A3 @ C3)[:2].copy()
        M[:, 2] -= (x0, y0)
        reg = cv2.warpAffine(plate, M.astype(np.float32), (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        reg = _apply_tint(reg, tint)
    if sil > 0:
        reg = reg * (1 - sil * 0.9) + SIL_COL * (sil * 0.9)
    glint = glint * min(1.0, 1.4 / math.sqrt(abs(np.linalg.det(A[:, :2])) + 1e-6))
    if glint > 0.01:
        # 유리 면을 가로지르는 사선 광택 띠 (각도에 따라 미끄러짐)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        cc = dst.mean(0)
        ext = max(8.0, float(np.ptp(dst[:, 0]) + np.ptp(dst[:, 1])) * 0.5)
        u = ((xx - cc[0]) * 0.70 + (yy - cc[1]) * 0.71) / ext - (1.0 - 2.0 * glint_pos)
        band = np.exp(-(u / 0.12) ** 2) + 0.35 * np.exp(-((u + 0.34) / 0.04) ** 2)
        reg = reg + band[..., None] * WHITE_HOT * glint
    region = comp[y0:y1, x0:x1]
    region *= (1 - a)
    region += a * reg
    if glint > 0.01:
        emit[y0:y1, x0:x1] += a * band[..., None] * WHITE_HOT * (glint * 0.25)
    if ink_w > 0:
        ink(comp, dst, ink_w, INK, (0, 0), closed=True, alpha=alpha)
    if rim > 0:
        cc = dst.mean(0)
        sa = float(np.dot(dst[:, 0], np.roll(dst[:, 1], -1)) - np.dot(dst[:, 1], np.roll(dst[:, 0], -1)))
        ori = -1.0 if sa < 0 else 1.0
        n = len(dst)
        for i in range(n):
            p, q = dst[i], dst[(i + 1) % n]
            e = q - p
            L = math.hypot(*e)
            if L < 4:
                continue
            nrm = np.array([e[1], -e[0]]) / L * ori
            lit = float(nrm @ LIGHT)
            if rim_seed is not None and hash01(812, int(rim_seed), i) < 0.5:
                continue
            ins = (cc - (p + q) / 2)
            ins = ins / (np.hypot(*ins) + 1e-6) * (2.0 + 0.4 * ink_w)
            pp, qq = p + ins + e * 0.08, q + ins - e * 0.08
            P1 = (int(pp[0] * 16), int(pp[1] * 16))
            Q1 = (int(qq[0] * 16), int(qq[1] * 16))
            if lit > 0.45:
                col = tuple(float(v) for v in (np.array([2.0, 1.9, 2.4]) * (lit - 0.3) * rim * alpha))
                cv2.line(emit, P1, Q1, col, 2, cv2.LINE_AA, 4)
            elif lit < -0.35 and edge_col:
                # 그늘진 쪽 단면: 유리 두께의 청록빛
                col = tuple(float(v) for v in (np.array([0.30, 0.62, 0.58]) * alpha))
                cv2.line(comp, P1, Q1, col, 2, cv2.LINE_AA, 4)


def _rot2(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _affine_about(c, Lin, t):
    """점 c 를 기준으로 선형변환 Lin 후 c+t 로 이동: x → Lin(x-c) + c + t."""
    A = np.zeros((2, 3))
    A[:, :2] = Lin
    A[:, 2] = c + t - Lin @ c
    return A


def _content(c, shift=(0, 0), rot=0.0, scale=1.0, anchor=None):
    """플레이트 좌표 → 정지 기준 좌표. anchor(기준 좌표)를 c 로 옮기며 축소/회전 (작은 얼굴 복제)."""
    Lin = _rot2(rot) * scale
    src = np.asarray(anchor if anchor is not None else c, np.float64)
    Cc = np.zeros((2, 3))
    Cc[:, :2] = Lin
    # plate p → base: Lin(p + P0 - src) + c + shift
    Cc[:, 2] = Lin @ (P0 - src) + c + np.asarray(shift, np.float64)
    return Cc


def _fly_state(sh, tau):
    """산산조각 단계의 파편 상태: 깊이 z, 변환 A, 내용 Cc, 뒷면 여부, 반짝임 세기/위치."""
    I = np.array(IMPACT)
    c = sh["c"]
    r = sh["r_end"] * ease_out(tau, 3.0)
    z = sh["z_end"] * ease_out(tau, 2.4)
    kz = 1.0 / (1.0 - z)
    pos_rest = c + sh["dir"] * r + np.array([0.0, 140.0 * tau * tau])
    pos = I + (pos_rest - I) * kz
    th = sh["flip"] * tau
    ct = math.cos(th)
    if abs(ct) < 0.06:
        ct = 0.06 * (1 if ct >= 0 else -1)
    ax = sh["flip_axis"]
    Rf = _rot2(math.degrees(ax))
    Lin = kz * _rot2(sh["spin"] * ease_out(tau, 1.5)) @ Rf @ np.diag([ct, 1.0]) @ Rf.T
    A = _affine_about(c, Lin, pos - c)
    back = ct < 0
    dth = abs(th) - sh["glint_at"]
    gl = 0.0 if back else math.exp(-(dth / 0.32) ** 2)
    gp = float(clamp(0.5 + dth / 0.64))
    if sh["hero"]:
        Cc = _content(c, (0, 0), sh["crot"], sh["cs"], anchor=(FACE_C[0], FACE_C[1] + 40))
    else:
        tilt_shift = np.array([math.sin(th) * 60 * math.cos(ax), math.sin(th) * 60 * math.sin(ax)])
        Cc = _content(c, tilt_shift, sh["crack_crot"])
    return z, A, Cc, back, gl, gp


def _crack_reveal(lf):
    return {T_CRACK: 230.0, T_CRACK + 1: 560.0}.get(lf, 5000.0)


def _void(lf, rng):
    """거울 뒤 공간 (플레이트 좌표 크기): 어둠 + TV 노이즈 + 차오르는 빛."""
    img = np.empty((PH, PW, 3), np.float32)
    img[:] = hexc("#0a0614")
    st = rng.random((PH // 3, PW // 6)).astype(np.float32)
    st = cv2.resize(st, (PW, PH), interpolation=cv2.INTER_NEAREST)
    img += (st ** 2)[..., None] * np.array([0.10, 0.10, 0.14], np.float32)
    ix, iy = IMPACT[0] - PX0, IMPACT[1] - PY0
    if lf >= T_SHATTER:
        k = clamp((lf - 46) / (T_WHITE - 46))
        burst = math.exp(-(lf - T_SHATTER) / 3.0)
        inten = 0.45 + 0.9 * burst + 3.4 * ease_in(k, 2.0)
        rad = 110 + 60 * burst + 800 * ease_in(k, 1.6)
    else:
        k = clamp((lf - 23) / 7.0)
        inten = 0.6 * k
        rad = 45 + 60 * k
    if inten > 0:
        d = radial_field(ix, iy, PW, PH)
        core = np.exp(-(d / rad) ** 2)
        img += core[..., None] * (ICE * 1.4 * inten)
        img += (np.exp(-(d / (rad * 0.35)) ** 2))[..., None] * (WHITE_HOT * 0.8 * inten)
    return img


# ---------------------------------------------------------------------------
# 시간 함수
# ---------------------------------------------------------------------------
def _bulb_state(i, lf):
    s = 1.0
    if i == 2 and 4 <= lf <= 6:
        s = (0.0, 0.3, 0.0)[lf - 4]
    if i == 14 and lf in (15, 16):
        s = 0.15 if lf == 15 else 0.6
    if 16 <= lf < T_CRACK:
        s *= 1.25
    pf = POP.get(i)
    if pf is not None and lf >= pf:
        s = 4.5 if lf == pf else (1.2 if lf == pf + 1 else 0.0)
    elif lf >= T_SHATTER:
        s *= 0.85 + 0.3 * hash01(i, lf // 2, 9)
    return s


def _expression(lf):
    if lf // 2 == 4:
        return "closed", (0.0, 0.0)
    if lf < 10:
        return "neutral", (-3.0, 2.0)
    if lf < 14:
        return "neutral", (1.5, 0.5)
    return "smile", (1.5, 0.0)


def _camera(lf):
    s = 1.03 + 0.05 * ease_in_out(clamp(lf / 17.0))
    if 14 <= lf < T_CRACK:
        s += 0.03 * math.exp(-(lf - 14) / 2.5)        # 웃음이 드러나는 순간의 짧은 T.U. 스냅
    dx = dy = rot = 0.0
    if lf >= T_CRACK:
        s += 0.012
        a = 16.0 * math.exp(-(lf - T_CRACK) / 3.0)
        dx += a * (hash01(301, lf) - 0.5) * 2
        dy += a * (hash01(302, lf) - 0.5) * 2
    if lf >= T_SHATTER:
        k = lf - T_SHATTER
        s += 0.07 * math.exp(-k / 3.5) + 0.06 * ease_out(clamp(k / 26.0))
        a = 30.0 * math.exp(-k / 4.0) + 2.5
        dx += a * (hash01(303, lf) - 0.5) * 2
        dy += a * (hash01(304, lf) - 0.5) * 2
        rot += 0.6 * math.exp(-k / 4.0) * (hash01(305, lf) - 0.5) * 2
    return s, dx, dy, rot


# ---------------------------------------------------------------------------
# 효과 레이어
# ---------------------------------------------------------------------------
def _bulbs(comp, lf):
    em = np.zeros((H, W), np.float32)
    for i, (x, y) in enumerate(BULBS):
        s = _bulb_state(i, lf) * (0.86 + 0.24 * hash01(i, 77))      # 손으로 칠한 듯 제각각인 밝기
        if s <= 0:
            continue
        fill_ellipse(comp, x, y, BULB_R, BULB_R, BULB * min(s, 1.6))
        fill_ellipse(comp, x - 5, y - 6, 6, 5, np.array([4.0, 3.6, 3.2]) * min(s, 1.6))
        fill_ellipse(em[..., None], x, y, BULB_R * 1.15, BULB_R * 1.15, (s,))
    g = _glow2d(em, (6, 22, 70), (0.55, 0.45, 0.42))
    comp += g[..., None] * BULB_GLOW * 1.4


def _sparks(emit, lf):
    for i, pf in POP.items():
        k = lf - pf
        if k < 0 or k > 9:
            continue
        x0, y0 = BULBS[i]
        rng = np.random.default_rng([515, i])
        n = 16
        ang = rng.uniform(0, 2 * np.pi, n)
        sp = rng.uniform(10, 34, n)
        kk = (k // 2) * 2 + 1.0          # 2코마
        for j in range(n):
            vx, vy = math.cos(ang[j]) * sp[j], math.sin(ang[j]) * sp[j] - 6
            def P(t):
                return (x0 + vx * t, y0 + vy * t + 1.6 * t * t)
            p1, p0 = P(kk), P(max(0.0, kk - 1.6))
            fade = 1 - kk / 11
            col = np.array([3.0, 1.8, 0.8]) * fade
            ink(emit, [p0, p1], 2.6, col, (0.0, 0.5))


def _glitter(emit, lf):
    """산산조각 직후 흩날리는 유리 가루 (2코마)."""
    if lf < T_SHATTER:
        return
    kk = ((lf - T_SHATTER) // 2) * 2 + 1
    tau = kk / 27.0
    rng = np.random.default_rng(616)
    n = 150
    ang = rng.uniform(0, 2 * np.pi, n)
    sp = rng.uniform(200, 1100, n)
    z = rng.uniform(0.1, 0.8, n)
    I = np.array(IMPACT)
    for j in range(n):
        r = sp[j] * ease_out(tau, 2.0)
        k = 1 / (1 - z[j] * ease_out(tau, 1.6) * 0.9)
        p = I + np.array([math.cos(ang[j]), math.sin(ang[j])]) * r * k + (0, 90 * tau * tau)
        if not (0 <= p[0] < W and 0 <= p[1] < H):
            continue
        tw = hash01(617, j, lf // 2)
        b = 0.4 + 2.2 * tw ** 3
        sz = 1.2 + 2.2 * z[j] * k
        if tw > 0.93:
            _twinkle(emit, p[0], p[1], 10 + 14 * z[j], WHITE_HOT * 0.9, rot=15 * j)
        else:
            fill_ellipse(emit, p[0], p[1], sz, sz, np.array([1.2, 1.3, 1.6]) * b, mode="add")


def _dust(emit, lf):
    rng = np.random.default_rng(4411)
    n = 26
    xs = rng.uniform(380, 1540, n)
    ys = rng.uniform(140, 860, n)
    vx = rng.uniform(-14, 14, n)
    vy = rng.uniform(-24, -6, n)
    ph = rng.uniform(0, 6.28, n)
    t = lf / 24.0
    lay = np.zeros((H, W, 3), np.float32)
    for i in range(n):
        x = xs[i] + vx[i] * t + 6 * math.sin(t * 2 + ph[i])
        y = ys[i] + vy[i] * t
        b = 0.5 + 0.5 * math.sin(t * 5 + ph[i] * 3)
        r = 1.6 + 1.8 * (i % 3) / 2
        fill_ellipse(lay, x, y, r, r, hexc("#ffd0a0") * (0.25 + 0.5 * b))
    emit += blur(lay, 1.2)


def _static_tex(rng, h, w):
    st = rng.random((max(1, h // 2), max(1, w // 5))).astype(np.float32)
    st = cv2.resize(st, (w, h), interpolation=cv2.INTER_NEAREST)
    return st


GLITCH = {12: 1, 15: 2, 19: 3, 20: 3, 24: 2, 27: 1, 32: 3, 33: 2, 36: 1, 39: 2, 44: 4, 45: 3, 49: 2, 50: 2,
          53: 1}


def _glitch(img, lf):
    lvl = GLITCH.get(lf, 0)
    if lvl == 0:
        return img
    rng = np.random.default_rng([1988, 404, lf])
    nb = {1: 1, 2: 3, 3: 5, 4: 7}[lvl]
    for _ in range(nb):
        hgt = int(rng.uniform(8, 30 + 50 * lvl))
        y0 = int(rng.uniform(0, H - hgt))
        if lf < T_SHATTER and y0 + hgt > 300 and y0 < 600:
            # 균열 단계에서는 얼굴을 가리지 않는다
            y0 = int(600 + rng.uniform(0, H - 600 - hgt)) if rng.random() < 0.6 else int(rng.uniform(0, max(1, 300 - hgt)))
        y1 = y0 + hgt
        dx = int(rng.normal(0, 30 + 40 * lvl))
        sp = int(rng.uniform(4, 6 + 8 * lvl))
        band = img[y0:y1].copy()
        img[y0:y1, :, 0] = np.roll(band[..., 0], dx + sp, axis=1)
        img[y0:y1, :, 1] = np.roll(band[..., 1], dx, axis=1)
        img[y0:y1, :, 2] = np.roll(band[..., 2], dx - sp, axis=1)
        if rng.random() < 0.6:
            st = _static_tex(rng, hgt, W)
            a = rng.uniform(0.25, 0.75)
            img[y0:y1] = img[y0:y1] * (1 - a) + (st[..., None] * np.array([1.0, 1.0, 1.1], np.float32)) * a * 1.2
        img[y0:y1:2] *= 0.55
    if lvl >= 3:
        img[::3] *= 0.82                     # 전면 주사선
    if lvl >= 4:
        st = _static_tex(rng, H, W)
        img[:] = img * 0.45 + st[..., None] * 0.75
        roll = int(rng.uniform(60, 140))
        img[:] = np.roll(img, roll, axis=0)
        img[roll - 26:roll] = np.array([0.02, 0.02, 0.03], np.float32)   # 수직 동기 바
    return img


@lru_cache(maxsize=4)
def _sub_mask(size):
    return text_mask("本当の私は、どこ？", size, FONT_JP, stroke=1)


def _subtitle(img, lf):
    if lf < 13 or lf >= T_WHITE:
        return img
    if lf < 21 and lf not in (13, 14, 16, 17):
        return img
    big = lf in (44, 45)
    if lf >= 21 and not (34 <= lf <= 45) and hash01(707, lf) < 0.2:
        return img
    size = 150 if big else 62
    m = _sub_mask(size)
    mh, mw = m.shape
    cx = W / 2 + (hash01(708, lf) - 0.5) * (40 if big else 6)
    cy = (H * 0.5 if big else 968) + (hash01(709, lf) - 0.5) * (30 if big else 4)
    sp = 3 + int(10 * hash01(710, lf) ** 2) + (14 if big else 0) + (6 if lf in GLITCH else 0)
    pad = 40
    x0, y0 = int(cx - mw / 2) - pad, int(cy - mh / 2) - pad
    hh, ww = mh + 2 * pad, mw + 2 * pad
    loc = np.zeros((hh, ww), np.float32)
    loc[pad:pad + mh, pad:pad + mw] = m
    # 글리치 프레임: 글자를 가로로 썰어 어긋나게
    if lf in GLITCH or big:
        rng = np.random.default_rng([711, lf])
        for _ in range(3):
            a0 = int(rng.uniform(pad, pad + mh - 6))
            a1 = a0 + int(rng.uniform(4, max(6, mh // 3)))
            loc[a0:a1] = np.roll(loc[a0:a1], int(rng.normal(0, 18)), axis=1)
    outline = cv2.dilate(loc, np.ones((7, 7), np.uint8))
    outline = cv2.GaussianBlur(outline, (0, 0), 1.2)
    reg = img[y0:y0 + hh, x0:x0 + ww]
    reg *= (1 - outline[..., None] * 0.75)
    gain = 1.4 if not big else 2.0
    for ch, off in ((0, -sp), (1, 0), (2, sp)):
        reg[..., ch] += np.roll(loc, off, axis=1) * gain
    return img


def _negative(img, duo=False):
    x = np.clip(img, 0, 1.2)
    if not duo:
        return ((1.15 - x) * np.array([0.95, 1.0, 1.15], np.float32)).astype(np.float32)
    # 듀오톤 반전: 남보라 잉크 ↔ 투과광 흰색
    L = np.clip(x[..., 0] * 0.3 + x[..., 1] * 0.59 + x[..., 2] * 0.11, 0, 1)
    t = (1.0 - L)[..., None] ** 1.4
    lo = np.array([0.10, 0.03, 0.16], np.float32)
    hi = np.array([1.7, 1.55, 1.8], np.float32)
    return (lo + (hi - lo) * t).astype(np.float32)


# ---------------------------------------------------------------------------
# 렌더
# ---------------------------------------------------------------------------
def render(ctx: Ctx) -> np.ndarray:
    lf = ctx.lframe
    if lf > T_WHITE:
        return np.full((H, W, 3), 1.6, np.float32)
    dwg = ctx.drawing(2)
    comp = _set_back().copy()
    emit = np.zeros((H, W, 3), np.float32)
    fg, fa = _set_front()
    shards = _shards()
    I = np.array(IMPACT)
    rng = ctx.rng(17, 1)

    breath = math.sin(ctx.on(2) * 2.4)
    tilt = 0.6 * math.sin(ctx.on(2) * 1.3)
    late = []      # 전경보다 앞에 그릴 파편
    ghosts = []    # 히어로 파편의 잔상

    if lf < T_SEP:
        # --- 온전한 거울 (금 가는 순간까지) ---
        var, gaze = _expression(lf)
        plate = _plate(var, seed=dwg, boil=0.9, gaze=gaze, breath=breath, tilt=tilt)
        gm = _glass_mask()
        reg = comp[PY0:PY1, PX0:PX1]
        reg *= (1 - gm[..., None])
        reg += plate * gm[..., None]
        if lf >= T_CRACK:
            R = _crack_reveal(lf)
            for ei, (p, q) in enumerate(_crack_edges()):
                mid = (p + q) / 2
                dm = float(np.hypot(*(mid - I)))
                if dm > R:
                    continue
                ink(comp, [p, q], 2.6 if dm < 200 else 2.0, INK, (0.05, 0.05))
                e = q - p
                L = math.hypot(*e) + 1e-6
                nrm = np.array([e[1], -e[0]]) / L
                lit = float(nrm @ LIGHT)
                if lit < 0:
                    nrm, lit = -nrm, -lit
                if lit < 0.3 or hash01(811, ei) < 0.35:
                    continue
                pp, qq = p + nrm * 2.4 + e * 0.1, q + nrm * 2.4 - e * 0.1
                cv2.line(emit, (int(pp[0] * 16), int(pp[1] * 16)), (int(qq[0] * 16), int(qq[1] * 16)),
                         tuple(float(v) for v in np.array([2.4, 2.3, 2.8]) * lit), 2, cv2.LINE_AA, 4)
            fill_ellipse(emit, I[0], I[1], 12, 12, WHITE_HOT * 1.5)
            _twinkle(emit, I[0], I[1], 90, WHITE_HOT * (1.0 if lf == T_CRACK else 0.5), rot=10)
    else:
        gx0, gy0 = GLASS[0], GLASS[1]
        void = _void(lf, rng)
        gm = _glass_mask()
        reg = comp[PY0:PY1, PX0:PX1]
        reg *= (1 - gm[..., None])
        reg += void * gm[..., None]
        if lf < T_SHATTER:
            # --- 어긋난 파편들 ---
            main = _plate("smile", seed=dwg, boil=0.9, gaze=(1.5, 0.0), breath=breath, tilt=tilt)
            leak = clamp((lf - 24) / 5.0)
            fallers = []
            for sh in shards:
                c = sh["c"]
                grow = 0.6 + 0.4 * clamp((lf - T_SEP) / 8.0)
                A = _affine_about(c, _rot2(sh["crack_rot"] * grow), sh["crack_off"] * grow)
                if sh["fall"] is not None and lf >= sh["fall"]:
                    k = ((lf - sh["fall"]) // 2) * 2 + 1
                    A = _affine_about(c, _rot2(sh["crack_rot"] + 7 * k), sh["crack_off"] + np.array([3 * k, 2.4 * k * k]))
                    fallers.append((sh, A))
                    continue
                v = sh["crack_var"]
                plate = main if v == "smile" else _plate_static(v)
                Cc = _content(c, sh["crack_cshift"] * grow, sh["crack_crot"] * grow)
                _draw_shard(comp, emit, sh["poly"], A, Cc, plate, sh["crack_tint"], gap=1.1 + 0.9 * leak,
                            rim=0.8 + 0.6 * leak, ink_w=1.8, rim_seed=sh["c"][0] * 7 + sh["c"][1], edge_col=False)
            # 균열 사이로 새는 빛
            if leak > 0:
                lm = np.zeros((PH, PW), np.float32)
                for p, q in _crack_edges():
                    cv2.line(lm, tuple(int(v * 16) for v in (p - P0)), tuple(int(v * 16) for v in (q - P0)),
                             1.0, 2, cv2.LINE_AA, 4)
                d = radial_field(I[0] - PX0, I[1] - PY0, PW, PH)
                lm *= np.exp(-(d / (140 + 300 * leak)) ** 2) * gm
                g = _glow2d(lm, (2, 8, 26), (0.8, 0.5, 0.35))
                emit[PY0:PY1, PX0:PX1] += (lm[..., None] * 1.5 + g[..., None] * 0.8) * (ICE * 1.8 * leak)
            for sh, A in fallers:
                Cc = _content(sh["c"], sh["crack_cshift"], sh["crack_crot"])
                late.append((sh, A, Cc, _plate_static(sh["crack_var"] if sh["crack_var"] != "smile" else "neutral"),
                             sh["crack_tint"], False, 0.0, 0.5))
        else:
            # --- 산산조각: 회전하며 카메라 쪽으로 (2코마) ---
            fe = T_SHATTER + ((lf - T_SHATTER) // 2) * 2
            tau = clamp((fe - T_SHATTER + 1.2) / 27.0)
            items = []
            for sh in shards:
                c = sh["c"]
                if sh["stuck"]:
                    A = _affine_about(c, _rot2(sh["crack_rot"]), sh["crack_off"])
                    Cc = _content(c, sh["crack_cshift"], sh["crack_crot"])
                    _draw_shard(comp, emit, sh["poly"], A, Cc, _plate_static("smile"), sh["crack_tint"], gap=1.6)
                    continue
                if sh["fall"] is not None:
                    continue
                z, A, Cc, back, gl, gp = _fly_state(sh, tau)
                items.append((z, sh, A, Cc, _plate_static(sh["fly_var"]), sh["fly_tint"], back, gl, gp))
                if sh["hero"] and fe > T_SHATTER:
                    # 残像: 히어로 파편은 지나온 자리에 색이 갈라진 잔상을 남긴다
                    for gi, (dt, gt) in enumerate(((2.0, "blue"), (4.0, "pink"))):
                        tz = clamp(tau - dt / 27.0)
                        z2, A2, C2, b2, _, _ = _fly_state(sh, tz)
                        ghosts.append((z2 - 0.001, sh, A2, C2, _plate_static(sh["fly_var"]), gt, b2,
                                       0.42 - 0.14 * gi))
            items.sort(key=lambda it: it[0])
            for z, sh, A, Cc, plate, tint, back, gl, gp in items:
                if z < 0.03:
                    _draw_shard(comp, emit, sh["poly"], A, Cc, plate, tint, back, gl, gap=1.6, glint_pos=gp)
                else:
                    late.append((sh, A, Cc, plate, tint, back, gl, gp))
    # 전경 세트 (화장대, 소녀) — 파편이 날아오기 전에는 그 앞
    _bulbs(comp, lf)
    # 끝: 거울 뒤의 빛이 방을 삼키고, 전경/파편은 역광 실루엣으로 남는다
    sil = float(clamp((lf - 51) / 5.0))
    if sil > 0:
        wv = {52: 0.15, 53: 0.35, 54: 0.6, 55: 0.82, 56: 0.95, 57: 1.0}[lf]
        comp *= (1 - wv)
        comp += WASH * wv
        fg = fg * (1 - sil * 0.88) + SIL_COL * (sil * 0.88)
    comp *= (1 - fa[..., None])
    comp += fg * fa[..., None]

    for z2, sh, A, Cc, plate, tint, back, al in sorted(ghosts, key=lambda g: g[0]):
        _draw_shard(comp, emit, sh["poly"], A, Cc, plate, tint, back, 0.0, gap=1.4, ink_w=0, rim=0, alpha=al, sil=sil)
    for sh, A, Cc, plate, tint, back, gl, gp in late:
        kz = math.sqrt(abs(np.linalg.det(A[:, :2])))
        _draw_shard(comp, emit, sh["poly"], A, Cc, plate, tint, back, gl, gap=1.4, ink_w=2.2 * min(2.2, kz),
                    rim=1.2, glint_pos=gp, sil=sil)
        if gl > 0.55 and kz < 2.4:
            dst = sh["poly"] @ A[:, :2].T + A[:, 2]
            tip = dst[np.argmax(dst @ LIGHT)]
            _twinkle(emit, tip[0], tip[1], 26 + 36 * gl, WHITE_HOT * (gl * 0.8), rot=20)

    if lf >= T_SHATTER:
        k = clamp((lf - 46) / (T_WHITE - 46))
        rays = radial_lines_mask(I[0], I[1], n=70, seed=900 + (lf // 2), r_in=40, r_jit=1.5,
                                 width=(0.003, 0.016))
        d = radial_field(I[0], I[1])
        rays = blur(rays, 3) * np.exp(-d / (260 + 900 * k))
        emit += rays[..., None] * (ICE * (0.18 + 0.5 * math.exp(-(lf - T_SHATTER) / 3.0) + 2.4 * ease_in(k, 3.0)))
    _sparks(emit, lf)
    _glitter(emit, lf)
    _dust(emit, lf)
    add_glow(comp, emit, (2, 7, 22), (0.7, 0.45, 0.3))
    comp += emit
    if T_SHATTER < lf <= T_SHATTER + 7:
        # 폭발광이 방 전체를 순간적으로 비춤
        fl = math.exp(-(lf - T_SHATTER) / 2.2)
        d = radial_field(I[0], I[1], sx=1.5)
        comp += (np.exp(-(d / 950) ** 2) * (0.6 * fl))[..., None] * ICE

    s, dx, dy, rot = _camera(lf)
    out = transform(comp, dx, dy, s, rot, CAM_C[0], CAM_C[1], border=cv2.BORDER_REPLICATE)

    if lf in (T_SHATTER, T_SHATTER + 1):
        out = zoom_blur(out, *_cam_pt(IMPACT, lf), amount=0.10 if lf == T_SHATTER else 0.05, steps=7)
    if T_SHATTER + 1 <= lf <= T_SHATTER + 4:
        fx_, fy_ = _cam_pt(IMPACT, lf)
        sl = radial_lines_mask(fx_, fy_, n=110, seed=950 + lf, r_in=380, r_jit=0.6, width=(0.002, 0.007))
        out += sl[..., None] * (np.array([1.4, 1.4, 1.6], np.float32) * (1 - (lf - T_SHATTER - 1) / 4.0))

    _glitch(out, lf)
    _subtitle(out, lf)
    if lf in (T_CRACK, T_SHATTER):
        out = _negative(out, duo=(lf == T_SHATTER))
    # 흰 플래시로 빨려 들어감
    if lf >= 55:
        wv = {55: 0.06, 56: 0.18, 57: 0.9}[lf]
        out = out * (1 - wv) + WASH * wv
    return out.astype(np.float32)


def _cam_pt(p, lf):
    s, dx, dy, rot = _camera(lf)
    M = cv2.getRotationMatrix2D(CAM_C, rot, s)
    M[0, 2] += dx
    M[1, 2] += dy
    return float(M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]), float(M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2])
