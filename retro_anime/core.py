"""공통 드로잉 엔진 — 셀 애니메이션 스타일 프로시저럴 그래픽용 프리미티브.

이미지 규약
-----------
* 모든 프레임은 ``float32`` ``(H, W, 3)`` RGB 배열(디스플레이 sRGB, 0..1 기준).
* 1.0 을 넘는 값(HDR)을 허용한다. 네온/투과광(透過光)처럼 빛나는 부분은 1.5~4.0
  정도로 칠하면 post.film() 의 할레이션/블룸이 자연스럽게 번진다.
* 좌표는 픽셀 단위(1920x1080), 원점은 좌상단, y 는 아래로 증가.

장면은 ``render(ctx) -> ndarray`` 순수 함수여야 한다(전역 난수 금지, ctx 시드 사용).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 24

FONT_JP = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
FONT_JP_ALT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
FONT_LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_SHIFT = 4          # cv2 서브픽셀 정밀도 (1/16 px)
_SUB = 1 << _SHIFT


# ---------------------------------------------------------------------------
# 시간 / 컨텍스트
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    frame: int          # 전체 타임라인 기준 프레임 번호
    lframe: int         # 장면 내부 프레임 번호 (0 부터)
    nframes: int        # 장면 길이(프레임)
    seed: int = 1988

    @property
    def t(self) -> float:
        """전체 타임라인 기준 초."""
        return self.frame / FPS

    @property
    def lt(self) -> float:
        """장면 내부 초."""
        return self.lframe / FPS

    @property
    def dur(self) -> float:
        return self.nframes / FPS

    @property
    def u(self) -> float:
        """장면 진행률 0..1."""
        return self.lframe / max(1, self.nframes - 1)

    def on(self, n: int = 2) -> float:
        """'n 코마 촬영'(on n's)으로 양자화된 장면 내부 시간(초).
        셀 캐릭터는 on(2)/on(3), 카메라 팬은 lt(1코마)를 쓰는 식으로 섞는다."""
        return (self.lframe // n) * n / FPS

    def onu(self, n: int = 2) -> float:
        """on(n) 의 진행률 버전."""
        return ((self.lframe // n) * n) / max(1, self.nframes - 1)

    def drawing(self, n: int = 2) -> int:
        """현재 '동화(드로잉)' 번호 — 선 떨림(line boil) 시드로 사용."""
        return self.lframe // n

    def rng(self, salt: int = 0, n: int = 1) -> np.random.Generator:
        """결정적 난수기. n 코마마다 바뀜 (n=0 이면 장면 전체에서 고정)."""
        key = 0 if n == 0 else self.lframe // n
        return np.random.default_rng([self.seed, salt, key])


# ---------------------------------------------------------------------------
# 수학 / 이징
# ---------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return np.clip(x, a, b) if isinstance(x, np.ndarray) else max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def remap(x, a, b):
    """x 를 [a,b] → [0,1] 로 선형 변환 후 클램프."""
    return clamp((x - a) / (b - a + 1e-12))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0 + 1e-12))
    return t * t * (3 - 2 * t)


def ease_in(t, p=2.0):
    return clamp(t) ** p


def ease_out(t, p=2.0):
    return 1 - (1 - clamp(t)) ** p


def ease_in_out(t, p=2.0):
    t = clamp(t)
    if isinstance(t, np.ndarray):
        return np.where(t < 0.5, 0.5 * (2 * t) ** p, 1 - 0.5 * (2 - 2 * t) ** p)
    return 0.5 * (2 * t) ** p if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** p


def ease_out_expo(t):
    t = clamp(t)
    return 1 - 2 ** (-10 * t) if t < 1 else 1.0


def hash01(*keys) -> float:
    """정수 키들 → 결정적 0..1 실수."""
    return float(np.random.default_rng([abs(int(k)) for k in keys]).random())


def wobble(t, seed=0, freqs=(1.3, 2.9, 6.1), amps=(1.0, 0.5, 0.25)):
    """부드러운 1D 노이즈(카메라 흔들림 등)."""
    s = 0.0
    for i, (f, a) in enumerate(zip(freqs, amps)):
        ph = hash01(seed, i) * 6.2831
        s += a * math.sin(t * f * 6.2831 + ph)
    return s / sum(amps)


# ---------------------------------------------------------------------------
# 색
# ---------------------------------------------------------------------------
def hexc(h: str, gain: float = 1.0) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32) * gain


def mix(c1, c2, t):
    return (np.asarray(c1, np.float32) * (1 - t) + np.asarray(c2, np.float32) * t).astype(np.float32)


def hsv(h, s, v) -> np.ndarray:
    """h,s,v ∈ 0..1 (v 는 1 초과 가능)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, clamp(s), 1.0)
    return np.array([r, g, b], np.float32) * v


def luma(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


# ---------------------------------------------------------------------------
# 캔버스 / 그라디언트
# ---------------------------------------------------------------------------
def blank(color=(0, 0, 0), w=W, h=H) -> np.ndarray:
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.asarray(color, np.float32)
    return img


def _ramp(stops, x):
    """stops: [(pos, color), ...] 를 x(배열, 0..1) 에 대해 보간 → (..., 3)."""
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([np.asarray(c, np.float32) for _, c in stops], np.float32)
    out = np.empty(x.shape + (3,), np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(x, pos, cols[:, ch])
    return out


def vgradient(stops, w=W, h=H, y0=0.0, y1=None) -> np.ndarray:
    """세로 그라디언트. stops 의 pos 0..1 은 y0..y1(기본 0..H) 에 대응."""
    y1 = h if y1 is None else y1
    ys = (np.arange(h, dtype=np.float32) + 0.5 - y0) / (y1 - y0)
    col = _ramp(stops, np.clip(ys, 0, 1))          # (h,3)
    return np.broadcast_to(col[:, None, :], (h, w, 3)).copy()


def linear_gradient(stops, angle_deg=90.0, w=W, h=H) -> np.ndarray:
    """임의 각도 그라디언트 (90° = 위→아래)."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    xs = np.arange(w, dtype=np.float32)[None, :] - w / 2
    ys = np.arange(h, dtype=np.float32)[:, None] - h / 2
    proj = xs * dx + ys * dy
    half = abs(w / 2 * dx) + abs(h / 2 * dy)
    return _ramp(stops, np.clip((proj + half) / (2 * half), 0, 1))


def radial_field(cx, cy, w=W, h=H, sx=1.0, sy=1.0) -> np.ndarray:
    """(cx,cy) 로부터의 거리 필드 (픽셀). sx/sy 로 타원형."""
    xs = (np.arange(w, dtype=np.float32)[None, :] - cx) / sx
    ys = (np.arange(h, dtype=np.float32)[:, None] - cy) / sy
    return np.sqrt(xs * xs + ys * ys)


def radial_gradient(cx, cy, r, stops, w=W, h=H, sx=1.0, sy=1.0) -> np.ndarray:
    d = radial_field(cx, cy, w, h, sx, sy) / r
    return _ramp(stops, np.clip(d, 0, 1))


def angle_field(cx, cy, w=W, h=H) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)[None, :] - cx
    ys = np.arange(h, dtype=np.float32)[:, None] - cy
    return np.arctan2(ys, xs)


# ---------------------------------------------------------------------------
# 노이즈 (타일링 가능한 FFT 프랙탈 노이즈 텍스처 + 아핀 샘플링)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def noise_tex(seed: int = 0, size: int = 1024, beta: float = 2.0) -> np.ndarray:
    """1/f^beta 스펙트럼의 타일링 노이즈 (0..1, 평균 0.5). beta 높을수록 부드러움."""
    rng = np.random.default_rng(seed)
    wn = rng.standard_normal((size, size)).astype(np.float32)
    f = np.fft.rfft2(wn)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    r[0, 0] = 1.0
    f *= r ** (-beta / 2.0)
    f[0, 0] = 0
    n = np.fft.irfft2(f, s=(size, size)).astype(np.float32)
    n = (n - n.mean()) / (n.std() + 1e-8)
    return np.clip(0.5 + n / 6.0, 0, 1).astype(np.float32)


def noise(w=W, h=H, scale=1.0, ox=0.0, oy=0.0, seed=0, beta=2.0,
          sx=1.0, sy=1.0, angle=0.0) -> np.ndarray:
    """화면 크기 노이즈 (H,W) 0..1.

    scale : 무늬 크기 배율(클수록 큼). sx/sy 로 방향별 늘이기(붓결/에어브러시 줄무늬).
    ox/oy : 텍스처 좌표 오프셋(텍셀) — 시간에 따라 바꾸면 흐름.
    """
    tex = noise_tex(seed, 1024, beta)
    c, s = math.cos(angle), math.sin(angle)
    kx = 1.0 / (scale * sx)
    ky = 1.0 / (scale * sy)
    # dst(x,y) = tex(M @ [x,y,1])
    M = np.array([[c * kx, -s * kx, ox], [s * ky, c * ky, oy]], np.float32)
    return cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_WRAP)


def fbm(w=W, h=H, scale=1.0, ox=0.0, oy=0.0, seed=0, octaves=3, gain=0.5, **kw) -> np.ndarray:
    """여러 스케일 노이즈 합 (0..1 근방)."""
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        acc += amp * noise(w, h, scale / (2 ** o), ox * (2 ** o), oy * (2 ** o), seed + o * 17, **kw)
        tot += amp
        amp *= gain
    return acc / tot


def painted_texture(img, amount=0.06, seed=0, scale=3.0, stretch=6.0, angle=0.0, ox=0.0, oy=0.0):
    """배경 미술(포스터컬러/과슈) 느낌의 붓결 얼룩을 곱해 준다 (in-place, 반환도 함)."""
    n1 = noise(img.shape[1], img.shape[0], scale, ox, oy, seed, beta=1.6, sx=stretch, sy=1.0, angle=angle)
    n2 = noise(img.shape[1], img.shape[0], scale * 4, ox * 0.25, oy * 0.25, seed + 5, beta=2.2)
    k = 1.0 + amount * ((n1 - 0.5) * 2.2 + (n2 - 0.5) * 1.4)
    img *= k[..., None]
    return img


# ---------------------------------------------------------------------------
# 래스터화 (안티앨리어스 마스크)
# ---------------------------------------------------------------------------
def _as_polys(polys):
    if isinstance(polys, np.ndarray) and polys.ndim == 2:
        return [polys]
    if len(polys) and np.ndim(polys[0]) == 1:   # 점 리스트 하나
        return [np.asarray(polys, np.float64)]
    return [np.asarray(p, np.float64) for p in polys]


def _fix(p, ox=0.0, oy=0.0):
    p = np.asarray(p, np.float64) - (ox, oy)
    return np.round(p * _SUB).astype(np.int32)


def poly_mask(polys, w=W, h=H) -> np.ndarray:
    """다각형(들) 채움 마스크 (H,W) float32 0..1, 안티앨리어스."""
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [_fix(p) for p in _as_polys(polys)], 255, cv2.LINE_AA, _SHIFT)
    return m.astype(np.float32) * (1 / 255.0)


def ellipse_mask(cx, cy, rx, ry, angle=0.0, w=W, h=H) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(round(cx * _SUB)), int(round(cy * _SUB))),
                (max(0, int(round(rx * _SUB))), max(0, int(round(ry * _SUB)))),
                angle, 0, 360, 255, -1, cv2.LINE_AA, _SHIFT)
    return m.astype(np.float32) * (1 / 255.0)


def circle_mask(cx, cy, r, w=W, h=H) -> np.ndarray:
    return ellipse_mask(cx, cy, r, r, 0, w, h)


def stroke_mask(pts, width=3.0, closed=False, w=W, h=H) -> np.ndarray:
    """균일 두께 선 마스크."""
    m = np.zeros((h, w), np.uint8)
    t = max(1, int(round(width)))
    cv2.polylines(m, [_fix(p) for p in _as_polys(pts)], closed, 255, t, cv2.LINE_AA, _SHIFT)
    out = m.astype(np.float32) * (1 / 255.0)
    if width < 1:
        out *= width
    return out


def soft_mask(mask, sigma):
    """마스크 가장자리 흐리기 (에어브러시 경계)."""
    return blur(mask, sigma)


# ---------------------------------------------------------------------------
# 합성
# ---------------------------------------------------------------------------
def fill(img, mask, color, alpha=1.0):
    """일반(over) 합성. mask (H,W) 0..1. in-place."""
    a = mask[..., None] * alpha if np.ndim(mask) == 2 else mask * alpha
    col = np.asarray(color, np.float32)
    img *= (1 - a)
    img += a * col
    return img


def fill_img(img, mask, src, alpha=1.0):
    """src 이미지(H,W,3)를 마스크로 합성. in-place."""
    a = mask[..., None] * alpha
    img *= (1 - a)
    img += a * src
    return img


def add_light(img, mask, color, strength=1.0):
    """가산 합성(투과광, 네온, 빔). in-place."""
    img += mask[..., None] * (np.asarray(color, np.float32) * strength)
    return img


def screen(img, src, alpha=1.0):
    img[:] = 1 - (1 - np.clip(img, 0, 1)) * (1 - np.clip(src, 0, 1) * alpha) + np.maximum(img - 1, 0)
    return img


def multiply(img, mask, color, alpha=1.0):
    """곱하기(그림자 셀). in-place."""
    a = mask[..., None] * alpha
    img *= (1 - a) + a * np.asarray(color, np.float32)
    return img


def _bbox(pts, pad, w, h):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    x0 = int(max(0, math.floor(pts[:, 0].min() - pad)))
    y0 = int(max(0, math.floor(pts[:, 1].min() - pad)))
    x1 = int(min(w, math.ceil(pts[:, 0].max() + pad) + 1))
    y1 = int(min(h, math.ceil(pts[:, 1].max() + pad) + 1))
    return x0, y0, x1, y1


def fill_poly(img, polys, color, alpha=1.0, mode="over"):
    """bbox 국소 래스터화로 빠른 다각형 채움. mode: over | add | mul. in-place."""
    polys = _as_polys(polys)
    h, w = img.shape[:2]
    allp = np.concatenate([p.reshape(-1, 2) for p in polys])
    x0, y0, x1, y1 = _bbox(allp, 2, w, h)
    if x1 <= x0 or y1 <= y0:
        return img
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [_fix(p, x0, y0) for p in polys], 255, cv2.LINE_AA, _SHIFT)
    _blend_region(img, m, x0, y0, color, alpha, mode)
    return img


def fill_ellipse(img, cx, cy, rx, ry, color, alpha=1.0, angle=0.0, mode="over"):
    h, w = img.shape[:2]
    r = max(rx, ry)
    x0, y0, x1, y1 = _bbox([(cx - r, cy - r), (cx + r, cy + r)], 2, w, h)
    if x1 <= x0 or y1 <= y0:
        return img
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(m, (int(round((cx - x0) * _SUB)), int(round((cy - y0) * _SUB))),
                (max(0, int(round(rx * _SUB))), max(0, int(round(ry * _SUB)))),
                angle, 0, 360, 255, -1, cv2.LINE_AA, _SHIFT)
    _blend_region(img, m, x0, y0, color, alpha, mode)
    return img


def _blend_region(img, m8, x0, y0, color, alpha, mode):
    hh, ww = m8.shape
    a = (m8.astype(np.float32) * (alpha / 255.0))[..., None]
    reg = img[y0:y0 + hh, x0:x0 + ww]
    col = np.asarray(color, np.float32)
    if mode == "add":
        reg += a * col
    elif mode == "mul":
        reg *= (1 - a) + a * col
    else:
        reg *= (1 - a)
        reg += a * col


# ---------------------------------------------------------------------------
# 곡선 / 잉크 선
# ---------------------------------------------------------------------------
def bezier(p0, p1, p2, p3, n=32) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def catmull(points, n=12, closed=False) -> np.ndarray:
    """Catmull-Rom 스플라인으로 제어점들을 부드러운 곡선으로."""
    P = np.asarray(points, np.float64)
    if closed:
        P = np.vstack([P[-1:], P, P[:2]])
    else:
        P = np.vstack([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-2][None, :])
    return np.vstack(out)


def jitter(pts, amp=1.5, seed=0, smooth=5) -> np.ndarray:
    """선 떨림(line boil): 점마다 저주파 오프셋. seed 에 ctx.drawing() 을 섞어 쓴다."""
    P = np.asarray(pts, np.float64)
    if amp <= 0 or len(P) < 2:
        return P
    rng = np.random.default_rng(abs(int(seed)))
    d = rng.standard_normal(P.shape)
    if smooth > 1 and len(P) > smooth:
        k = np.ones(smooth) / smooth
        d = np.stack([np.convolve(np.pad(d[:, i], smooth, mode="wrap"), k, "same")[smooth:-smooth]
                      for i in range(2)], 1)
        d /= (d.std() + 1e-8)
    return P + d * amp


def ink(img, pts, width=3.0, color=(0.03, 0.02, 0.04), taper=(0.15, 0.25), alpha=1.0,
        closed=False, pressure=None):
    """붓펜 느낌의 강약 있는 잉크 선 (셀 주선). in-place.

    width  : 최대 굵기(px).
    taper  : (시작, 끝) 가늘어지는 구간 비율 — 0 이면 뭉툭.
    pressure : 점별 굵기 배율 배열(선택).
    """
    P = np.asarray(pts, np.float64)
    if len(P) < 2:
        return img
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.diff(P, axis=0)
    L = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    tot = L[-1] if L[-1] > 0 else 1.0
    s = L / tot
    wv = np.ones(len(P))
    if not closed:
        t0, t1 = taper
        if t0 > 0:
            wv *= np.clip(s / t0, 0.0, 1.0) ** 0.7
        if t1 > 0:
            wv *= np.clip((1 - s) / t1, 0.0, 1.0) ** 0.7
    if pressure is not None:
        wv *= np.interp(np.linspace(0, 1, len(P)), np.linspace(0, 1, len(pressure)), pressure)
    wv = np.maximum(wv * width * 0.5, 0.35)
    # 법선
    tang = np.gradient(P, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    left = P + nrm * wv[:, None]
    right = P - nrm * wv[:, None]
    poly = np.vstack([left, right[::-1]])
    fill_poly(img, poly, color, alpha)
    return img


def ink_ellipse(img, cx, cy, rx, ry, width=3.0, color=(0.03, 0.02, 0.04), angle=0.0, seed=0,
                boil=0.0, n=72, alpha=1.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = math.radians(angle)
    x = rx * np.cos(th)
    y = ry * np.sin(th)
    pts = np.stack([cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)], 1)
    if boil:
        pts = jitter(pts, boil, seed)
    press = 0.75 + 0.25 * np.sin(th * 2 + seed)
    return ink(img, pts, width, color, (0, 0), alpha, closed=True, pressure=press)


def cel_shape(img, pts, base, shade=None, shade_pts=None, line=(0.03, 0.02, 0.04), line_w=3.0,
              boil=0.0, seed=0, hi=None, hi_pts=None, closed=True):
    """셀 한 장: 베이스 색 + (선택) 그림자 색 영역 + (선택) 하이라이트 영역 + 주선.

    shade_pts / hi_pts 는 그림자/하이라이트 다각형(셀 그림자는 경계가 딱딱하다).
    두 영역은 베이스 다각형 안으로 클리핑된다.
    """
    P = jitter(pts, boil, seed) if boil else np.asarray(pts, np.float64)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _bbox(P, 4 + line_w, w, h)
    if x1 <= x0 or y1 <= y0:
        return img
    base_m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(base_m, [_fix(P, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
    _blend_region(img, base_m, x0, y0, base, 1.0, "over")
    for col, sp in ((shade, shade_pts), (hi, hi_pts)):
        if col is None or sp is None:
            continue
        SP = jitter(sp, boil, seed + 7) if boil else np.asarray(sp, np.float64)
        sm = np.zeros_like(base_m)
        cv2.fillPoly(sm, [_fix(SP, x0, y0)], 255, cv2.LINE_AA, _SHIFT)
        sm = cv2.min(sm, base_m)
        _blend_region(img, sm, x0, y0, col, 1.0, "over")
    if line is not None and line_w > 0:
        ink(img, P, line_w, line, (0, 0), closed=closed,
            pressure=0.8 + 0.2 * np.sin(np.linspace(0, 6.28 * 3, len(P)) + seed))
    return img


# ---------------------------------------------------------------------------
# 블러 / 글로우
# ---------------------------------------------------------------------------
def blur(img, sigma):
    """가우시안 블러. 큰 sigma 는 다운샘플로 빠르게."""
    if sigma <= 0.05:
        return img.copy()
    h, w = img.shape[:2]
    if sigma <= 6:
        return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    k = int(min(16, max(2, sigma / 3)))
    sw, sh = max(1, w // k), max(1, h // k)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def glow(src, sigmas=(4, 16, 48), weights=(0.6, 0.35, 0.25)):
    """다중 스케일 글로우 (src 는 이미지 또는 마스크)."""
    out = np.zeros_like(src, dtype=np.float32)
    for s, wgt in zip(sigmas, weights):
        out += blur(src.astype(np.float32), s) * wgt
    return out


def add_glow(img, src, sigmas=(4, 16, 48), weights=(0.6, 0.35, 0.25), color=None, strength=1.0):
    """src(마스크/이미지)의 글로우를 img 에 가산. in-place."""
    g = glow(src, sigmas, weights)
    if g.ndim == 2:
        col = np.asarray(color if color is not None else (1, 1, 1), np.float32)
        img += g[..., None] * col * strength
    else:
        img += g * strength * (np.asarray(color, np.float32) if color is not None else 1.0)
    return img


def motion_blur(img, length, angle_deg=0.0):
    """방향성 모션 블러 (스피드 스미어)."""
    L = int(max(1, round(length)))
    if L <= 1:
        return img.copy()
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    k /= k.sum() + 1e-8
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)


def zoom_blur(img, cx, cy, amount=0.08, steps=8):
    """방사형 줌 블러 (폭발/돌진)."""
    h, w = img.shape[:2]
    acc = img.astype(np.float32).copy()
    for i in range(1, steps):
        s = 1 + amount * i / steps
        M = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], np.float32)
        acc += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return acc / steps


# ---------------------------------------------------------------------------
# 변형
# ---------------------------------------------------------------------------
def affine(img, M, w=None, h=None, border=cv2.BORDER_REFLECT):
    h0, w0 = img.shape[:2]
    return cv2.warpAffine(img, np.asarray(M, np.float32), (w or w0, h or h0),
                          flags=cv2.INTER_LINEAR, borderMode=border)


def transform(img, dx=0.0, dy=0.0, scale=1.0, rot_deg=0.0, cx=None, cy=None,
              border=cv2.BORDER_REFLECT):
    """중심 (cx,cy) 기준 확대/회전 후 이동 (카메라 트럭/팬/셰이크)."""
    h, w = img.shape[:2]
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)


# ---------------------------------------------------------------------------
# 효과선 (集中線 / 流線)
# ---------------------------------------------------------------------------
def radial_lines_mask(cx, cy, n=120, seed=0, r_in=200.0, r_jit=0.5, width=(0.002, 0.010),
                      w=W, h=H) -> np.ndarray:
    """집중선 마스크. r_in: 안쪽 빈 원 반지름, r_jit: 시작 반경 랜덤폭(비율)."""
    rng = np.random.default_rng(seed)
    R = math.hypot(w, h) * 1.2
    polys = []
    for _ in range(n):
        a = rng.uniform(0, 2 * np.pi)
        da = rng.uniform(*width)
        r0 = r_in * (1 + rng.uniform(0, r_jit))
        tip = (cx + math.cos(a) * r0, cy + math.sin(a) * r0)
        polys.append(np.array([tip,
                               (cx + math.cos(a - da) * R, cy + math.sin(a - da) * R),
                               (cx + math.cos(a + da) * R, cy + math.sin(a + da) * R)]))
    return poly_mask(polys, w, h)


def streak_lines_mask(angle_deg=0.0, n=80, seed=0, phase=0.0, length=(200, 900), width=(1.0, 5.0),
                      w=W, h=H) -> np.ndarray:
    """평행 유선(流線) 마스크. phase(px) 를 시간에 따라 증가시키면 흘러간다."""
    rng = np.random.default_rng(seed)
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-d[1], d[0]])
    diag = math.hypot(w, h)
    polys = []
    for _ in range(n):
        off = rng.uniform(-diag / 2, diag / 2)
        ln = rng.uniform(*length)
        wd = rng.uniform(*width) / 2
        span = diag + ln * 2
        pos = (rng.uniform(0, span) + phase * rng.uniform(0.7, 1.3)) % span - span / 2
        c = np.array([w / 2, h / 2]) + nrm * off + d * pos
        p0, p1 = c - d * ln / 2, c + d * ln / 2
        polys.append(np.array([p0, p1 - nrm * wd, p1 + nrm * wd]) if wd < 0.8 else
                     np.array([p0, p1 - nrm * wd, p1 + nrm * wd]))
    return poly_mask(polys, w, h)


# ---------------------------------------------------------------------------
# 텍스트
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype(FONT_JP_ALT, size)


def text_mask(text, size=120, font=FONT_JP, stroke=0, vertical=False, spacing=0) -> np.ndarray:
    """글자 마스크(꼭 맞게 잘린 float32 2D). stroke>0 이면 굵게(붓글씨풍 볼드).
    vertical=True 면 세로쓰기(한 글자씩 아래로)."""
    f = _font(font, int(size))
    if vertical:
        chars = list(text)
        cw = int(size * 1.15) + stroke * 2
        ch = int(size * 1.08) + spacing
        im = Image.new("L", (cw, ch * len(chars) + stroke * 2 + 8), 0)
        d = ImageDraw.Draw(im)
        for i, c in enumerate(chars):
            bb = d.textbbox((0, 0), c, font=f, stroke_width=stroke)
            x = (cw - (bb[2] - bb[0])) / 2 - bb[0]
            d.text((x, i * ch + stroke + 4), c, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    else:
        d0 = ImageDraw.Draw(Image.new("L", (8, 8)))
        bb = d0.textbbox((0, 0), text, font=f, stroke_width=stroke)
        tw = bb[2] - bb[0] + spacing * max(0, len(text) - 1) + 8
        im = Image.new("L", (int(tw) + 8, int(bb[3] - bb[1]) + 12), 0)
        d = ImageDraw.Draw(im)
        if spacing:
            x = 4 - bb[0]
            for c in text:
                d.text((x, 4 - bb[1]), c, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
                x += d.textlength(c, font=f) + spacing
        else:
            d.text((4 - bb[0], 4 - bb[1]), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    a = np.asarray(im, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.01)
    if len(xs) == 0:
        return a
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def place_mask(m, x, y, w=W, h=H, scale=1.0, rot_deg=0.0, anchor="center") -> np.ndarray:
    """작은 마스크 m 을 화면 크기 마스크로 배치. anchor='center' 면 (x,y)=중심."""
    mh, mw = m.shape
    ax, ay = (mw / 2, mh / 2) if anchor == "center" else (0.0, 0.0)
    M = cv2.getRotationMatrix2D((ax, ay), rot_deg, scale)
    M[0, 2] += x - ax
    M[1, 2] += y - ay
    return cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# 기타 셀 질감
# ---------------------------------------------------------------------------
def posterize(x, levels=3):
    """연속 값을 단계(셀 톤 분리)로."""
    return np.floor(clamp(x) * levels) / max(1, levels - 1)


def halftone_dots(w=W, h=H, cell=8, angle=45.0, value=None):
    """망점(스크린톤) 마스크. value(H,W) 0..1 이 진할수록 점이 큼."""
    a = math.radians(angle)
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    u = (xs * math.cos(a) + ys * math.sin(a)) / cell
    v = (-xs * math.sin(a) + ys * math.cos(a)) / cell
    d = np.hypot(u - np.floor(u) - 0.5, v - np.floor(v) - 0.5)
    r = 0.5 * np.sqrt(value if value is not None else 0.5)
    return np.clip((r - d) * cell, 0, 1)


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
