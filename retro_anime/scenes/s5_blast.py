"""초능력 대폭발 — AKIRA 오마주 클라이맥스 (60f, 11.0–13.5s).

연출 (장면 내부 프레임)
  f0–1   순백 (4의 흰 플래시에서 이어짐)
  f2–6   흰빛이 도시 한가운데의 '씨앗' 광점으로 빨려 들어가며 밤의 메가시티(고공 부감)가 드러남
  f6,10  심장 박동: 씨앗이 맥동, 지면에 희미한 파문, 빛 기둥
  f8–13  정전 파도: 중심에서 바깥으로 창문 불빛이 꺼짐 → 숨을 들이켜는 암전
  f14    폭발 — 반전(네거) 프레임, f15 적색 단색, f16 과노출
  f14–27 순백 구체(청백 림, 투과광)가 돔처럼 솟아 팽창, 동심원 충격파 링이 지면을 질주,
         집중선, 빌딩이 날아가고(셀, 2코마) 구름에 구멍, 히어로 타워 상부가 부러져 넘어감
  f28    2차 서지 — 적색 단색, f29 반전(검은 구체 + 붉은 림), 응결 고리 구름, 방전 아크, 불티
  f28–39 거대한 먼지 벽(콜리플라워 뭉게 셀 3톤 + 백열/주황 역광 띠, 2코마)이 도시를 휩쓸며 카메라 쪽으로
         솟아오름, f36 전경 타워 붕괴. 히어로 타워 창문은 충격파가 오며 칸칸이 꺼진다
  f40    충격파 도달 — 반전, f41 적색: 순백 빛 속 도시 실루엣을 '빛의 전선'이 중심부터 먹어 들어감
  f42–51 빛 속을 스치는 파편 스미어·먼지, 백열로 차오름
  f52–59 완전한 흰색 (HDR) → 타이틀로 하드 컷

구조 (2.5D)
  * 고정 카메라(높이 HC, 지평선 HZ)의 핀홀 투영. 도시는 박스 빌딩 수천 채를 화가 알고리즘으로
    한 번만 그려 캐시: 밤 색(N), 폭발 실루엣(B), 폭발광 기여(L), 창문 발광(W), 커버리지(A),
    깊이(Z), 폭심 거리(D), 면 종류(K). 매 프레임은 이 레이어들을 섞기만 한다.
    원·중경은 면 ID 버퍼(cv2)로 래스터 후 LUT 채색, 근경만 안티앨리어스 경로 (캐시 ~3.5s/프로세스).
  * 구체·충격파 링·먼지 벽은 Z/D 맵으로 깊이 합성(구체는 지면 앞쪽에 가려 돔이 된다).
  * 카메라 셰이크/푸시/줌 블러는 2D 촬영 처리(1코마), 작화(파편/타워 붕괴/먼지)는 2코마.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from ..core import (H, W, Ctx, _SHIFT, _fix, cel_shape, clamp, ease_in, ease_out, fill_poly,
                    hash01, hexc, luma, mix, noise, noise_tex, painted_texture, radial_lines_mask,
                    smoothstep, transform, wobble, zoom_blur)

NAME = "s5_blast"
FRAMES = 60

EVENTS = [
    (0, "flash"),        # 4의 흰 플래시에서 이어짐
    (3, "swell"),        # 흰빛이 씨앗으로 빨려 들어감
    (6, "heartbeat"),
    (10, "heartbeat"),
    (12, "swell"),       # 정전 → 숨 들이켜기
    (14, "boom"),        # 폭발 (반전 프레임)
    (14, "impact"),
    (14, "flash"),
    (15, "shockwave"),   # 적색 단색 프레임, 첫 충격파
    (16, "flash"),
    (18, "shatter"),     # 오른쪽 히어로 타워 상부 부러짐
    (20, "shatter"),     # 왼쪽 히어로 타워
    (24, "whoosh"),      # 파편 비산
    (28, "boom"),        # 2차 서지 (적색)
    (28, "shockwave"),
    (29, "flash"),       # 반전
    (36, "shatter"),     # 전경 타워 붕괴
    (38, "whoosh"),      # 먼지 벽이 카메라로
    (40, "impact"),      # 충격파 도달 (반전)
    (40, "shockwave"),
    (40, "flash"),
    (41, "boom"),        # 적색, 빛의 전선이 도시를 먹어 들어감
    (43, "passby"),
    (45, "passby"),
    (47, "swell"),       # 백열로 차오름
    (52, "flash"),       # 순백 → 타이틀 하드 컷
]

# ---------------------------------------------------------------------------
# 카메라 (고정 핀홀) — 지면 Y=0, 카메라 (0, HC, 0) 에서 +Z 를 바라봄
# ---------------------------------------------------------------------------
F = 1500.0
CX = 960.0
HZ = 380.0
HC = 280.0
Z0 = 1800.0          # 폭심(구체 중심) 깊이 (m)
T_BANG = 14
T_WHITE = 52
YCL = 720.0          # 구름층 고도 (m)
SEED_Y = 40.0

f32 = np.float32
YY = np.arange(H, dtype=f32)[:, None]
XX = np.arange(W, dtype=f32)[None, :]


def sx(X, Z):
    return CX + F * X / Z


def sy(Y, Z):
    return HZ + F * (HC - Y) / Z


# ---------------------------------------------------------------------------
# 팔레트
# ---------------------------------------------------------------------------
INK = hexc("#0a0612")
RIM = hexc("#bfe8ff")
CORE = np.array([1.70, 1.74, 1.85], f32)
WHITE_HOT = np.array([2.4, 2.45, 2.6], f32)
SODIUM = hexc("#ff8a2a")
CYAN = hexc("#3ff0ff")
RED = hexc("#d4141f")
LIGHT_NEAR = np.array([0.88, 0.96, 1.0], f32)
LIGHT_FAR = np.array([0.35, 0.58, 1.0], f32)
HAZE_N = hexc("#5c2a62")
HAZE_BLAST = np.array([0.55, 0.72, 1.0], f32)
FIRE = np.array([1.9, 0.75, 0.25], f32)
RIMC = np.array([0.22, 0.74, 1.25], f32)        # 구체 림 (포화된 청백, HDR — 필름 할레이션 후에도 푸르게)
DUST_D = hexc("#24162a")
WARM_HOT = np.array([2.4, 2.0, 1.55], f32)      # 역광 림 백열
WARM_ORG = np.array([1.7, 0.68, 0.24], f32)     # 역광 림 주황
WARM_LIT = np.array([1.9, 1.15, 0.60], f32)     # 셀 조각의 구체 쪽 하이라이트
WIN_TINT = np.array([1.25, 0.80, 0.50], f32)    # 폭발광에 물든 창문
DUST_M = hexc("#4a2a3c")


# ---------------------------------------------------------------------------
# 타이밍 곡선
# ---------------------------------------------------------------------------
SPH_F = np.array([14, 16, 18, 20, 24, 27, 32, 36, 40, 44, 47, 49, 51, 53], np.float64)
SPH_R = np.array([45, 120, 195, 250, 320, 365, 410, 455, 540, 690, 900, 1130, 1420, 1600], np.float64)


def sphere_R(lf: float) -> float:
    if lf < T_BANG:
        return 0.0
    return float(np.interp(lf, SPH_F, SPH_R))


def sphere_screen(R: float):
    """구체 → 화면 (cx, cy, r_px, Yc)."""
    Yc = 30.0 + 0.32 * R
    D = math.hypot(Z0, HC - Yc)
    r = F * R / math.sqrt(max(D * D - R * R, 400.0))
    return CX, HZ + F * (HC - Yc) / Z0, r, Yc


# (발생 프레임, 시작 반경, 끝 반경, 지속, 이징 지수, 밝기, 파괴력)
RINGS = [
    (14, 90.0, 1250.0, 22, 2.3, 1.0, True),
    (20, 80.0, 950.0, 18, 2.0, 0.55, False),
    (28, 430.0, 2150.0, 18, 1.45, 1.25, True),
    (35, 520.0, 1500.0, 14, 2.0, 0.5, False),
]


def ring_R(k: int, lf: float):
    e, r0, r1, dur, p, s, _ = RINGS[k]
    if lf < e:
        return None
    u = (lf - e) / dur
    return r0 + (r1 - r0) * float(ease_out(min(u, 1.0), p))


def destroy_R(lf: float) -> float:
    """파괴 전선 반경 (단조 증가)."""
    best = 0.0
    for k, rg in enumerate(RINGS):
        if not rg[6]:
            continue
        r = ring_R(k, lf)
        if r is not None:
            best = max(best, r)
    return max(best - 25.0, sphere_R(lf))


def hit_frame(d: float) -> float:
    """거리 d 의 대상이 파괴 전선에 닿는 (소수) 프레임."""
    if destroy_R(59) < d:
        return 1e9
    lo, hi = float(T_BANG), 59.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if destroy_R(mid) >= d:
            hi = mid
        else:
            lo = mid
    return hi


def intensity(lf: float) -> float:
    if lf < T_BANG:
        return 0.0
    v = 2.1 + 2.2 * math.exp(-(lf - T_BANG) / 2.2)
    v += 1.1 * math.exp(-max(lf - 28, 0) / 2.0) * (lf >= 28)
    v += 1.6 * float(clamp((lf - 40) / 11.0)) ** 1.5
    return v


def shake_amp(lf: float) -> float:
    a = 0.0
    if lf >= 5:
        a += 1.5 * float(clamp((lf - 5) / 8.0))
    for t0, amp, dec in ((14, 34.0, 3.0), (28, 22.0, 3.0), (40, 40.0, 4.0)):
        if lf >= t0:
            a += amp * math.exp(-(lf - t0) / dec)
    if lf >= 14:
        a += 4.0 + 8.0 * float(clamp((lf - 36) / 14.0))
    return a


# ---------------------------------------------------------------------------
# 래스터 캔버스 (국소 bbox 래스터화)
# ---------------------------------------------------------------------------
class _Cv:
    """N(0:3) B(3:6) L(6:9) W(9:12) 를 한 배열에 묶은 레이어 캔버스."""

    def __init__(self, x0=0, y0=0, w=W, h=H):
        self.ox, self.oy, self.w, self.h = x0, y0, w, h
        self.C = np.zeros((h, w, 12), f32)
        self.A = np.zeros((h, w), f32)
        self.Z = np.full((h, w), 1e7, f32)
        self.D = np.full((h, w), 1e7, f32)
        self.K = np.zeros((h, w), np.uint8)
        self.I = np.full((h, w), -1, np.int32)
        self.wp = []          # 창문 파라미터 (정면마다 한 줄)


def _cov1d(a, b, n):
    ia = max(0, int(math.floor(a)))
    ib = min(n, int(math.ceil(b)))
    if ib <= ia:
        return None
    i = np.arange(ia, ib, dtype=f32)
    c = np.clip(np.minimum(i + 1, b) - np.maximum(i, a), 0, 1).astype(f32)
    return ia, c


def _rect_mask(cv, x0, x1, y0, y1):
    cx_ = _cov1d(x0 - cv.ox, x1 - cv.ox, cv.w)
    cy_ = _cov1d(y0 - cv.oy, y1 - cv.oy, cv.h)
    if cx_ is None or cy_ is None:
        return None
    return cy_[1][:, None] * cx_[1][None, :], cx_[0], cy_[0]


def _poly_mask_local(cv, polys):
    polys = [np.asarray(p, np.float64) - (cv.ox, cv.oy) for p in polys]
    allp = np.concatenate([p.reshape(-1, 2) for p in polys])
    x0 = int(max(0, math.floor(allp[:, 0].min() - 2)))
    y0 = int(max(0, math.floor(allp[:, 1].min() - 2)))
    x1 = int(min(cv.w, math.ceil(allp[:, 0].max() + 2) + 1))
    y1 = int(min(cv.h, math.ceil(allp[:, 1].max() + 2) + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for p in polys:
        cv2.fillConvexPoly(m, _fix(p, x0, y0), 255, cv2.LINE_AA, _SHIFT)
    return m.astype(f32) * (1 / 255.0), x0, y0


def _pack(cN, cB, cL):
    arrs = [np.asarray(c, f32) for c in (cN, cB, cL)]
    if all(a.ndim == 1 for a in arrs):
        return np.concatenate(arrs)
    shp = np.broadcast_shapes(*[a.shape for a in arrs])
    return np.concatenate([np.broadcast_to(a, shp) for a in arrs], -1)


def _put(cv, mm, c9, z, d, kind, bid=-1):
    """커버리지 mm=(m,x0,y0) 로 면 하나를 덮어 그린다 (창문 발광은 가려짐)."""
    m, x0, y0 = mm
    h, w = m.shape
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    a = m[..., None]
    reg = cv.C[sl]
    reg *= 1 - a
    reg[..., :9] += a * c9
    ar = cv.A[sl]
    ar += m * (1 - ar)
    sel = m > 0.5
    np.copyto(cv.Z[sl], f32(z), where=sel)
    np.copyto(cv.D[sl], f32(d), where=sel)
    np.copyto(cv.K[sl], np.uint8(kind), where=sel)
    np.copyto(cv.I[sl], np.int32(bid), where=sel)


def _add_rect_light(cv, x0, x1, y0, y1, col):
    mr = _rect_mask(cv, x0, x1, y0, y1)
    if mr is not None:
        m, ix, iy = mr
        cv.C[iy:iy + m.shape[0], ix:ix + m.shape[1], 6:9] += m[..., None] * col


# ---------------------------------------------------------------------------
# 지면 좌표 맵 (캐시)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ground_maps():
    y = np.arange(H, dtype=np.float64)[:, None] + 0.5
    x = (np.arange(W, dtype=f32)[None, :] + 0.5 - f32(CX)) / f32(F)
    dy = y - HZ
    Zg = np.where(dy > 0.25, F * HC / np.maximum(dy, 0.25), 1e7).astype(f32)
    Xg = x * Zg
    Dg = np.sqrt(Xg * Xg + (Zg - f32(Z0)) ** 2)
    Dg[(dy[:, 0] <= 0.25)] = 1e7
    return Zg, Xg, Dg


# ---------------------------------------------------------------------------
# 하늘 (캐시) — 그라디언트 + 원근 구름층 + 별
# ---------------------------------------------------------------------------
def _remap_tex(tex, u, v):
    n = tex.shape[0]
    u, v = np.broadcast_arrays(u, v)
    mu = (u % 1.0) * n
    mv = (v % 1.0) * n
    return cv2.remap(tex, mu.astype(f32), mv.astype(f32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def _cloud_n(Xc, Zc):
    t1 = noise_tex(41, 1024, 2.8)
    t2 = noise_tex(57, 1024, 2.2)
    return (_remap_tex(t1, Xc / 14000.0, Zc / 6000.0) * 0.72
            + _remap_tex(t2, Xc / 4200.0, Zc / 2000.0) * 0.28)


@lru_cache(maxsize=1)
def _sky():
    hs = int(HZ) + 40
    y = np.arange(hs, dtype=f32)[:, None] + 0.5
    x = np.arange(W, dtype=f32)[None, :] + 0.5
    stops = [(0, "#05061a"), (120, "#0c0b2c"), (230, "#1b1446"), (310, "#3a1a56"),
             (352, "#6a2462"), (378, "#a2386a"), (hs, "#c24a6a")]
    ys = np.array([p for p, _ in stops], f32)
    cols = np.array([hexc(c) for _, c in stops], f32)
    grad = np.stack([np.interp(y[:, 0], ys, cols[:, c]) for c in range(3)], 1)
    sky = np.broadcast_to(grad[:, None, :], (hs, W, 3)).astype(f32).copy()
    hb = np.exp(-((HZ - y) / 26.0) ** 2) * (0.8 + 0.4 * noise(W, hs, 6.0, ox=300.0, seed=57, beta=2.2))
    sky += hb[..., None] * np.array([0.55, 0.20, 0.22], f32)

    # 원근 구름층 (YCL 고도의 평면)
    dy = np.maximum(HZ - y, 0.6)
    Zc = F * (YCL - HC) / dy
    Xc = (x - CX) * Zc / F
    n = _cloud_n(Xc, Zc)
    fade = np.exp(-Zc / 30000.0)
    n = n * fade + 0.47 * (1 - fade)
    dens = smoothstep(0.46, 0.56, n) * (0.35 + 0.65 * fade)
    dist = np.hypot(Xc, Zc - Z0) + 1e-3
    shx, shz = -Xc / dist * 300.0, (Z0 - Zc) / dist * 300.0
    n_sh = _cloud_n(Xc + shx, Zc + shz)
    n_sh = n_sh * fade + 0.47 * (1 - fade)
    edge = np.clip((n - n_sh) * 16.0, 0, 1) * dens
    hor = np.clip(1 - (HZ - y) / 300.0, 0, 1) ** 1.5
    c_dark = hexc("#140d2c")
    c_lit = hexc("#5a2252")
    cloud = c_dark[None, None, :] * (1 - hor[..., None]) + c_lit * hor[..., None]
    cloud = cloud + edge[..., None] * np.array([0.32, 0.10, 0.22], f32)
    sky = sky * (1 - dens[..., None] * 0.92) + cloud * dens[..., None] * 0.92 + \
        (hor * dens)[..., None] * np.array([0.25, 0.07, 0.12], f32)
    rng = np.random.default_rng(77)
    st = np.zeros((hs, W), f32)
    k = 260
    st[rng.uniform(0, 300, k).astype(int), rng.uniform(0, W, k).astype(int)] = rng.uniform(0.3, 1.0, k)
    st = cv2.GaussianBlur(st, (0, 0), 0.7) * 2.5
    sky += (st * (1 - dens) * np.clip((HZ - 120 - y) / 160.0, 0, 1))[..., None] * \
        np.array([0.9, 0.9, 1.0], f32)
    painted_texture(sky, 0.05, seed=501, scale=5.0, stretch=8.0, ox=211.0, oy=97.0)
    full = np.zeros((H, W, 3), f32)
    full[:hs] = sky
    full[hs:] = sky[-1]
    D = np.full((H, W), 1e7, f32)
    D[:hs] = dist
    E = np.zeros((H, W), f32)
    E[:hs] = edge
    Dn = np.zeros((H, W), f32)
    Dn[:hs] = dens
    return full, Dn, E, D


# ---------------------------------------------------------------------------
# 도시 생성
# ---------------------------------------------------------------------------
P_BLOCK = 112.0
STREET = 26.0
AVE = 60.0                 # 중앙 대로 반폭
RIVER = (930.0, 1045.0)    # 강 (Z 범위)
PLAZA = 240.0              # 폭심 광장 반경


def _gen_buildings():
    rng = np.random.default_rng(1988)
    out = []
    zs = np.arange(300.0, 8200.0, P_BLOCK)
    for zA in zs:
        zB = zA + P_BLOCK - STREET
        if zB > RIVER[0] - 6 and zA < RIVER[1] + 6:
            continue
        xmax = (CX + 160) / F * zB + 140
        far = zA > 4200
        pb = P_BLOCK * (2 if zA > 4500 else 1)
        nk = int(xmax / pb) + 1
        for side in (-1, 1):
            for k in range(nk):
                xa = AVE + k * pb
                xb = xa + pb - STREET
                if xa > xmax:
                    break
                r = rng.random()
                if far or r < 0.30:
                    lots = [(xa, xb, zA, zB)]
                elif r < 0.72:
                    if rng.random() < 0.5:
                        m = xa + (xb - xa) * rng.uniform(0.38, 0.62)
                        lots = [(xa, m - 3, zA, zB), (m + 3, xb, zA, zB)]
                    else:
                        m = zA + (zB - zA) * rng.uniform(0.38, 0.62)
                        lots = [(xa, xb, zA, m - 3), (xa, xb, m + 3, zB)]
                else:
                    mx = xa + (xb - xa) * rng.uniform(0.4, 0.6)
                    mz = zA + (zB - zA) * rng.uniform(0.4, 0.6)
                    lots = [(xa, mx - 3, zA, mz - 3), (mx + 3, xb, zA, mz - 3),
                            (xa, mx - 3, mz + 3, zB), (mx + 3, xb, mz + 3, zB)]
                for (x0, x1, z0, z1) in lots:
                    sb = rng.uniform(0, 5)
                    x0, x1, z0, z1 = x0 + sb, x1 - sb, z0 + sb, z1 - sb
                    if side < 0:
                        x0, x1 = -x1, -x0
                    xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
                    d = math.hypot(xc, zc - Z0)
                    if d < PLAZA + 40:
                        continue
                    cbd = math.exp(-(d / 1150.0) ** 2)
                    base = 20 + 210 * cbd + 30 * math.exp(-zc / 6000.0)
                    if zc < 1650 and abs(xc) < 560:         # 폭심 앞 저층 지구 (시야 확보)
                        base = 14 + 22 * abs(xc) / 560
                    elif zc < 950:
                        base = min(base, 55.0)
                    hgt = base * math.exp(rng.normal(0, 0.42))
                    if rng.random() < 0.05 and not (zc < 1650 and abs(xc) < 560):
                        hgt *= 1.8
                    hgt = float(np.clip(hgt, 9, 430))
                    nx = min(max(0.0, x0), x1)
                    nz = min(max(Z0, z0), z1)
                    dn = math.hypot(nx, nz - Z0)
                    out.append((x0, x1, z0, z1, hgt, dn, int(rng.integers(1 << 30))))
    return out


N_FRONT_T = hexc("#17132e")
N_FRONT_B = hexc("#2c1c40")
N_SIDE = hexc("#0f0c22")
N_ROOF = hexc("#2f2550")
B_SIL = hexc("#07060e")
B_ROOF = hexc("#0c0a16")
B_HAZE = hexc("#1a1230")
WIN_COLS = np.array([hexc("#ffe2b0"), hexc("#ffc27a"), hexc("#ffd890"), hexc("#bff0ff"),
                     hexc("#9fe0ff"), hexc("#ff8ab8"), hexc("#ffb060")], f32)
WIN_P = np.array([0.34, 0.2, 0.2, 0.1, 0.06, 0.04, 0.06])


def _light_I(d):
    return 1.0 / (1.0 + (d / 650.0) ** 2)


def _light_col(d):
    return mix(LIGHT_NEAR, LIGHT_FAR, float(clamp(d / 2200.0)))


def _hash(a, b, c, salt):
    x = (a.astype(np.int64) * 73856093) ^ (b.astype(np.int64) * 19349663) ^ \
        (c.astype(np.int64) * 83492791) ^ np.int64(salt * 2654435761 & 0xFFFFFFFF)
    x = (x ^ (x >> 13)) * 1274126177
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def _windows_vec(cv, Wn):
    """보이는 정면 픽셀 전체에 손으로 찍은 듯한 창문 불빛을 한 번에 (가려진 면은 계산 안 함)."""
    if not cv.wp:
        return
    P = np.array(cv.wp, np.float64)
    flat = cv.I.ravel()
    idx = np.flatnonzero(flat >= 0)
    bid = flat[idx]
    ys = (idx // cv.w).astype(np.float64) + cv.oy + 0.5
    xs = (idx % cv.w).astype(np.float64) + cv.ox + 0.5
    fx0, fy0, ch, cw, occ, ww, wh, att, cmain, sd = (P[bid, k] for k in range(10))
    v = (ys - fy0) / ch
    u = (xs - fx0) / cw
    fi = np.floor(v)
    cj = np.floor(u)
    sdi = sd.astype(np.int64)
    h1 = _hash(sdi, fi, cj, 1)
    hf = _hash(sdi, fi, np.zeros_like(fi), 2)
    lit = ((h1 < occ) & (hf >= 0.2)) | ((hf > 0.9) & (h1 < 0.92))
    hc = _hash(sdi, fi, cj, 3)
    crand = np.searchsorted(np.cumsum(WIN_P) / WIN_P.sum(), _hash(sdi, fi, cj, 4))
    ci = np.where(hc < 0.7, cmain.astype(np.int64), crand)
    ci = np.minimum(ci, len(WIN_COLS) - 1)
    bri = (0.5 + 0.85 * _hash(sdi, fi, cj, 5)) * att
    fv = v - fi
    fu = u - cj
    big = (ch >= 2.2) & (cw >= 1.6)
    cov = np.where(big,
                   np.clip((wh - np.abs(fv - 0.55)) * ch + 0.5, 0, 1) *
                   np.clip((ww - np.abs(fu - 0.5)) * cw + 0.5, 0, 1),
                   (2 * ww) * (2 * wh) * 0.9)
    amt = (lit * cov * bri).astype(f32)
    Wn.reshape(-1, 3)[idx] += WIN_COLS[ci] * amt[:, None]


def _box_faces(x0, x1, z0, z1, y0, y1, d, seed, rim=True, win_min=9.0):
    """박스 하나의 보이는 면들(옆면/지붕/정면)과 각 면의 색·깊이·림·창문 파라미터."""
    rng = np.random.default_rng(seed)
    hz = 1 - math.exp(-max(z0 - 300, 0) / 5200.0)
    tint = rng.uniform(0.82, 1.15)
    hue = rng.uniform(-0.05, 0.05)
    tcol = np.array([1 + hue, 1, 1 - hue], f32) * tint
    I = _light_I(d)
    lc = _light_col(d)
    xc = (x0 + x1) / 2
    cBh = B_SIL * (1 - hz * 0.6) + B_HAZE * (hz * 0.6)
    faces = []
    # 옆면 (중심선 쪽 = 폭심 쪽 면만 보인다 → 폭발광을 받음)
    sidex = x0 if x0 > 0 else (x1 if x1 < 0 else None)
    if sidex is not None and z0 < 5200:
        pts = np.array([(sx(sidex, z0), sy(y0, z0)), (sx(sidex, z0), sy(y1, z0)),
                        (sx(sidex, z1), sy(y1, z1)), (sx(sidex, z1), sy(y0, z1))])
        zm = (z0 + z1) / 2
        dist = math.hypot(sidex, zm - Z0) + 1
        cosv = abs(sidex) / dist
        cN = N_SIDE * tcol * (1 - hz * 0.85) + HAZE_N * (hz * 0.85)
        faces.append(dict(kind=3, poly=pts, cNt=cN, cNb=cN, cB=cBh, cL=lc * I ** 1.4 * (0.05 + 0.22 * cosv),
                          lgrad=0, z=zm, d=d, fy0=0.0, fy1=1.0, wp=None,
                          rim=(3, sx(sidex, z0), 1.0 if sidex > 0 else -1.0, lc * I * 0.9) if rim else None))
    # 지붕
    if y1 < HC - 2 and (z0 < 5200 or win_min < 9):
        pts = np.array([(sx(x0, z0), sy(y1, z0)), (sx(x1, z0), sy(y1, z0)),
                        (sx(x1, z1), sy(y1, z1)), (sx(x0, z1), sy(y1, z1))])
        cN = N_ROOF * tcol * (1 - hz * 0.8) + HAZE_N * 1.1 * (hz * 0.8)
        cB = B_ROOF * (1 - hz * 0.6) + B_HAZE * (hz * 0.6)
        up = 0.04 + 0.05 * float(clamp((Z0 - z1) / 1500.0 + 0.5))
        rr = (2, sy(y1, z1), 0.0, lc * I * I * 1.6) if (rim and z1 < Z0 and z0 < 5000) else None
        faces.append(dict(kind=4, poly=pts, cNt=cN, cNb=cN, cB=cB, cL=lc * I * up, lgrad=0,
                          z=(z0 + z1) / 2, d=d, fy0=0.0, fy1=1.0, wp=None, rim=rr))
    # 정면 (카메라를 향함)
    fx0, fx1 = sx(x0, z0), sx(x1, z0)
    fy0, fy1 = sy(y1, z0), sy(y0, z0)
    top = N_FRONT_T * tcol * (1 - hz * 0.85) + HAZE_N * (hz * 0.85)
    bot = N_FRONT_B * tcol * (1 - hz * 0.85) + HAZE_N * 1.15 * (hz * 0.85)
    if z0 > Z0:
        dist = math.hypot(xc, z0 - Z0) + 1
        cL = lc * I * (0.05 + 0.22 * (z0 - Z0) / dist)     # 폭심 쪽(아래)이 밝은 세로 그라디언트
        lgrad = 1
    else:
        cL = lc * I * 0.02
        lgrad = 0
    floor_h = rng.uniform(3.6, 4.8)
    col_w = rng.uniform(3.8, 7.0)
    nf = max(1, int((y1 - y0) / floor_h))
    nc = max(1, int((x1 - x0) / col_w))
    wp = (fx0, fy0, (fy1 - fy0) / nf, (fx1 - fx0) / nc, rng.uniform(0.10, 0.5),
          rng.uniform(0.22, 0.34), rng.uniform(0.2, 0.3), 1.0 - 0.75 * hz,
          float(np.searchsorted(np.cumsum(WIN_P), rng.random())), seed & 0xFFFFF)
    rr = (1, fy0, 0.0, lc * I * min(1.0, 2.2 * I) * 1.4) if (rim and z0 < Z0) else None
    faces.append(dict(kind=2, poly=np.array([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)]),
                      cNt=top, cNb=bot, cB=cBh, cL=cL, lgrad=lgrad, z=z0, d=d, fy0=fy0, fy1=fy1, wp=wp,
                      rim=rr))
    return faces


RIM_HOT = np.array([1.30, 1.12, 0.92], f32)      # 역광 림: 가장자리 백열 (청백 광원색에 곱함)
RIM_ORG = np.array([1.25, 0.58, 0.24], f32)      #          안쪽 주황 띠


def _rim_light(cv, x0, x1, y0, y1, col, ws, axis, sgn):
    """역광 림 2톤: 얇은 백열 선 + (가까운 빌딩일수록 넓은) 주황 띠. axis=0 세로 림(옆면), 1 가로 림(윗면).
    sgn 은 띠가 안쪽으로 넓어지는 방향."""
    _add_rect_light(cv, x0, x1, y0, y1, col * RIM_HOT)
    if ws > 1.05:
        if axis == 0:
            w = (x1 - x0) * (ws - 1.0) * 1.6
            xa, xb = (x1, x1 + w) if sgn > 0 else (x0 - w, x0)
            _add_rect_light(cv, xa, xb, y0, y1, col * RIM_ORG * 0.7)
        else:
            w = (y1 - y0) * (ws - 1.0) * 1.6
            _add_rect_light(cv, x0, x1, y1, y1 + w, col * RIM_ORG * 0.7)


def _draw_box(cv, x0, x1, z0, z1, y0, y1, d, seed, rim=True, win_min=9.0):
    """박스 하나를 안티앨리어스로 캔버스에 그린다 (근경/히어로용)."""
    ws = float(np.clip(820.0 / max(z0, 1.0), 1.0, 3.2))        # 가까울수록 림이 넓다
    for fc in _box_faces(x0, x1, z0, z1, y0, y1, d, seed, rim, win_min):
        if fc["kind"] != 2:
            mm = _poly_mask_local(cv, [fc["poly"]])
            if mm is None:
                continue
            _put(cv, mm, _pack(fc["cNt"], fc["cB"], fc["cL"]), fc["z"], d, fc["kind"])
            r = fc["rim"]
            if r is not None and r[0] == 3:
                ex = r[1]
                if r[2] > 0:
                    _rim_light(cv, ex - 1.2 * ws, ex + 0.2, fc["poly"][1][1], fc["poly"][0][1], r[3], ws, 0, -1)
                else:
                    _rim_light(cv, ex - 0.2, ex + 1.2 * ws, fc["poly"][1][1], fc["poly"][0][1], r[3], ws, 0, 1)
            elif r is not None and r[0] == 2:
                P = fc["poly"]
                _rim_light(cv, P[3][0], P[2][0], r[1] - 0.4, r[1] + 1.2 * ws, r[3], ws, 1, 1)
            continue
        fx0, fy0 = fc["poly"][0]
        fx1, fy1 = fc["poly"][2]
        mm = _rect_mask(cv, fx0, fx1, fy0, fy1)
        if mm is None:
            continue
        m, ix, iy = mm
        hp = m.shape[0]
        t0 = (iy + cv.oy + 0.5 - fy0) / max(fy1 - fy0, 1)
        t1 = (iy + cv.oy + hp - 0.5 - fy0) / max(fy1 - fy0, 1)
        tt = np.clip(np.linspace(t0, t1, hp, dtype=f32), 0, 1)[:, None, None] ** 2.2
        cN = fc["cNt"] * (1 - tt) + fc["cNb"] * tt
        cL = fc["cL"][None, None, :] * (0.25 + 0.75 * tt ** 0.6) if fc["lgrad"] else fc["cL"]
        cv.wp.append(fc["wp"])
        _put(cv, mm, _pack(cN, fc["cB"], cL), fc["z"], d, 2, len(cv.wp) - 1)
        if fc["rim"] is not None:
            _rim_light(cv, fx0, fx1, fy0, fy0 + 1.7 * ws, fc["rim"][3], ws, 1, 1)


def _draw_boxes_id(cv, boxes):
    """원경/중경 박스들을 면 ID 버퍼(cv2 C 루프)로 한 번에 래스터화한 뒤 LUT 로 색칠 (AA 없음)."""
    idm = np.zeros((H, W), np.int32)
    faces = []
    for (x0, x1, z0, z1, hgt, dn, seed) in boxes:
        for fc in _box_faces(x0, x1, z0, z1, 0.0, hgt, dn, seed):
            P = fc["poly"]
            if P[:, 0].max() < 0 or P[:, 0].min() >= W or P[:, 1].max() < 0 or P[:, 1].min() >= H:
                continue
            faces.append(fc)
            cv2.fillConvexPoly(idm, np.round(P * 16).astype(np.int32), len(faces), cv2.LINE_8, 4)
    if not faces:
        return
    nfc = len(faces)
    kind = np.array([fc["kind"] for fc in faces], np.uint8)
    cNt = np.array([fc["cNt"] for fc in faces], f32)
    cNb = np.array([fc["cNb"] for fc in faces], f32)
    cB = np.array([fc["cB"] for fc in faces], f32)
    cL = np.array([fc["cL"] for fc in faces], f32)
    lgrad = np.array([fc["lgrad"] for fc in faces], bool)
    z = np.array([fc["z"] for fc in faces], f32)
    d = np.array([fc["d"] for fc in faces], f32)
    fy0 = np.array([fc["fy0"] for fc in faces], f32)
    fy1 = np.array([fc["fy1"] for fc in faces], f32)
    rt = np.array([fc["rim"][0] if fc["rim"] else 0 for fc in faces], np.uint8)
    ra = np.array([fc["rim"][1] if fc["rim"] else 0.0 for fc in faces], f32)
    rs = np.array([fc["rim"][2] if fc["rim"] else 0.0 for fc in faces], f32)
    rc = np.array([fc["rim"][3] if fc["rim"] else np.zeros(3, f32) for fc in faces], f32)
    wpi = np.full(nfc, -1, np.int32)
    for i, fc in enumerate(faces):
        if fc["wp"] is not None:
            cv.wp.append(fc["wp"])
            wpi[i] = len(cv.wp) - 1
    Pix = np.flatnonzero(idm)
    f = idm.ravel()[Pix] - 1
    ys = (Pix // W).astype(f32) + 0.5
    xs = (Pix % W).astype(f32) + 0.5
    tt = np.clip((ys - fy0[f]) / np.maximum(fy1[f] - fy0[f], 1), 0, 1) ** 2.2
    Nc = cNt[f] * (1 - tt[:, None]) + cNb[f] * tt[:, None]
    Lc = cL[f] * np.where(lgrad[f], 0.25 + 0.75 * tt ** 0.6, 1.0)[:, None]
    rf = rt[f]
    rim = (((rf == 1) & (ys - fy0[f] < 1.7)) | ((rf == 2) & (ys - ra[f] < 1.6)) |
           ((rf == 3) & (np.abs(xs - ra[f] + rs[f] * 0.5) < 0.9)))
    Lc += rc[f] * rim[:, None]
    C2 = cv.C.reshape(-1, 12)
    C2[Pix, 0:3] = Nc
    C2[Pix, 3:6] = cB[f]
    C2[Pix, 6:9] = Lc
    C2[Pix, 9:12] = 0.0
    cv.A.ravel()[Pix] = 1.0
    cv.Z.ravel()[Pix] = z[f]
    cv.D.ravel()[Pix] = d[f]
    cv.K.ravel()[Pix] = kind[f]
    cv.I.ravel()[Pix] = wpi[f]


def _lamps(_unused, rng):
    lampc = np.zeros((H, W, 3), f32)

    def lamp(X_, Z_, c, s=1.0):
        if Z_ < 250:
            return
        px, py = sx(X_, Z_), sy(6.0, Z_)
        if not (0 <= px < W and HZ < py < H):
            return
        r = max(0.6, 2200.0 / Z_) * s
        cv2.circle(lampc, (int(px * 16), int(py * 16)), int(r * 16), tuple(float(v) for v in c), -1,
                   cv2.LINE_AA, 4)

    for zz in np.arange(320.0, 7000.0, 34.0):
        for xx in (-AVE + 5, AVE - 5):
            lamp(xx, zz, SODIUM * 1.6)
        for lane in range(3):
            if rng.random() < 0.5:
                lamp(rng.uniform(-AVE + 12, -6), zz + rng.uniform(0, 30), hexc("#ff3020") * 1.4, 0.5)
            if rng.random() < 0.5:
                lamp(rng.uniform(6, AVE - 12), zz + rng.uniform(0, 30), np.array([1.6, 1.55, 1.4], f32), 0.55)
    for xx in np.arange(-5000.0, 5000.0, 30.0):
        for zz in (RIVER[0] - 6, RIVER[1] + 6):
            lamp(xx, zz, SODIUM * 1.4, 0.8)
    for zr in np.arange(300.0, 6000.0, P_BLOCK):
        zz = zr + P_BLOCK - STREET / 2
        for xx in np.arange(-4000.0, 4000.0, 52.0):
            if rng.random() < 0.5:
                lamp(xx + rng.uniform(-8, 8), zz, SODIUM * rng.uniform(0.8, 1.5), 0.7)
    for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        lamp(math.cos(a) * PLAZA * 0.62, Z0 + math.sin(a) * PLAZA * 0.62, CYAN * 1.3, 0.8)
        lamp(math.cos(a) * PLAZA * 0.95, Z0 + math.sin(a) * PLAZA * 0.95, SODIUM * 1.2, 0.7)
    small = cv2.resize(lampc, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5) * 0.8 + cv2.GaussianBlur(small, (0, 0), 8) * 0.5,
                      (W, H), interpolation=cv2.INTER_LINEAR)
    return lampc + halo


def _ground():
    """지면 레이어 (N, B, L, W) 와 Z/D/K 맵 — 전부 연속 배열로 만든다."""
    Zg, Xg, Dg = _ground_maps()
    r0 = int(np.nonzero(Zg[:, 0] < 1e6)[0][0])
    Z = np.broadcast_to(Zg[r0:], (H - r0, W))
    X = Xg[r0:]
    D = Dg[r0:]
    hz = (1 - np.exp(-np.maximum(Z - 300, 0) / 5200.0)).astype(f32)
    fade = np.clip(1 - Z / 5000.0, 0, 1)
    pxz = Z * Z / (F * HC)
    pxx = Z / F
    xm = (np.abs(X) - AVE) % P_BLOCK
    road_x = np.clip(1 - np.abs(xm - (P_BLOCK - STREET / 2)) / (STREET / 2 + pxx), 0, 1)
    road_x = np.where(np.abs(X) < AVE, f32(1.0), road_x)
    zm = (Z - 300.0) % P_BLOCK
    road_z = np.clip((STREET + pxz - np.abs(zm - (P_BLOCK - STREET / 2)) * 2) / (STREET + pxz), 0, 1)
    road = np.maximum(road_x, road_z * fade).astype(f32)
    river = ((Z > RIVER[0]) & (Z < RIVER[1])).astype(f32)
    rv = cv2.GaussianBlur(river, (0, 0), 0.8)
    plaza = np.clip((PLAZA - D) / 12.0, 0, 1).astype(f32)
    pr = np.zeros_like(plaza)
    for rr, c in ((0.62, 1.0), (0.95, 0.75), (0.3, 0.7)):
        pr += np.exp(-((D - PLAZA * rr) / 5.0) ** 2) * c
    pr *= plaza
    img = np.empty((H - r0, W, 3), f32)
    for c in range(3):
        v = hexc("#100c1e")[c] + road * (0.10, 0.06, 0.09)[c]
        v = v * (1 - rv) + (0.025, 0.035, 0.09)[c] * rv
        v = v * (1 - plaza) + hexc("#231d38")[c] * plaza + pr * (0.12, 0.08, 0.14)[c]
        img[..., c] = v * (1 - hz * 0.85) + HAZE_N[c] * hz * 0.85
    N = np.zeros((H, W, 3), f32)
    N[r0:] = img
    B = np.zeros((H, W, 3), f32)
    for c in range(3):
        v = B_SIL[c] * 1.3 * (1 - hz * 0.6) + B_HAZE[c] * hz * 0.6
        B[r0:, :, c] = v * (1 - rv) + (0.01, 0.015, 0.04)[c] * rv
    L = np.zeros((H, W, 3), f32)
    I = (1.0 / (1.0 + (D / 650.0) ** 2) * (0.30 + 0.5 * plaza)).astype(f32)
    t = np.clip(D / 2200.0, 0, 1)
    for c in range(3):
        L[r0:, :, c] = (LIGHT_NEAR[c] * (1 - t) + LIGHT_FAR[c] * t) * I
    rng = np.random.default_rng(404)
    Wl = _lamps(None, rng)
    refl = np.zeros((H, W, 3), f32)
    rr = int(sy(0, RIVER[1])), int(sy(0, RIVER[0]))
    for _ in range(300):
        x = rng.uniform(0, W)
        c = WIN_COLS[rng.integers(len(WIN_COLS))] * rng.uniform(0.3, 0.9)
        y0 = rng.uniform(rr[0], rr[0] + 6)
        y1 = y0 + rng.uniform(6, rr[1] - rr[0])
        cv2.line(refl, (int(x * 16), int(y0 * 16)), (int(x * 16), int(y1 * 16)),
                 tuple(float(v) for v in c), 1, cv2.LINE_AA, 4)
    refl = cv2.GaussianBlur(refl, (0, 0), 0.8)
    rvf = np.zeros((H, W), f32)
    rvf[r0:] = rv
    Wl += refl * rvf[..., None]
    A = np.zeros((H, W), f32)
    A[r0:] = 1.0
    Zm = np.full((H, W), 1e7, f32)
    Zm[r0:] = Z
    Dm = np.full((H, W), 1e7, f32)
    Dm[r0:] = D
    K = np.zeros((H, W), np.uint8)
    K[r0:] = np.where(river > 0.5, 6, 1).astype(np.uint8)
    return N, B, L, Wl, A, Zm, Dm, K, rvf


def _shadows(blds):
    """폭심(높이 YS)의 점광원이 만드는 지면 그림자 마스크 (반해상도)."""
    YS = 150.0
    s = 2
    m = np.zeros((H // s, W // s), np.uint8)
    zn = 200.0
    for (x0, x1, z0, z1, hgt, dn, seed) in blds:
        if z0 > 5200:
            continue
        foot = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        pts = list(foot)
        for (px, pz) in foot:
            vx, vz = px, pz - Z0
            L_ = math.hypot(vx, vz) + 1e-6
            if hgt < YS * 0.9:
                ln = min(L_ * (YS / (YS - hgt) - 1), 2600.0)
            else:
                ln = 2600.0
            pts.append((px + vx / L_ * ln, pz + vz / L_ * ln))
        hull = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2).astype(np.float64)
        out = []
        n = len(hull)
        for i in range(n):
            a, b = hull[i], hull[(i + 1) % n]
            ina, inb = a[1] >= zn, b[1] >= zn
            if ina:
                out.append(a)
            if ina != inb:
                t = (zn - a[1]) / (b[1] - a[1])
                out.append(a + (b - a) * t)
        if len(out) < 3:
            continue
        Q = np.array([(sx(p[0], p[1]) / s, sy(0.0, p[1]) / s) for p in out])
        Q[:, 1] = np.minimum(Q[:, 1], H / s + 50)
        cv2.fillConvexPoly(m, np.round(Q * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
    return cv2.resize(m.astype(f32) / 255.0, (W, H), interpolation=cv2.INTER_LINEAR)


def _far_skyline(cv):
    """지평선 너머 초고층 아콜로지 실루엣 (하늘을 배경으로)."""
    rng = np.random.default_rng(909)
    ant = np.zeros((H, W), np.uint8)
    lights = np.zeros((H, W, 3), f32)
    for i in range(18):
        z = rng.uniform(9500, 15000)
        x = rng.uniform(-1.0, 1.0) * 0.72 * z
        w = rng.uniform(140, 320)
        hgt = rng.uniform(380, 690)
        d = math.hypot(x, z - Z0)
        steps = int(rng.integers(2, 4))
        y = 0.0
        ww = w
        for s_ in range(steps):
            yt = y + hgt * (0.55 if s_ == 0 else 0.45 / (steps - 1))
            _draw_box(cv, x - ww / 2, x + ww / 2, z, z + ww, y, yt, d, int(rng.integers(1 << 30)),
                      rim=False, win_min=6.0)
            y = yt
            ww *= rng.uniform(0.55, 0.8)
        px, py = sx(x, z), sy(y + 30, z)
        cv2.line(ant, (int(px * 16), int(sy(y, z) * 16)), (int(px * 16), int(py * 16)), 255, 1,
                 cv2.LINE_AA, 4)
        cv2.circle(lights, (int(px * 16), int(py * 16)), 24, (2.2, 0.25, 0.2), -1, cv2.LINE_AA, 4)
    m = (ant.astype(f32) / 255.0)[..., None]
    cv.C[..., 0:3] = cv.C[..., 0:3] * (1 - m) + m * np.array([0.25, 0.12, 0.25], f32)
    cv.C[..., 9:12] += lights


Z_AA = 1500.0          # 이보다 가까운 빌딩만 안티앨리어스 경로로


@lru_cache(maxsize=1)
def _city():
    cv = _Cv()
    N, B, L, Wl, cv.A, cv.Z, cv.D, cv.K, river = _ground()
    cv.C = np.concatenate([N, B, L, Wl], axis=2)
    del N, B, L, Wl
    _far_skyline(cv)
    blds = _gen_buildings()
    sh = _shadows(blds)
    cv.C[..., 6:9] *= (1 - sh * 0.93)[..., None]
    order = sorted(range(len(blds)), key=lambda i: (-blds[i][2], -abs(blds[i][0] + blds[i][1])))
    # 원경·중경 (Z>1500): 면 ID 래스터 → 근경: 안티앨리어스 (화가 알고리즘 순서 유지)
    _draw_boxes_id(cv, [blds[i] for i in order if blds[i][2] > Z_AA])
    for i in order:
        x0, x1, z0, z1, hgt, dn, seed = blds[i]
        if z0 <= Z_AA:
            _draw_box(cv, x0, x1, z0, z1, 0.0, hgt, dn, seed)
    # 먼 정면들: 창문 반점 (한 번에)
    N = np.ascontiguousarray(cv.C[..., 0:3])
    B = np.ascontiguousarray(cv.C[..., 3:6])
    L = np.ascontiguousarray(cv.C[..., 6:9])
    Wn = np.ascontiguousarray(cv.C[..., 9:12])
    del cv.C
    _windows_vec(cv, Wn)
    painted_texture(N, 0.08, seed=501, scale=2.0, stretch=7.0, angle=math.pi / 2)  # 세로 붓결
    painted_texture(B, 0.10, seed=501, scale=2.0, stretch=7.0, angle=math.pi / 2, ox=431.0, oy=173.0)
    N = cv2.GaussianBlur(N, (0, 0), 0.55)
    haze = np.exp(-((YY - HZ - 18) / 70.0) ** 2) * 0.22
    N += (haze * cv.A)[..., None] * np.array([0.55, 0.18, 0.30], f32)
    arr = np.array([b[:6] for b in blds], np.float64)
    return dict(N=N, B=B, L=L, W=Wn, A=cv.A, Z=cv.Z, D=cv.D, K=cv.K, river=river,
                blds=arr, bseed=np.array([b[6] for b in blds], np.int64))


# 히어로 타워 (오리지널 실루엣): 이름, 중심 X, 정면 Z, 부품 [(폭, 깊이, y0, y1, 후퇴)], 안테나 높이
HEROES = [
    ("T2", 450.0, 1480.0, [(118, 100, 0, 120, 0), (92, 84, 120, 400, 6), (70, 64, 400, 452, 16),
                           (28, 28, 452, 480, 30)], 560.0),
    ("T1", -330.0, 1170.0, [(124, 104, 0, 150, 0), (98, 86, 150, 470, 8), (76, 66, 470, 528, 18),
                            (34, 30, 528, 560, 34)], 650.0),
    ("T4", 262.0, 440.0, [(76, 62, 0, 300, 0), (58, 50, 300, 336, 6)], 380.0),
    ("T5", -232.0, 404.0, [(64, 56, 0, 236, 0), (46, 40, 236, 262, 6)], 300.0),
]


@lru_cache(maxsize=1)
def _hero_sprites():
    """히어로 타워를 각자 국소 캔버스의 스프라이트로."""
    out = []
    for hi, (name, xc, z0, parts, ant) in enumerate(HEROES):
        pts = []
        for (w, dpt, y0, y1, sb) in parts:
            for X in (xc - w / 2, xc + w / 2):
                for Z in (z0 + sb, z0 + sb + dpt):
                    for Y in (y0, y1):
                        pts.append((sx(X, Z), sy(Y, Z)))
        pts.append((sx(xc, z0), sy(ant, z0)))
        P = np.array(pts)
        x0b = int(max(0, math.floor(P[:, 0].min() - 12)))
        x1b = int(min(W, math.ceil(P[:, 0].max() + 12)))
        y0b = int(max(0, math.floor(P[:, 1].min() - 12)))
        y1b = int(min(H, math.ceil(P[:, 1].max() + 12)))
        cv = _Cv(x0b, y0b, x1b - x0b, y1b - y0b)
        dmin = 1e9
        for (w, dpt, y0, y1, sb) in parts:
            nx = min(max(0.0, xc - w / 2), xc + w / 2)
            dmin = min(dmin, math.hypot(nx, z0 + sb + dpt / 2 - Z0))
        for (w, dpt, y0, y1, sb) in parts:
            _draw_box(cv, xc - w / 2, xc + w / 2, z0 + sb, z0 + sb + dpt, y0, y1, dmin,
                      int(hash01(hi, y0) * 1e9))
        top = parts[-1][3]
        px = sx(xc, z0 + parts[-1][4] + 10) - x0b
        lw = max(1.0, 3.0 * 440.0 / z0)
        m8 = np.zeros((cv.h, cv.w), np.uint8)
        cv2.line(m8, (int(px * 16), int((sy(top, z0) - y0b) * 16)),
                 (int(px * 16), int((sy(ant, z0) - y0b) * 16)), 255, int(round(lw)), cv2.LINE_AA, 4)
        mf = m8.astype(f32) / 255.0
        c9 = _pack(hexc("#2a1d40"), B_SIL, RIM * 0.5)
        cv.C[..., :9] = cv.C[..., :9] * (1 - mf[..., None]) + mf[..., None] * c9
        cv.A = np.maximum(cv.A, mf)
        cv.Z = np.where(mf > 0.5, z0, cv.Z).astype(f32)
        bl = (int(px * 16), int((sy(ant, z0) - y0b) * 16))
        lt = np.zeros((cv.h, cv.w, 3), f32)
        cv2.circle(lt, bl, int(16 * max(2.0, 6.0 * 440 / z0)), (2.6, 0.3, 0.25), -1, cv2.LINE_AA, 4)
        cv.C[..., 9:12] += lt
        N = np.ascontiguousarray(cv.C[..., 0:3])
        painted_texture(N, 0.07, seed=501, scale=1.5, stretch=6, angle=math.pi / 2, ox=97.0 * hi, oy=53.0 * hi)
        Wn = np.ascontiguousarray(cv.C[..., 9:12])
        _windows_vec(cv, Wn)
        br = np.random.default_rng(700 + hi).random((cv.h // 4 + 1, cv.w // 9 + 1)).astype(f32)
        blk = np.repeat(np.repeat(br, 4, 0), 9, 1)[:cv.h, :cv.w]          # 층·칸 단위 소등 순서
        out.append(dict(name=name, box=(x0b, y0b, x1b, y1b), N=N, blk=blk,
                        B=np.ascontiguousarray(cv.C[..., 3:6]),
                        L=np.ascontiguousarray(cv.C[..., 6:9]),
                        W=Wn, A=cv.A, Z=cv.Z, d=dmin, z=z0,
                        xc=sx(xc, z0), base_y=sy(0, z0), top_y=sy(top, z0),
                        half_w=F * parts[0][0] / 2 / z0, side=-1 if xc < 0 else 1))
    return out


# ---------------------------------------------------------------------------
# 프레임 공통 헬퍼
# ---------------------------------------------------------------------------
def _small_glow(mask, sigmas=(6, 24, 80), weights=(0.6, 0.35, 0.25), down=4):
    """마스크/이미지의 큰 글로우를 저해상도에서 계산."""
    sw, sh = W // down, H // down
    sm = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(sm)
    for s, wgt in zip(sigmas, weights):
        acc += cv2.GaussianBlur(sm, (0, 0), s / down) * wgt
    return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def _blackout_R(lf):
    if lf < 8:
        return -200.0
    return 4600.0 * float(ease_in((lf - 8) / 5.0, 2.0))


def _seed_I(lf):
    return 0.45 + 0.9 * math.exp(-((lf - 6) / 1.1) ** 2) + 1.2 * math.exp(-((lf - 10) / 1.1) ** 2) + \
        0.6 * float(clamp((lf - 11) / 2.0))


def _composite_hero(img, Zf, Df, hs, col, sel=None, z_override=None):
    x0, y0, x1, y1 = hs["box"]
    a = hs["A"] if sel is None else hs["A"] * sel
    zt = hs["z"] if z_override is None else z_override
    vis = a * (Zf[y0:y1, x0:x1] > zt)
    reg = img[y0:y1, x0:x1]
    reg *= (1 - vis[..., None])
    reg += vis[..., None] * col
    m = vis > 0.5
    np.copyto(Zf[y0:y1, x0:x1], f32(zt), where=m)
    np.copyto(Df[y0:y1, x0:x1], f32(hs["d"]), where=m)
    return vis


def _star(img, x, y, length, width, col, angle=0.0):
    """십자 크로스 필터 광채 (가산)."""
    for k, (ln, wd) in enumerate(((length, width), (length * 0.55, width * 0.8),
                                  (length * 0.22, width * 0.6), (length * 0.22, width * 0.6))):
        a = math.radians(angle + (0, 90, 45, 135)[k])
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        c = np.array([x, y])
        poly = np.array([c - d * ln, c - n * wd, c + d * ln, c + n * wd])
        m = np.zeros((H, W), np.uint8)
        x0 = int(max(0, min(poly[:, 0]) - 2))
        x1 = int(min(W, max(poly[:, 0]) + 3))
        y0 = int(max(0, min(poly[:, 1]) - 2))
        y1 = int(min(H, max(poly[:, 1]) + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        mm = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mm, _fix(poly, x0, y0), 255, cv2.LINE_AA, _SHIFT)
        mf = cv2.GaussianBlur(mm.astype(f32) / 255.0, (0, 0), 1.0)
        img[y0:y1, x0:x1] += mf[..., None] * col


# ---------------------------------------------------------------------------
# 폭발 전 (f2–13)
# ---------------------------------------------------------------------------
def _pre(ctx):
    lf = ctx.lframe
    S = _city()
    sky = _sky()[0]
    Zg, Xg, Dg = _ground_maps()
    A = S["A"][..., None]
    Rb = _blackout_R(lf)
    on = smoothstep(Rb - 160, Rb + 160, S["D"])
    Is = _seed_I(lf)
    img = sky * (1 - A) + S["N"] + S["W"] * on[..., None] + S["L"] * (0.16 * Is)
    Zf = S["Z"].copy()
    Df = S["D"].copy()
    for hs in _hero_sprites():
        onh = 1.0 if hs["d"] > Rb + 100 else 0.0
        col = hs["N"] + hs["W"] * onh + hs["L"] * (0.16 * Is)
        _composite_hero(img, Zf, Df, hs, col)
    # 씨앗 광점
    px, py = CX, sy(SEED_Y, Z0)
    k_dark = float(clamp((lf - 11) / 2.0))
    img *= 1 - 0.55 * k_dark * (1 - 0.0)
    # 파문 (지면)
    for t0 in (6, 10):
        if t0 <= lf < t0 + 5:
            Rp = 40 + 85 * (lf - t0)
            g = np.exp(-((Dg - Rp) / (10 + 4 * (lf - t0))) ** 2) * (1 - (lf - t0) / 5.0)
            gm = (S["K"] == 1) | (S["K"] == 6)
            img += (g * gm)[..., None] * (RIM * 1.4)
    rd = np.hypot(XX - px, YY - py)
    core_r = 7.0 if lf < 12 else 4.0
    halo = (np.exp(-(rd / (core_r * 2.2)) ** 2) * 4.0 * Is
            + np.exp(-rd / 40.0) * 0.9 * Is + np.exp(-rd / 180.0) * 0.28 * Is
            + np.exp(-rd / 520.0) * 0.10 * Is)
    img += halo[..., None] * np.array([0.85, 0.95, 1.0], f32)
    # 빛 기둥
    if lf >= 6:
        pk = float(clamp((lf - 6) / 4.0)) * (0.6 + 0.4 * Is)
        bw = 2.0 + 1.5 * Is
        beam = np.exp(-((XX - px) / bw) ** 2) * (YY < py) * np.clip((py - YY) / 60.0, 0, 1)
        img += (beam * pk * 2.2)[..., None] * np.array([0.8, 0.95, 1.0], f32)
        bg = np.exp(-np.abs(XX - px) / 16.0) * (YY < py) * np.exp(-np.maximum(py - YY, 0) / 500.0)
        img += (bg * pk * 0.16)[..., None] * RIM
    _star(img, px, py, 260 + 220 * Is, 2.2 + Is, np.array([1.6, 1.8, 2.0], f32) * (0.5 + 0.5 * Is))
    # 흰빛 수축 (f2–6)
    if lf <= 6:
        k = (lf - 1) / 5.0
        rc = 2300.0 * (1 - k) ** 1.6
        rays = radial_lines_mask(px, py, n=90, seed=77 + lf, r_in=rc * 0.4, r_jit=0.3,
                                 width=(0.004, 0.016))
        rays = cv2.GaussianBlur(rays, (0, 0), 2.0)
        m = 1 - smoothstep(rc * 0.55, rc + 80, rd)
        m = np.maximum(m, rays * (1 - smoothstep(rc, rc * 2.2 + 300, rd)) * 0.9 * (1 - 0.75 * k))
        wg = (1 - k) ** 2 * 0.45
        wmask = np.maximum(m * (1 - 0.2 * k), wg)
        img = img * (1 - wmask[..., None]) + WHITE_HOT * wmask[..., None]
    return img, Zf


# ---------------------------------------------------------------------------
# 폭발 (f14–51)
# ---------------------------------------------------------------------------
# 먼지 벽 로브 행: (반원당 덩어리 수, 중심 높이 비율, 반지름 비율, 바깥쪽 오프셋 비율)
SMK_ROWS = (
    (60, (0.42, 0.64), 0.36, 0.00),
    (78, (0.18, 0.36), 0.29, 0.22),
    (96, (-0.04, 0.10), 0.23, 0.44),
)
SMK_K = 9             # 덩어리당 원 (본체 1 + 혹 최대 8)
# 역광(앞쪽 벽): 구체 쪽 윤곽만 초승달 띠로 빛나고 몸통은 중간 톤, 반대쪽은 짙은 그림자
SMK_FRONT = dict(d_lit=0.15, g_hot=0.055, g_sh=0.34,
                 lit_hot=np.array([2.40, 2.05, 1.65], f32), lit_far=np.array([1.45, 0.52, 0.16], f32),
                 mid_top=np.array([0.165, 0.08, 0.17], f32), mid_bot=np.array([0.21, 0.085, 0.12], f32),
                 sh=np.array([0.055, 0.022, 0.085], f32), ink=hexc("#12060c"))
# 순광(구체 뒤쪽 벽): 멀고 엷다
SMK_BACK = dict(d_lit=0.20, g_hot=0.07, g_sh=0.16,
                lit_hot=np.array([0.98, 0.90, 1.05], f32), lit_far=np.array([0.90, 0.68, 0.72], f32),
                mid_top=np.array([0.25, 0.19, 0.32], f32), mid_bot=np.array([0.28, 0.18, 0.26], f32),
                sh=np.array([0.12, 0.075, 0.18], f32), ink=np.array([0.09, 0.05, 0.13], f32))
FIRE_LIT = np.array([2.4, 1.0, 0.36], f32)
FIRE_MID = np.array([0.48, 0.14, 0.09], f32)
FIRE_SH = np.array([0.20, 0.05, 0.07], f32)
SMK_HMAX = 170.0      # 덩어리 크기 상한용 벽 높이 (카메라 직전에서 공 몇 개로 화면을 덮지 않게)
SMK_YMAX = 300.0      # 쌓이는 높이 상한 (카메라 직전에서 벽이 솟아올라 화면 아래를 덮는다)


@lru_cache(maxsize=8)
def _smoke_params(seed):
    rows = []
    for ri, (n, yr, rbase, off) in enumerate(SMK_ROWS):
        rng = np.random.default_rng([seed, ri, 77])
        u = (np.arange(n) + rng.uniform(0.0, 1.0, n)) / n
        yc = rng.uniform(*yr, n)
        rf = rbase * np.exp(rng.normal(0, 0.28, n))
        plume = rng.random(n) < (0.14 if ri == 0 else 0.04)
        rf = np.where(plume, rf * 1.3, rf)
        yc = np.where(plume, yc + 0.10, yc)
        rows.append(dict(u=u, yc=yc, rf=rf, off=off, ph=rng.uniform(0, 6.283, n),
                         rise=rng.uniform(0.5, 1.5, n), nsub=rng.integers(3, SMK_K, n),
                         sub=rng.uniform(0, 1, (n, SMK_K, 3)), lw=rng.uniform(0.8, 1.3, n)))
    return rows


def _smoke_wall(img, Zf, Rw, h0, alpha, lf, seed, scx, scy, sr, back=False, lit=1.0):
    """충격파를 뒤따르는 먼지 벽 — 셀 애니 폭연 (2코마).
    원통 위에 고정된 덩어리(본체 원 + 구체 쪽 혹 원들의 합집합 = 콜리플라워 실루엣)를
    덩어리마다 4톤 면 ID 버퍼에 화가 순서로 그린다 (전부 cv2 C 루프):
      잉크   = 광원 반대로 밀린 큰 원들 (그림자 쪽 선이 굵다)
      밝은 띠 = 원들 그대로
      중간 톤 = 광원 반대로 dl 밀고 dl 줄인 원들 → 구체 쪽 윤곽만 혹을 따라 굽이치는 초승달 띠가 남음
      그림자  = 본체 원 안쪽, 광원 반대편에 접한 원
    """
    if alpha <= 0.01 or Rw <= 0:
        return
    lf2 = (lf // 2) * 2
    P = SMK_BACK if back else SMK_FRONT
    hy = min(h0, SMK_YMAX)
    h0 = min(h0, SMK_HMAX)
    params = _smoke_params(seed)
    prims = []
    for ri, p in enumerate(params):
        Rr = Rw + (-1 if back else 1) * p["off"] * h0
        th = math.pi * p["u"] if back else -math.pi + math.pi * p["u"]
        X = Rr * np.cos(th)
        Z = Z0 + Rr * np.sin(th)
        Yc = (p["yc"] + 0.006 * p["rise"] * lf2) * hy
        r = p["rf"] * h0 * (1 + 0.05 * np.sin(0.9 * lf2 + p["ph"]))
        Zs = np.maximum(Z, 300.0)          # 카메라 직전: 더 가까워지지 않고 솟아오르기만 (투영용)
        px = CX + F * X / Zs
        py = HZ + F * (HC - Yc) / Zs
        rp = np.minimum(F * r / Zs, 300.0)
        ok = (Z > 45) & (px + 1.7 * rp > 0) & (px - 1.7 * rp < W) & (py + 1.7 * rp > 0) & \
            (py - 1.7 * rp < H) & (rp > 0.8)
        for i in np.nonzero(ok)[0]:
            prims.append((float(Z[i]), ri, int(i), float(px[i]), float(py[i]), float(rp[i]), float(Yc[i])))
    if not prims:
        return
    prims.sort(key=lambda q: -q[0])
    nm = len(prims) + 1
    pal = np.zeros((nm * 5, 3), f32)
    zm = np.full(nm, 1e7, f32)
    circs = []
    for m, (Z, ri, i, px, py, rp, Yc) in enumerate(prims, 1):
        p = params[ri]
        ldx, ldy = scx - px, scy - 0.6 * sr - py         # 거대한 구체의 위쪽 절반에서 오는 빛
        ln = math.hypot(ldx, ldy)
        ldx, ldy = (0.0, -1.0) if ln < 1e-3 else (ldx / ln, ldy / ln)
        base = math.atan2(ldy, ldx)
        side = 1.0 if px > scx else -1.0
        roll = 0.009 * lf2 * side * (-1 if back else 1)      # 혹이 윤곽을 따라 굴러간다 (호 안에서 순환)
        cc = [(px, py, rp)]
        if ri == 0 and rp > 4:                           # 윗줄 덩어리는 아래로 몸통을 이어 공중에 뜨지 않게
            cc.append((px - ldx * 0.15 * rp, py + 0.72 * rp, 0.86 * rp))
        if rp > 4:
            nsub = int(p["nsub"][i]) + (3 if rp > 60 else 0)      # 가까운 큰 덩어리일수록 혹이 잘다
            big = 1.0 if rp <= 60 else 0.85
            for k in range(min(nsub, SMK_K - len(cc))):
                s0, s1, s2 = p["sub"][i, k]
                a = base + ((s0 + roll) % 1.0 - 0.5) * 3.8
                d = rp * (0.80 + 0.22 * s1)
                cc.append((px + d * math.cos(a), py + d * math.sin(a), rp * (0.20 + 0.22 * s2) * big))
        lw = float(np.clip(rp * 0.04, 1.0, 6.5)) * float(p["lw"][i])
        dlf = P["d_lit"] * (0.75 + 0.55 * float(p["sub"][i, -1, 0]))
        circs.append((m, ldx, ldy, lw, dlf * rp, cc))
        zm[m] = Z
        # 덩어리 팔레트 (잉크, 밝은 띠, 중간, 그림자)
        hf = float(np.clip(Yc / max(hy, 1.0), 0, 1))
        near = math.exp(-max(math.hypot(px - scx, py - scy) - sr, 0) / (0.9 * sr + 260.0))
        c_lit = (P["lit_far"] + (P["lit_hot"] - P["lit_far"]) * f32(0.2 * near)) * f32(lit ** 0.6)
        c_hot = P["lit_hot"] * f32(lit ** 0.6)
        c_mid = (P["mid_top"] * hf + P["mid_bot"] * (1 - hf)) * f32(0.75 + 0.25 * lit)
        c_sh = P["sh"]
        if not back and hf < 0.2:                       # 지면 가까이: 불길에 물든 덩어리
            c_lit, c_mid, c_sh = FIRE_LIT * f32(lit ** 0.6), FIRE_MID, FIRE_SH
        pal[5 * m - 4:5 * m + 1] = (P["ink"], c_hot, c_lit, c_mid, c_sh)
    allc = np.array([c for (_, _, _, _, _, cc) in circs for c in cc])
    ext = allc[:, 2] + 16
    x0 = int(max(0, math.floor((allc[:, 0] - ext).min())))
    x1 = int(min(W, math.ceil((allc[:, 0] + ext).max()) + 1))
    y0 = int(max(0, math.floor((allc[:, 1] - ext).min())))
    y1 = int(min(H, math.ceil((allc[:, 1] + ext).max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return
    idb = np.zeros((y1 - y0, x1 - x0), np.int32)

    def circ(cx_, cy_, rr, v):
        if rr > 0.3:
            cv2.circle(idb, (int(round((cx_ - x0) * 16)), int(round((cy_ - y0) * 16))), int(round(rr * 16)),
                       int(v), -1, cv2.LINE_8, 4)

    gs, gh = P["g_sh"], P["g_hot"]
    for (m, lx, ly, lw, dlf, cc) in circs:
        b = 5 * m
        for (c0, c1, rr) in cc:                       # 잉크
            circ(c0 - lx * lw * 0.7, c1 - ly * lw * 0.7, rr + lw, b - 4)
        for (c0, c1, rr) in cc:                       # 백열 가장자리 (전체)
            circ(c0, c1, rr, b - 3)
        rp = cc[0][2]
        for (c0, c1, rr) in cc:                       # 주황 띠: 광원 쪽에 폭 g_hot 의 백열 선을 남김
            g = min(rp * gh, 0.2 * rr)
            circ(c0 - lx * g * 0.5, c1 - ly * g * 0.5, rr - g * 0.5, b - 2)
        for (c0, c1, rr) in cc:                       # 그림자: 광원 쪽에 폭 g_l (덩어리 기준) 의 밝은 띠를 남김
            gl = min(dlf, 0.42 * rr)
            circ(c0 - lx * gl * 0.5, c1 - ly * gl * 0.5, rr - gl * 0.5, b)
        for (c0, c1, rr) in cc:                       # 중간 톤: 반대쪽에 폭 g_s 의 그림자 초승달을 남김
            gl, g2 = min(dlf, 0.42 * rr), rr * gs
            d = (g2 - gl) * 0.5
            circ(c0 + lx * d, c1 + ly * d, rr - (gl + g2) * 0.5, b - 1)
    col = pal[idb]
    zt = zm[(idb + 4) // 5]
    vis = ((idb > 0) & (Zf[y0:y1, x0:x1] > zt)).astype(f32) * f32(alpha)
    pc = cv2.GaussianBlur(col * vis[..., None], (0, 0), 0.6)
    vb = cv2.GaussianBlur(vis, (0, 0), 0.6)
    reg = img[y0:y1, x0:x1]
    reg *= 1 - vb[..., None]
    reg += pc
    np.copyto(Zf[y0:y1, x0:x1], zt, where=vis > 0.5)


def _puff(img, cx, cy, r, ldir, body, rimc, lw):
    """먼지 뭉치 원 하나 (잉크 테 + 몸통 + 광원 쪽 초승달 림)."""
    R_ = r + lw + 2
    x0 = int(max(0, math.floor(cx - R_)))
    x1 = int(min(W, math.ceil(cx + R_) + 1))
    y0 = int(max(0, math.floor(cy - R_)))
    y1 = int(min(H, math.ceil(cy + R_) + 1))
    if x1 <= x0 or y1 <= y0:
        return
    hh, ww = y1 - y0, x1 - x0
    m1 = np.zeros((hh, ww), np.uint8)
    m2 = np.zeros((hh, ww), np.uint8)
    m3 = np.zeros((hh, ww), np.uint8)
    c = (int(round((cx - x0) * 16)), int(round((cy - y0) * 16)))
    cv2.circle(m1, c, int((r + lw) * 16), 255, -1, cv2.LINE_AA, 4)
    cv2.circle(m2, c, int(r * 16), 255, -1, cv2.LINE_AA, 4)
    sh = (int(round((cx - x0 - ldir[0] * r * 0.22) * 16)), int(round((cy - y0 - ldir[1] * r * 0.22) * 16)))
    cv2.circle(m3, sh, int(r * 16), 255, -1, cv2.LINE_AA, 4)
    a1 = m1.astype(f32)[..., None] / 255
    a2 = m2.astype(f32)[..., None] / 255
    a3 = np.clip(a2 - m3.astype(f32)[..., None] / 255, 0, 1)
    reg = img[y0:y1, x0:x1]
    reg *= 1 - a1
    reg += a1 * INK
    reg *= 1 - a2
    reg += a2 * body
    reg *= 1 - a3
    reg += a3 * rimc


def _rush(img, lf, scx, scy, r):
    """충격파가 카메라를 덮친 뒤 (f40–47): 빛 속을 스쳐 지나가는 파편·먼지 셀 (2코마).
    가장자리엔 먼지 뭉치(터널 비네트), 가운데엔 빛, 그 사이로 늘어진(스미어) 파편 실루엣."""
    if lf < 40:
        return
    lf2 = (lf // 2) * 2
    diag = math.hypot(W, H) / 2
    # 동심원 빛 링 (화면 공간 충격파)
    for k, t0 in enumerate((40, 43)):
        a = lf - t0
        if 0 <= a < 6:
            rr = diag * (0.25 + 0.32 * a)
            g = np.exp(-((np.sqrt((XX - scx) ** 2 + (YY - scy) ** 2) - rr) / (14 + 10 * a)) ** 2)
            img += (g * (1 - a / 6.0) * 1.4)[..., None] * RIM
    # 가장자리 먼지 뭉치
    rng = np.random.default_rng(4242)
    n = 44
    th = rng.uniform(0, 2 * math.pi, n)
    start = rng.uniform(37, 45, n)
    rho0 = rng.uniform(0.60, 0.98, n)
    rad0 = rng.uniform(0.06, 0.13, n)
    items = []
    for i in range(n):
        a = lf2 - start[i]
        if a < 0:
            continue
        g = math.exp(0.30 * a)
        rho = rho0[i] * g * diag
        rad = rad0[i] * g * diag
        if rho - rad * 1.4 > diag * 1.1:
            continue
        items.append((rad, scx + math.cos(th[i]) * rho * 1.15, scy + math.sin(th[i]) * rho * 0.75, i))
    items.sort()
    for rad, cx, cy, i in items:
        r2 = np.random.default_rng([4243, i])
        k = int(r2.integers(5, 9))
        offs = r2.normal(0, 0.5, (k, 2)) * rad * np.array([1.3, 0.8])
        rads = rad * r2.uniform(0.32, 0.62, k)
        for j in np.argsort(offs[:, 1]):
            px, py = cx + offs[j, 0], cy + offs[j, 1]
            ld = np.array([scx - px, scy - py])
            ld /= np.linalg.norm(ld) + 1e-6
            _puff(img, px, py, rads[j], ld, np.array([0.13, 0.08, 0.12], f32) * r2.uniform(0.8, 1.2),
                  np.array([0.85, 0.95, 1.1], f32) * 1.2, max(2.0, rads[j] * 0.02))
    # 스미어 파편 (방사 방향으로 늘어진 실루엣)
    rng = np.random.default_rng(5151)
    n = 46
    th = rng.uniform(0, 2 * math.pi, n)
    start = rng.uniform(38, 46, n)
    rho0 = rng.uniform(0.08, 0.5, n)
    sz0 = rng.uniform(0.012, 0.035, n)
    for i in range(n):
        a = lf2 - start[i]
        if a < 0:
            continue
        g = math.exp(0.42 * a)
        rho = rho0[i] * g * diag
        sz = min(sz0[i] * g * diag, 64.0 + 20.0 * sz0[i] * 30)
        if rho - sz * 3 > diag * 1.2:
            continue
        u = np.array([math.cos(th[i]), math.sin(th[i]) * 0.8])
        u /= np.linalg.norm(u)
        nrm = np.array([-u[1], u[0]])
        c = np.array([scx, scy]) + u * rho
        r2 = np.random.default_rng([5152, i])
        shape = _rot(_debris_poly(r2) * np.array([1.0, r2.uniform(0.4, 0.8)]), r2.uniform(0, 6.28) + a * 0.6)
        body = shape * sz + c
        smear = min(sz * (1.2 + 0.5 * a), 240.0)
        # 스미어: 바깥→안쪽으로 가늘어지는 꼬리
        tail = np.array([c + nrm * sz * 0.32, c - nrm * sz * 0.32, c - u * smear - nrm * sz * 0.05,
                         c - u * smear + nrm * sz * 0.05])
        fill_poly(img, tail, np.array([0.38, 0.40, 0.52], f32), alpha=0.32)
        ld = -u
        if sz > 6:
            _cel_chunk(img, body, ld, B_SIL * 1.3, RIM * 1.3, max(1.2, sz * 0.05), int(i))
        else:
            fill_poly(img, body, B_SIL)


def _debris_poly(rng, n=None):
    n = n or int(rng.integers(4, 7))
    a = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.55, 1.0, n)
    return np.stack([np.cos(a) * r, np.sin(a) * r], 1)


def _rot(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return p @ np.array([[c, s], [-s, c]])


def _cel_chunk(img, poly, light_dir, base, lit, line_w, seed, boil=0.0):
    """조각 셀: 어두운 바탕 + 광원 쪽 초승달 림 + 잉크."""
    span = np.ptp(poly, axis=0).max()
    cen = poly.mean(0)
    ev = np.linalg.eigvalsh(np.cov((poly - cen).T) + 1e-6 * np.eye(2))
    minor = 2.0 * math.sqrt(max(ev[0], 1e-6))
    off = -np.asarray(light_dir) * min(max(1.3, span * 0.07), 0.22 * minor)
    # 바탕을 림 색으로, 광원 반대로 밀린 다각형을 그림자(어두운) 톤으로
    cel_shape(img, poly, lit, shade=base, shade_pts=poly + off, line=INK, line_w=line_w,
              boil=boil, seed=seed)


def _debris(img, lf, R, Rd, scx, scy):
    """충격파에 날아가는 빌딩 조각 (2코마)."""
    S = _city()
    B = S["blds"]
    seeds = S["bseed"]
    lf2 = (lf // 2) * 2
    d = B[:, 5]
    cand = np.nonzero((d < Rd) & (d > Rd - 520) & (d > R) & (B[:, 2] < Z0 + 150))[0]
    if len(cand) == 0:
        return
    # 화면에서 큰 것부터
    size = (B[cand, 1] - B[cand, 0]) / B[cand, 2] * B[cand, 4]
    cand = cand[np.argsort(-size)][:70]
    items = []
    for i in cand:
        x0, x1, z0, z1, hgt, dn = B[i]
        fh = hit_frame(dn)
        age = lf2 - fh + 1.0
        if age < 0 or age > 10:
            continue
        rng = np.random.default_rng([int(seeds[i]) & 0x7fffffff, 3])
        xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
        dl = math.hypot(xc, zc - Z0) + 1
        ux, uz = xc / dl, (zc - Z0) / dl
        wid = x1 - x0
        nch = 1 + int(min(3, hgt / 70))
        for j in range(nch):
            X = rng.uniform(x0, x1)
            Y = rng.uniform(0.35, 1.0) * hgt
            Z = rng.uniform(z0, z1)
            v = rng.uniform(26, 60)
            vy = rng.uniform(4, 16)
            a = age
            X += ux * v * a * (1 - 0.025 * a)
            Z += uz * v * a * (1 - 0.025 * a)
            Y += vy * a - 1.4 * a * a
            if Z < 60 or Y < -20:
                continue
            s = rng.uniform(0.18, 0.42) * min(wid, 70) * (1 + 0.04 * a)
            items.append((Z, X, Y, s, rng.uniform(0, 6.28) + rng.uniform(-0.5, 0.5) * a,
                          int(rng.integers(1 << 30))))
    items.sort(key=lambda t: -t[0])
    for (Z, X, Y, s, ang, sd) in items:
        px, py = sx(X, Z), sy(Y, Z)
        rp = F * s / Z
        if rp < 1.2 or not (-50 < px < W + 50 and -50 < py < H + 50):
            continue
        r2 = np.random.default_rng(sd)
        poly = _rot(_debris_poly(r2) * np.array([1.0, r2.uniform(0.45, 0.9)]), ang) * rp + (px, py)
        ldir = np.array([scx - px, scy - py])
        ldir /= np.linalg.norm(ldir) + 1e-6
        if rp < 5:
                    fill_poly(img, poly, B_SIL)
        else:
            _cel_chunk(img, poly, ldir, B_SIL * 1.4, WARM_LIT if sd & 1 else RIM * 1.1,
                       max(1.0, min(3.0, rp * 0.12)), sd)


def _lean_buildings(img, lf, R, Rd, scx, scy, Zf):
    """충격파를 맞은 큰 빌딩이 통째로 뒤로 젖혀지는 셀 실루엣 (2코마)."""
    S = _city()
    B = S["blds"]
    lf2 = (lf // 2) * 2
    d = B[:, 5]
    cand = np.nonzero((d < Rd) & (d > Rd - 260) & (d > R) & (B[:, 2] < Z0))[0]
    if len(cand) == 0:
        return
    hpx = B[cand, 4] * F / B[cand, 2]
    cand = cand[hpx > 40]
    cand = cand[np.argsort(-hpx[hpx > 40])][:5]
    for i in cand[np.argsort(-B[cand, 2])]:
        x0, x1, z0, z1, hgt, dn = B[i]
        fh = hit_frame(dn)
        age = lf2 - fh + 1.0
        if age < 0 or age > 4:
            continue
        bx0, bx1 = sx(x0, z0), sx(x1, z0)
        by, ty = sy(0, z0), sy(hgt, z0)
        xc = (bx0 + bx1) / 2
        sgn = 1.0 if xc > scx else -1.0
        ang = math.radians(6 + 9 * age) * sgn
        hh = by - ty
        cut = hh * (0.35 + 0.15 * hash01(i, 5))
        top_y = by - hh
        piv = np.array([xc, by - cut])
        poly = np.array([(bx0, by - cut), (bx1, by - cut), (bx1, top_y), (bx0, top_y)]) - piv
        c, s = math.cos(ang), math.sin(ang)
        poly = poly @ np.array([[c, s], [-s, c]]) + piv + (sgn * age * 3.0, age * 1.5)
        ldir = np.array([scx - xc, scy - piv[1]])
        ldir /= np.linalg.norm(ldir) + 1e-6
        _cel_chunk(img, poly, ldir, B_SIL * 1.3, WARM_LIT, max(1.2, min(3.5, hh * 0.015)), int(i))


def _sphere(img, Zf, Df, R, scx, scy, r, lf, I):
    x0 = int(max(0, math.floor(scx - r - 3)))
    x1 = int(min(W, math.ceil(scx + r + 3)))
    y0 = int(max(0, math.floor(scy - r - 3)))
    y1 = int(min(H, math.ceil(scy + r + 3)))
    full_mask = np.zeros((H, W), f32)
    if x1 <= x0 or y1 <= y0:
        return full_mask
    yy = YY[y0:y1]
    xx = XX[:, x0:x1]
    dd = np.sqrt((xx - scx) ** 2 + (yy - scy) ** 2)
    rho = dd / max(r, 1e-3)
    disk = np.clip(r - dd + 0.5, 0, 1)
    zfront = Z0 - R * np.sqrt(np.clip(1 - rho * rho, 0, 1))
    vis = disk * ((zfront < Zf[y0:y1, x0:x1]) | (Df[y0:y1, x0:x1] < R))
    # 순백 코어 → 포화된 청백 림 (셀 2톤처럼 경계가 비교적 또렷) → 가장자리 밝은 선
    t_rim = smoothstep(0.66, 0.84, rho)[..., None]
    col = CORE * (1 - t_rim) + RIMC * t_rim
    edge = np.exp(-((rho - 0.972) / 0.016) ** 2)[..., None]
    col = col + edge * np.array([0.9, 1.15, 1.25], f32)
    # 림 속 에너지 결 (2코마로 꿈틀)
    reg = img[y0:y1, x0:x1]
    reg *= 1 - vis[..., None]
    reg += vis[..., None] * col * (I / 2.4) ** 0.35
    full_mask[y0:y1, x0:x1] = vis
    m = vis > 0.5
    np.copyto(Zf[y0:y1, x0:x1], np.minimum(zfront, Zf[y0:y1, x0:x1]).astype(f32), where=m)
    return full_mask


def _wall_h(k, Rk):
    if k == 0:
        return 28.0 + 0.06 * Rk
    return 40.0 + 0.09 * Rk + 230.0 * float(clamp((Rk - 1250.0) / 550.0)) ** 2


def _vapor_ring(img, lf, scx, scy, r, I, front):
    """돔을 두른 응결 고리 구름 (윌슨 구름): 뒤쪽 호는 구체 뒤, 앞쪽 호는 구체 앞."""
    if not (16 <= lf <= 36):
        return
    a = (lf - 16) / 20.0
    rx = r * (1.25 + 0.75 * a)
    ry = rx * 0.20
    cy = scy + r * 0.18
    wb = rx * 0.11
    x0 = int(max(0, scx - rx - wb * 2))
    x1 = int(min(W, scx + rx + wb * 2))
    y0 = int(max(0, cy - ry - wb * 2))
    y1 = int(min(H, cy + ry + wb * 2))
    if x1 <= x0 or y1 <= y0:
        return
    yy = YY[y0:y1]
    xx = XX[:, x0:x1]
    e = np.sqrt(((xx - scx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    dist = (e - 1) * ry                                      # 대략 픽셀 단위
    nz = noise(x1 - x0, y1 - y0, 0.9, ox=x0 * 0.4 + (lf // 2) * 6, oy=y0 * 0.4, seed=57, beta=2.2)
    thick = wb * (0.45 + 0.9 * nz)
    m = np.clip(thick - np.abs(dist * 3.0) + 0.5, 0, 1)
    m *= (yy > cy) if front else (yy <= cy)
    env = math.sin(min(1.0, a * 1.4) * math.pi) ** 0.7 if a < 0.7 else max(0.0, (1 - a) / 0.3)
    m = m * env
    if m.max() <= 0:
        return
    lit = np.clip(0.55 - dist / (wb * 1.5), 0, 1)
    col = np.array([0.55, 0.70, 0.95], f32) * (0.6 + 0.9 * lit[..., None]) * (I / 2.4)
    reg = img[y0:y1, x0:x1]
    reg *= 1 - m[..., None] * 0.85
    reg += m[..., None] * 0.85 * col


@lru_cache(maxsize=1)
def _dissolve_noise():
    n1 = noise(W, H, 6.0, ox=517.0, seed=41, beta=2.8)
    n2 = noise(W, H, 1.2, ox=333.0, seed=501, beta=1.6, sx=0.3, sy=4.0)
    n = 0.7 * n1 + 0.3 * n2
    return ((n - n.min()) / (np.ptp(n) + 1e-6)).astype(f32)


def _climax_silhouettes(img, lf, scx, scy):
    """빛 속에서 바스러지는 도시 실루엣 (f40–47), 이후 f51 까지 빛이 화면을 계속 덮는다."""
    if not (40 <= lf <= 51):
        return
    S = _city()
    a = lf - 40
    # 빛의 범람: 배경을 순백~청백 빛으로
    rn0 = np.sqrt((XX - scx) ** 2 + (YY - scy) ** 2) / (0.5 * math.hypot(W, H))
    flood = float(clamp(0.72 + 0.07 * a))
    lightbg = WHITE_HOT * np.exp(-rn0 * 0.9)[..., None] + np.array([0.35, 0.6, 1.1], f32) * \
        (1 - np.exp(-rn0 * 0.9))[..., None]
    img *= 1 - flood
    img += lightbg * flood
    K = S["K"]
    sil = S["A"] * ((K >= 2) & (K <= 5))
    sc = 1.0 + 0.03 * a + 0.004 * a * a
    M = cv2.getRotationMatrix2D((scx, scy), 0.0, sc)
    sil = cv2.warpAffine(sil.astype(f32), M, (W, H), flags=cv2.INTER_LINEAR)
    # 빛의 전선: 구체 중심에서 퍼지는 들쭉날쭉한 경계가 실루엣을 먹어 들어감 (구멍 없이 또렷하게)
    N = _dissolve_noise()
    rn = np.sqrt((XX - scx) ** 2 + ((YY - scy) * 1.25) ** 2) / (0.5 * math.hypot(W, H))
    front = 0.05 + 0.115 * a + 0.006 * a * a
    field = rn + 0.16 * (N - 0.5)
    keep = np.clip((field - front) * 140.0 + 0.5, 0, 1)
    m = (sil * keep).astype(f32)
    edge = np.exp(-((field - front) / 0.012) ** 2) * sil
    er = cv2.erode(m, np.ones((5, 5), np.uint8))
    rim = np.clip(m - er, 0, 1)
    img *= 1 - m[..., None]
    img += m[..., None] * B_SIL
    img += rim[..., None] * RIM * 1.4 + edge[..., None] * np.array([1.6, 1.8, 2.2], f32)


def _arcs_embers(img, lf, scx, scy, r, I):
    """구체 가장자리를 기어 다니는 초능력 방전(청백 HDR) + 돔 밑동에서 튀는 불티 (2코마)."""
    if not (T_BANG <= lf <= 41):
        return
    lf2 = (lf // 2) * 2
    em = np.zeros((H, W), np.uint8)       # 방전
    fe = np.zeros((H, W), np.uint8)       # 불티
    rng = np.random.default_rng([6061, lf2])
    n_arc = 5 if lf < 30 else 7
    for _ in range(n_arc):
        a0 = rng.uniform(math.pi * 0.95, math.pi * 2.05)     # 위쪽 반구 위주
        span = rng.uniform(0.25, 0.7) * rng.choice([-1, 1])
        k = 26
        t = np.linspace(0, 1, k)
        ang = a0 + span * t
        rr = r * (1.0 + 0.10 * np.sin(t * math.pi) * rng.uniform(0.5, 1.6)) + rng.normal(0, r * 0.018, k)
        pts = np.stack([scx + np.cos(ang) * rr, scy + np.sin(ang) * rr], 1)
        cv2.polylines(em, [np.round(pts * 16).astype(np.int32)], False, 255, 2, cv2.LINE_AA, 4)
        # 가지
        j = int(rng.integers(5, k - 5))
        br = [pts[j]]
        for q in range(8):
            br.append(br[-1] + (pts[j] - (scx, scy)) / max(r, 1) * r * 0.025 + rng.normal(0, r * 0.012, 2))
        cv2.polylines(em, [np.round(np.array(br) * 16).astype(np.int32)], False, 255, 1, cv2.LINE_AA, 4)
    # 불티: 돔 밑동 타원에서 바깥·위로
    base_y = sy(0, Z0) - 6
    rng2 = np.random.default_rng(6062)
    n = 280
    born = rng2.uniform(T_BANG, 40, n)
    th = rng2.uniform(0, 2 * math.pi, n)
    v = rng2.uniform(9, 30, n)
    vy = rng2.uniform(6, 22, n)
    for i in range(n):
        a = lf2 - born[i]
        if a < 0 or a > 9:
            continue
        rad = r * 1.02 + 6
        x0_ = scx + math.cos(th[i]) * rad
        y0_ = base_y + math.sin(th[i]) * rad * 0.16
        def pos(t):
            return (x0_ + math.cos(th[i]) * v[i] * t * (1 + 0.04 * a),
                    y0_ - vy[i] * t + 0.9 * t * t + math.sin(th[i]) * v[i] * 0.2 * t)
        pa, pb = pos(a), pos(max(a - 1.1, 0))
        cv2.line(fe, (int(pa[0] * 16), int(pa[1] * 16)), (int(pb[0] * 16), int(pb[1] * 16)), 255,
                 2 + int(i % 3 == 0), cv2.LINE_AA, 4)
    emf = em.astype(f32) / 255.0
    fef = fe.astype(f32) / 255.0
    k = I / 2.4
    img += emf[..., None] * np.array([1.6, 2.2, 3.0], f32) * k
    img += _small_glow(emf, (4, 14, 40), (0.6, 0.4, 0.25), down=2)[..., None] * \
        np.array([0.25, 0.6, 1.2], f32) * k
    img += fef[..., None] * np.array([2.6, 1.4, 0.55], f32)
    img += _small_glow(fef, (3, 10), (0.7, 0.4), down=2)[..., None] * FIRE * 0.7


def _blast(ctx):
    lf = ctx.lframe
    S = _city()
    sky, dens, edge, Dc = _sky()
    Zg, Xg, Dg = _ground_maps()
    I = intensity(lf)
    R = sphere_R(lf)
    scx, scy, r, Yc = sphere_screen(R)
    Rd = destroy_R(lf)
    A = S["A"][..., None]
    rd = np.sqrt((XX - scx) ** 2 + (YY - scy) ** 2)

    # --- 하늘: 폭발광을 받는 구름 + 구멍 + 응결 링
    Rh = 3600.0 * float(ease_out((lf - 15) / 20.0, 1.6)) if lf >= 15 else 0.0
    hole = smoothstep(Rh - 260, Rh + 260, Dc) if Rh > 0 else np.ones_like(Dc)
    dn = dens * hole
    ring_c = np.exp(-((Dc - Rh) / 380.0) ** 2) * (Rh > 0) * (YY < HZ) * \
        (0.45 + 0.55 * np.exp(-np.abs(XX - scx) / 900.0))
    ci = 1.0 / (1.0 + (Dc / 1500.0) ** 2)
    skyb = sky * 0.42 * (1 - dn[..., None] * 0.4)
    skyb += (dn * (0.12 + 1.3 * edge) * ci * I * 0.30)[..., None] * HAZE_BLAST
    # --- 도시: 실루엣 + 폭발광
    city = S["B"] + S["L"] * I
    img = skyb * (1 - A) + city
    Zf = S["Z"].copy()
    Df = S["D"].copy()
    K = S["K"]

    # --- 히어로 타워
    heroes = _hero_sprites()
    hero_vis = []
    for hs in heroes:
        fh = hit_frame(hs["d"])
        col = hs["B"] + hs["L"] * I
        # 창문: 폭발광에 물들었다가 충격파가 오며 칸칸이 꺼진다 (2코마)
        thr = float(clamp(((lf // 2) * 2 - T_BANG + 2) / max(fh + 6 - T_BANG, 1.0)))
        if thr < 1:
            col = col + hs["W"] * ((hs["blk"] > thr) * 1.3)[..., None] * WIN_TINT
        if lf < fh:
            _composite_hero(img, Zf, Df, hs, col)
            hero_vis.append(None)
        else:
            hero_vis.append((hs, fh, col))

    # --- 파괴 영역: 충격파 안쪽은 먼지/열기
    Rw_lead = Rd
    inside = smoothstep(Rw_lead + 10, Rw_lead - 30, Df)
    if Rw_lead > R + 5:
        nz = noise(W, H, 3.0, ox=(lf // 2) * 13.0 + 700.0, oy=(lf // 2) * 7.0, seed=57, beta=2.2)
        hot = np.exp(-np.maximum(Dg - R, 0) / 90.0)[..., None]
        band = (nz > 0.53).astype(f32)[..., None]          # 셀 2톤 먼지
        dust = DUST_D * (0.75 + 0.5 * band) + hot * FIRE * (0.55 + 0.5 * band)
        gnd = smoothstep(HZ - 1.0, HZ + 6.0, YY)[..., None]
        icol = dust * gnd + skyb * (1 - gnd)
        img = img * (1 - inside[..., None]) + icol * inside[..., None]
        Zf = np.where(inside > 0.5, Zg, Zf).astype(f32)

    # --- 충격파 링 (지면)
    gvis = ((K == 1) | (K == 6) | (inside > 0.5)).astype(f32)
    for k, rg in enumerate(RINGS):
        Rk = ring_R(k, lf)
        if Rk is None:
            continue
        age = lf - rg[0]
        fade = math.exp(-max(age - rg[3] * 0.8, 0) / 3.0)
        wk = 6.0 + 0.008 * Rk
        g = np.exp(-((Dg - Rk) / wk) ** 2) + 0.18 * np.exp(-((Dg - Rk) / (wk * 6)) ** 2)
        g = g * gvis * rg[5] * fade
        img += g[..., None] * (RIM * 2.2)
        if rg[6]:
            # 충격파 전선이 빌딩 면을 때리는 순간의 섬광
            fl = np.exp(-((Df - Rk) / 18.0) ** 2) * (1 - gvis) * (Df < 1e6) * fade
            img += fl[..., None] * (RIM * 0.6)

    late = lf >= 42                 # 빛의 범람이 거의 다 덮는 구간: 보이지 않는 레이어 생략
    walls = lf < 40                 # 충격파 도달(f40~): 먼지 벽은 빛 속에 묻힌다
    # --- 먼 쪽 먼지 벽 (구체 뒤)
    for k in ((0, 2) if walls else ()):
        Rk = ring_R(k, (lf // 2) * 2)          # 작화(먼지 벽)는 2코마
        if Rk is None:
            continue
        al = math.exp(-max(lf - RINGS[k][0] - RINGS[k][3], 0) / 3.0)
        _smoke_wall(img, Zf, Rk - 35, _wall_h(k, Rk), al, lf, 3 + k, scx, scy, r, back=True, lit=I / 2.4)

    # --- 구체 (+ 응결 고리 뒤쪽 호)
    _vapor_ring(img, lf, scx, scy, r, I, front=False)
    smask = _sphere(img, Zf, Df, R, scx, scy, r, lf, I)
    _vapor_ring(img, lf, scx, scy, r, I, front=True)
    # 투과광 글로우 + 대기 산란
    gl = _small_glow(smask, (8, 30, 120), (0.35, 0.24, 0.18))
    img += gl[..., None] * (np.array([0.14, 0.62, 1.05], f32) * 0.85) * (I / 2.4)
    haze = 1 - np.exp(-np.minimum(Zf, 2e4) / 4500.0)
    ag = np.exp(-np.maximum(rd - r, 0) / (0.5 * r + 140.0))
    img += (haze * (0.10 + 0.5 * ag) * I * 0.16)[..., None] * HAZE_BLAST
    base_glow = np.exp(-((YY - (sy(0, Z0) - 10)) / (50 + 0.08 * r)) ** 2) * \
        np.exp(-np.abs(XX - scx) / (r * 1.2 + 160))
    img += (base_glow * I * 0.22)[..., None] * FIRE

    # --- 히어로 타워 붕괴 (셀로 전환)
    for item in hero_vis:
        if item is not None:
            _hero_break(img, Zf, Df, item, lf, scx, scy, I)

    # --- 앞쪽 먼지 벽
    for k in ((0, 2) if walls else ()):
        Rk = ring_R(k, (lf // 2) * 2)          # 작화(먼지 벽)는 2코마
        if Rk is None:
            continue
        al = math.exp(-max(lf - RINGS[k][0] - RINGS[k][3], 0) / 3.0)
        _smoke_wall(img, Zf, Rk - 35, _wall_h(k, Rk), al, lf, 3 + k, scx, scy, r, back=False, lit=I / 2.4)

    # --- 파편 / 젖혀지는 빌딩
    if not late:
        _lean_buildings(img, lf, R, Rd, scx, scy, Zf)
        _debris(img, lf, R, Rd, scx, scy)

    _arcs_embers(img, lf, scx, scy, r, I)
    # --- 코로나 광선 (투과광 스파이크)
    rl = radial_lines_mask(scx, scy, n=64, seed=900 + lf // 2, r_in=r * 0.96, r_jit=0.05,
                           width=(0.002, 0.007))
    fall = np.exp(-np.maximum(rd - r, 0) / (0.35 * r + 60))
    img += (rl * fall * (I / 2.4) * 0.35)[..., None] * RIMC
    _climax_silhouettes(img, lf, scx, scy)
    _rush(img, lf, scx, scy, r)
    return img, Zf, (scx, scy, r)


def _hero_break(img, Zf, Df, item, lf, scx, scy, I):
    hs, fh, col = item
    x0, y0, x1, y1 = hs["box"]
    lf2 = (lf // 2) * 2
    age = max(0.0, lf2 - fh + 1.0)
    h, w = hs["A"].shape
    rng = np.random.default_rng(int(hs["d"]))
    frac = rng.uniform(0.42, 0.58)
    yb = (hs["top_y"] + (hs["base_y"] - hs["top_y"]) * frac) - y0
    xs = np.arange(w, dtype=f32)
    jag = yb + np.interp(xs, np.linspace(0, w, 9), rng.uniform(-14, 14, 9) * (w / 200.0 + 0.3))
    ys = np.arange(h, dtype=f32)[:, None]
    upper = np.clip(jag[None, :] - ys + 0.5, 0, 1)
    lower = 1 - upper
    # 하부: 남아 있음 (먼지 벽이 덮을 때까지)
    if age < 18:
        _composite_hero(img, Zf, Df, hs, col, sel=lower)
        # 부러진 단면의 불길
        fl = np.exp(-((ys - jag[None, :]) / 3.0) ** 2) * hs["A"]
        img[y0:y1, x0:x1] += fl[..., None] * FIRE * 1.4
    # 상부: 셀로 전환되어 넘어진다
    side = hs["side"]
    ang = side * -(2.5 + 2.4 * age + 0.7 * age * age)
    pivx = (hs["xc"] + side * hs["half_w"]) - x0
    pivy = float(np.interp(pivx, xs, jag))
    M = cv2.getRotationMatrix2D((pivx, pivy), ang, 1.0 + 0.012 * age)
    M[0, 2] += x0 + side * (3.5 * age + 1.2 * age * age)
    M[1, 2] += y0 + 0.6 * age * age
    pad_src = np.concatenate([col * upper[..., None], (hs["A"] * upper)[..., None]], -1).astype(f32)
    # 변환 후 bbox 만 워프 (전체 프레임 연산 회피)
    rows = np.nonzero(upper.max(1) > 0)[0]
    if len(rows) == 0:
        return
    cor = np.array([(0, 0), (w, 0), (0, rows.max() + 1), (w, rows.max() + 1)], np.float64)
    tc = cor @ M[:, :2].T + M[:, 2]
    bx0 = int(max(0, math.floor(tc[:, 0].min()) - 8))
    by0 = int(max(0, math.floor(tc[:, 1].min()) - 8))
    bx1 = int(min(W, math.ceil(tc[:, 0].max()) + 8))
    by1 = int(min(H, math.ceil(tc[:, 1].max()) + 8))
    if bx1 <= bx0 or by1 <= by0:
        return
    M2 = M.copy()
    M2[0, 2] -= bx0
    M2[1, 2] -= by0
    warped = cv2.warpAffine(pad_src, M2, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    a = warped[..., 3]
    if a.max() <= 0:
        return
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = np.clip(cv2.dilate(a, k) - a, 0, 1)
    reg = img[by0:by1, bx0:bx1]
    reg *= 1 - a[..., None]
    reg += warped[..., :3]
    reg *= 1 - ring[..., None] * 0.9
    reg += ring[..., None] * INK * 0.9
    # 광원(구체) 쪽 역광 림 2톤: 백열 선 + 주황 띠 (셀 하이라이트)
    sx_ = -(scx - hs["xc"]) / abs(scx - hs["xc"] + 1e-3)
    sh1 = cv2.warpAffine(a, np.float32([[1, 0, sx_ * 2.5], [0, 1, 2.5]]), (bx1 - bx0, by1 - by0))
    sh2 = cv2.warpAffine(a, np.float32([[1, 0, sx_ * 9.0], [0, 1, 8.0]]), (bx1 - bx0, by1 - by0))
    hot = np.clip(a - sh1, 0, 1)
    org = np.clip(a - sh2 - hot, 0, 1)
    reg += hot[..., None] * WARM_HOT + org[..., None] * WARM_ORG


# ---------------------------------------------------------------------------
# 촬영 처리 / 색 반전
# ---------------------------------------------------------------------------
def _camera(img, lf, focus):
    amp = shake_amp(lf)
    dx = amp * (0.7 * wobble(lf * 0.9, 51, (2.1, 4.7, 9.3)) + 0.6 * (hash01(52, lf) - 0.5))
    dy = amp * 0.8 * (0.7 * wobble(lf * 0.9, 53, (1.7, 5.1, 8.9)) + 0.6 * (hash01(54, lf) - 0.5))
    rot = amp * 0.02 * (hash01(55, lf) - 0.5)
    if lf < T_BANG:
        sc = 1.0 + 0.05 * float(ease_in(clamp((lf - 2) / 11.0), 1.6))
    else:
        sc = 1.05 + 0.06 * math.exp(-(lf - T_BANG) / 2.5) + 0.03 * math.exp(-max(lf - 28, 0) / 2.5) * \
            (lf >= 28) + 0.10 * float(clamp((lf - 40) / 11.0)) ** 2
    sc += 0.02 + amp / 900.0
    fx, fy = focus
    return transform(img, dx, dy, sc, rot, cx=fx, cy=fy, border=cv2.BORDER_REFLECT)


def _negative(img):
    v = np.clip(img, 0, 1.0)
    out = 1.0 - v
    return (out * np.array([0.95, 1.0, 1.05], f32)).astype(f32)


def _red_mono(img):
    L = luma(np.clip(img, 0, 3.0))
    stops_x = np.array([0.0, 0.12, 0.45, 1.0, 2.0], f32)
    cols = np.array([(0.04, 0.0, 0.01), (0.30, 0.01, 0.03), RED, (1.6, 0.45, 0.32), (2.6, 1.9, 1.6)], f32)
    out = np.empty(img.shape, f32)
    for c in range(3):
        out[..., c] = np.interp(L, stops_x, cols[:, c])
    return out


NEG_FRAMES = (14, 29, 40)
RED_FRAMES = (15, 28, 41)


def render(ctx: Ctx) -> np.ndarray:
    lf = ctx.lframe
    if lf <= 1 or lf >= T_WHITE:
        return np.broadcast_to(WHITE_HOT, (H, W, 3)).astype(f32).copy()
    if lf < T_BANG:
        img, Zf = _pre(ctx)
        img = _camera(img, lf, (CX, sy(SEED_Y, Z0)))
        return img.astype(f32)
    img, Zf, (scx, scy, r) = _blast(ctx)
    # 집중선
    sl = 0.0
    for t0, a, dec in ((14, 1.0, 3.2), (28, 0.8, 3.0), (40, 1.0, 5.0)):
        if lf >= t0:
            sl = max(sl, a * math.exp(-(lf - t0) / dec))
    sl = max(sl, 0.10)
    if sl > 0.02:
        lines = radial_lines_mask(scx, scy, n=120, seed=1200 + lf // 2, r_in=r * 1.15 + 120, r_jit=1.0,
                                  width=(0.0010, 0.0038))
        img += (lines * sl * 1.4)[..., None] * np.array([0.9, 0.97, 1.0], f32)
    # 갓레이 (저해상도 줌 블러)
    sw, sh = W // 3, H // 3
    small = cv2.resize(np.clip(img - 1.6, 0, 4), (sw, sh), interpolation=cv2.INTER_AREA)
    gr = zoom_blur(small, scx / 3, scy / 3, amount=0.45, steps=10)
    img += cv2.resize(gr, (W, H), interpolation=cv2.INTER_LINEAR) * 0.35 * np.array([0.8, 0.95, 1.0], f32)
    img = _camera(img, lf, (scx, scy))
    zb = 0.0
    for t0, a, dec in ((14, 0.10, 2.5), (28, 0.06, 2.0), (40, 0.12, 4.0)):
        if lf >= t0:
            zb = max(zb, a * math.exp(-(lf - t0) / dec))
    if lf >= 44:
        zb = max(zb, 0.03)
    if 41 <= lf <= 47:
        zb = min(zb, 0.035)
    if zb > 0.01 and lf not in NEG_FRAMES:
        img = zoom_blur(img, scx, scy, amount=zb, steps=8)
    # 엔딩: 구체가 화면을 삼킨다
    if lf >= 47:
        k = float(clamp((lf - 47) / 4.5)) ** 1.5
        img = img * (1 - k) + WHITE_HOT * k
    if lf == 16:
        img = img * 1.6 + 0.35
    if lf in NEG_FRAMES:
        img = _negative(img)
    elif lf in RED_FRAMES:
        img = _red_mono(img)
    return img.astype(f32)
