"""残像 ZANZŌ — 완전 프로시저럴 사운드트랙 (numpy/scipy만 사용, 샘플·다운로드 없음).

render_audio(path, start_frame=0, end_frame=360, fps=24)
    48 kHz 16-bit 스테레오 WAV 를 쓴다. 항상 전체 믹스를 렌더한 뒤 요청 구간만 잘라낸다.

구조 (컷 지점 = scenes.starts(), 기본 3.0 / 6.0 / 8.5 / 11.0 / 13.5 s)
    s1 city    디튠 소 패드(LPF 서서히 개방) + 도시 험·먼 사이렌 + 가믈란 벨, 끝에 라이저
    s2 chase   132 BPM 타이코/시메/베이스 + 엔진(rpm 곡선) + 도플러 패스바이·스키드·불꽃·임팩트
    s3 eye     거의 무음: 심장박동, 이명, 숨결 노이즈 패드, 눈물 차임
    s4 mirror  어긋나는 오르골 아이돌 팝 + 유리 파쇄(그래뉼러) + TV 스태틱/스터터 + 역재생 스웰
    s5 blast   BOOM(서브 드롭 90→28 Hz + 왜곡 럼블) + 충격파 + 콰이어 "아" → 화이트아웃
    s6 title   오다이코 + 공 + 저음 금관 스탭 → 영사기 플러터/런아웃 → 15.0 s 무음
    전 구간    아주 작은 영사기 모터음 + 히스/크래클 베드

동기: 각 장면 모듈의 EVENTS = [(local_frame, label)] 를 읽어
(starts()[i] + local_frame) / 24 초에 장면별 사운드 디자인으로 배치한다.
모든 난수는 문자열 시드(crc32) 기반이라 결정적이다.
"""
from __future__ import annotations

import math
import wave
import zlib
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.linalg import hadamard
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
FPS = 24                       # 타임라인 정의 프레임레이트
TAU = 2.0 * np.pi
DEFAULT_FRAMES = (72, 72, 60, 60, 60, 36)
DEFAULT_NAMES = ("s1_city", "s2_chase", "s3_eye", "s4_mirror", "s5_blast", "s6_title")
KINDS = ("city", "chase", "eye", "mirror", "blast", "title")


# ═══════════════════════════ 기본 유틸 ═══════════════════════════
def db(x):
    return 10.0 ** (x / 20.0)


def rng(*parts):
    return np.random.default_rng(zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF)


def nsamp(sec):
    return max(1, int(round(sec * SR)))


def idx(sec):
    return int(round(sec * SR))


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def norm(x, peak=1.0):
    m = float(np.max(np.abs(x))) if np.size(x) else 0.0
    return x * (peak / m) if m > 1e-12 else x


def env_perc(n, att, tau):
    t = np.arange(n) / SR
    e = np.exp(-np.maximum(t - att, 0.0) / max(tau, 1e-4))
    if att > 0:
        e *= np.minimum(t / att, 1.0)
    return e


def edge_fade(x, fi=0.002, fo=0.004):
    x = np.array(x, float, copy=True)
    a, b = min(len(x), nsamp(fi)), min(len(x), nsamp(fo))
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    x[:a] = (x[:a].T * w).T
    w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b))
    x[len(x) - b:] = (x[len(x) - b:].T * w).T
    return x


def smooth_rand(n, rate, r):
    """-1..1 사이를 rate Hz 로 떠도는 부드러운 랜덤 곡선."""
    m = max(2, int(n / SR * rate) + 3)
    pts = r.uniform(-1, 1, m)
    x = np.linspace(0, m - 1.001, n)
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


def white(n, r):
    return r.standard_normal(n)


def pink(n, r):
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    y = signal.lfilter(b, a, r.standard_normal(n + 2000))[2000:]
    return y / (np.std(y) + 1e-12)


def brown(n, r):
    y = signal.lfilter([1.0], [1.0, -0.997], r.standard_normal(n + 6000))[6000:]
    y -= y.mean()
    return y / (np.std(y) + 1e-12)


def pan_st(x, pan=0.0):
    """모노 → 스테레오 (등전력, 센터 = 1). pan 은 스칼라/배열 -1..1."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        p = np.clip(pan, -1, 1)
        return np.stack([x[:, 0] * np.minimum(1, 1 - p), x[:, 1] * np.minimum(1, 1 + p)], 1)
    th = (np.clip(pan, -1, 1) + 1.0) * (np.pi / 4)
    return np.stack([x * np.cos(th), x * np.sin(th)], 1) * math.sqrt(2.0)


def vread(x, pos):
    """소수 샘플 위치 pos 로 x 읽기(선형 보간, 범위 밖 0)."""
    i = np.arange(len(x))
    if x.ndim == 1:
        return np.interp(pos, i, x, left=0.0, right=0.0)
    return np.stack([np.interp(pos, i, x[:, c], left=0.0, right=0.0) for c in range(x.shape[1])], 1)


# ═══════════════════════════ 필터 ═══════════════════════════
def _bq(kind, fc, q=0.7071, gain_db=0.0):
    fc = min(max(float(fc), 8.0), 0.46 * SR)
    w = TAU * fc / SR
    cw, sw = math.cos(w), math.sin(w)
    al = sw / (2.0 * q)
    if kind == "lp":
        b = ((1 - cw) / 2, 1 - cw, (1 - cw) / 2); a = (1 + al, -2 * cw, 1 - al)
    elif kind == "hp":
        b = ((1 + cw) / 2, -(1 + cw), (1 + cw) / 2); a = (1 + al, -2 * cw, 1 - al)
    elif kind == "bp":
        b = (al, 0.0, -al); a = (1 + al, -2 * cw, 1 - al)
    elif kind == "peak":
        A = 10 ** (gain_db / 40)
        b = (1 + al * A, -2 * cw, 1 - al * A); a = (1 + al / A, -2 * cw, 1 - al / A)
    elif kind in ("ls", "hs"):
        A = 10 ** (gain_db / 40)
        s = 2 * math.sqrt(A) * sw / 2 * math.sqrt(2)
        if kind == "ls":
            b = (A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s))
            a = ((A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s)
        else:
            b = (A * ((A + 1) + (A - 1) * cw + s), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s))
            a = ((A + 1) - (A - 1) * cw + s, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s)
    else:
        raise ValueError(kind)
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def bq(x, kind, fc, q=0.7071, gain_db=0.0, stages=1):
    sos = np.array([_bq(kind, fc, q, gain_db)] * stages)
    return signal.sosfilt(sos, x, axis=0)


def tv_bq(x, kind, fc, q=0.7071, stages=1, block=64):
    """시변 바이쿼드 (블록마다 계수 갱신, 상태 유지)."""
    x = np.asarray(x, float)
    n = x.shape[0]
    fc = np.broadcast_to(np.asarray(fc, float), (n,))
    qv = np.broadcast_to(np.asarray(q, float), (n,))
    y = np.empty_like(x)
    zi = np.zeros((stages, 2) + x.shape[1:])
    for s in range(0, n, block):
        e = min(n, s + block)
        m = (s + e) // 2
        sos = np.array([_bq(kind, fc[m], qv[m])] * stages)
        y[s:e], zi = signal.sosfilt(sos, x[s:e], axis=0, zi=zi)
    return y


def formant(x, vowel="a"):
    table = {
        "a": ((800, 9, 1.0), (1150, 11, 0.55), (2900, 20, 0.28), (3900, 26, 0.2), (4950, 30, 0.1)),
        "o": ((450, 7, 1.0), (800, 9, 0.5), (2830, 20, 0.2), (3800, 26, 0.15), (4950, 30, 0.08)),
        "h": ((700, 2.0, 1.0), (1300, 3.0, 0.8), (2600, 4.0, 0.6), (4200, 5.0, 0.3)),
    }[vowel]
    return sum(g * bq(x, "bp", f, q) for f, q, g in table)


# ═══════════════════════════ 오실레이터 ═══════════════════════════
def phase(freq, n=None, ph0=0.0):
    if np.ndim(freq) == 0:
        return ph0 + np.arange(n) * (float(freq) / SR)
    f = np.asarray(freq, float)
    out = np.empty(len(f))
    out[0] = 0.0
    np.cumsum(f[:-1] / SR, out=out[1:])
    return out + ph0


def osc_sin(freq, n=None, ph0=0.0):
    return np.sin(TAU * phase(freq, n, ph0))


def osc_saw(freq, n=None, ph0=0.0):
    """PolyBLEP 밴드리밋 톱니."""
    ph = phase(freq, n, ph0)
    p = ph % 1.0
    dt = np.broadcast_to(np.abs(np.asarray(freq, float)) / SR + 1e-12, p.shape)
    y = 2.0 * p - 1.0
    m = p < dt
    x = p[m] / dt[m]
    y[m] -= x + x - x * x - 1.0
    m = p > 1.0 - dt
    x = (p[m] - 1.0) / dt[m]
    y[m] -= x * x + x + x + 1.0
    return y


def osc_sq(freq, n=None, ph0=0.0):
    return 0.5 * (osc_saw(freq, n, ph0) - osc_saw(freq, n, ph0 + 0.5))


# ═══════════════════════════ 리버브 (FDN → IR) ═══════════════════════════
def _isprime(k):
    if k < 2:
        return False
    i = 2
    while i * i <= k:
        if k % i == 0:
            return False
        i += 1
    return True


def _fdn_ir(t60, sec, alpha, pre, dmin, dmax, hp, seed, nl=16):
    """16 라인 Hadamard FDN(Jot 흡수 필터) + 올패스 확산 → 스테레오 IR."""
    r = rng("fdn", seed)
    N = nsamp(sec)
    used, d = set(), []
    for v in np.exp(np.linspace(np.log(dmin), np.log(dmax), nl)) * SR:
        k = int(v * r.uniform(0.96, 1.04))
        while not _isprime(k) or k in used:
            k += 1
        used.add(k)
        d.append(k)
    d = np.array(d)
    B, D = int(d.min()), int(d.max())
    A = hadamard(nl) / math.sqrt(nl)
    g = 10.0 ** (-3.0 * d / (SR * t60))
    p = np.clip(math.log(10) / 4 * np.log10(g) * (1 - 1 / alpha ** 2), 0.0, 0.95)
    # 직렬 슈뢰더 올패스 IR(해석해)로 확산된 입력
    x = np.zeros(N); x[0] = 1.0
    for ms, ga in ((4.7, 0.7), (3.6, 0.7), (12.7, 0.6), (9.3, 0.6)):
        M = int(ms * SR / 1000)
        h = np.zeros(N); h[0] = -ga
        k = np.arange(1, N // M + 1)
        ok = k * M < N
        h[k[ok] * M] = (1 - ga * ga) * ga ** (k[ok] - 1)
        x = signal.fftconvolve(x, h)[:N]
    b_in = r.choice([-1.0, 1.0], nl)
    cL, cR = r.choice([-1.0, 1.0], nl), r.choice([-1.0, 1.0], nl)
    u = np.zeros((nl, N + D))
    y = np.zeros((nl, N))
    zi = np.zeros(nl)
    for s in range(0, N, B):
        e = min(N, s + B)
        for i in range(nl):
            seg = u[i, s + D - d[i]: e + D - d[i]]
            z, zf = signal.lfilter([(1 - p[i]) * g[i]], [1.0, -p[i]], seg, zi=[zi[i]])
            zi[i] = zf[0]
            y[i, s:e] = z
        u[:, s + D: e + D] = A @ y[:, s:e] + b_in[:, None] * x[s:e]
    ir = np.stack([cL @ y, cR @ y], 1)
    ir = np.concatenate([np.zeros((nsamp(pre), 2)), ir])[:N]
    ir = bq(ir, "hp", hp, 0.6, stages=2)
    fo = int(N * 0.3)
    ir[N - fo:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fo)))[:, None]
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True)) + 1e-12
    return ir


_IRCFG = {
    "room": dict(t60=0.8, sec=1.3, alpha=0.45, pre=0.004, dmin=0.006, dmax=0.031, hp=150.0, seed=11),
    "plate": dict(t60=2.4, sec=3.0, alpha=0.6, pre=0.002, dmin=0.004, dmax=0.027, hp=220.0, seed=12),
    "hall": dict(t60=2.8, sec=3.6, alpha=0.35, pre=0.018, dmin=0.012, dmax=0.067, hp=160.0, seed=13),
    "huge": dict(t60=5.0, sec=6.0, alpha=0.3, pre=0.035, dmin=0.021, dmax=0.113, hp=140.0, seed=14),
}


@lru_cache(maxsize=None)
def _ir(name):
    return _fdn_ir(**_IRCFG[name])


def reverb(x, name):
    ir = _ir(name)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    n = len(x)
    return np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:n] for c in range(2)], 1)


class Bus:
    """장면 버스: dry + 리버브 센드들 (구간 [t0, t1) 만 메모리에 둔다)."""

    def __init__(self, t0, t1):
        self.t0 = t0
        self.i0 = idx(t0)
        self.n = idx(t1) - self.i0
        self.dry = np.zeros((self.n, 2))
        self.sends = {}

    def add(self, sig, t, gain=1.0, pan=0.0, send=None):
        st = pan_st(sig, pan) * gain
        a = idx(t) - self.i0
        lo, hi = max(a, 0), min(a + len(st), self.n)
        if hi <= lo:
            return
        seg = st[lo - a: hi - a]
        self.dry[lo:hi] += seg
        for k, v in (send or {}).items():
            if v > 0:
                if k not in self.sends:
                    self.sends[k] = np.zeros((self.n, 2))
                self.sends[k][lo:hi] += seg * v

    def local_t(self):
        return self.t0 + np.arange(self.n) / SR

    def render(self):
        out = self.dry.copy()
        for k, buf in self.sends.items():
            out += reverb(buf, k)
        return out


# ═══════════════════════════ 악기 / 효과음 ═══════════════════════════
def taiko(f0=70.0, dur=1.4, decay=0.45, seed=0, slap=0.5, drive=1.6, thump=1.0):
    """피치가 떨어지는 막(membrane) 합성 + 채 소리 + 쿵."""
    r = rng("taiko", seed, f0)
    n = nsamp(dur)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.45 * np.exp(-t / 0.022) + 0.06 * np.exp(-t / 0.3))
    ph = phase(f)
    y = np.sin(TAU * ph) * np.exp(-t / decay)
    for ratio, amp, dk in ((1.59, .30, .40), (2.14, .20, .28), (2.30, .16, .22), (2.65, .10, .16),
                           (2.92, .08, .12), (3.16, .06, .10), (3.50, .05, .08)):
        y += amp * np.sin(TAU * ph * ratio * r.uniform(0.99, 1.01) + r.uniform(0, TAU)) * np.exp(-t / (decay * dk))
    nz = r.standard_normal(n)
    y += thump * 6.0 * bq(nz, "lp", 200, 0.8) * np.exp(-t / 0.035)
    y += slap * 2.5 * bq(nz, "bp", 1100, 0.9) * np.exp(-t / 0.010)
    y = np.tanh(drive * norm(y)) / math.tanh(drive) * np.minimum(t / 0.0008, 1)
    return norm(y)


def ka(seed=0, f=2400.0, dur=0.09):
    """테두리/시메다이코 '카'."""
    r = rng("ka", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    y = 3.0 * bq(r.standard_normal(n), "bp", f, 2.0) * np.exp(-t / 0.006)
    y += 0.6 * np.sin(TAU * 820 * t) * np.exp(-t / 0.02) + 0.4 * np.sin(TAU * 1730 * t) * np.exp(-t / 0.012)
    return norm(y * np.minimum(t / 0.0004, 1))


def gong(f0=55.0, dur=5.0, seed=0, bright=1.0, bloom=0.3, npart=40):
    """탐탐/공: 비조화 부분음 구름, 고역은 늦게 피어남(bloom), 맥놀이."""
    r = rng("gong", seed, f0)
    n = nsamp(dur)
    t = np.arange(n) / SR
    base = [1.0, 1.47, 2.09, 2.56, 2.98, 3.6, 4.2]
    freqs = [f0 * b for b in base] + list(np.exp(r.uniform(np.log(f0 * 4.5), np.log(7500), npart)))
    y = np.zeros(n)
    for i, fr in enumerate(freqs):
        rel = fr / f0
        a = r.uniform(0.5, 1.0) * rel ** -0.55 * (1.0 if i < len(base) else 0.8 * bright)
        T = float(np.clip(dur * 0.5 * rel ** -0.35 * r.uniform(0.7, 1.2), 0.2, dur))
        att = 0.002 + bloom * min(1.0, rel / 30) * r.uniform(0.5, 1.5)
        e = (1 - np.exp(-t / att)) * np.exp(-t / T)
        bt = r.uniform(0.2, 1.8)
        y += a * e * (np.sin(TAU * fr * t + r.uniform(0, TAU)) + 0.7 * np.sin(TAU * (fr + bt) * t + r.uniform(0, TAU)))
    y += 8.0 * bq(r.standard_normal(n), "lp", 300) * np.exp(-t / 0.03)
    return norm(y * np.minimum(t / 0.001, 1))


BELLS = {
    "gender": ((1, 1, 1), (2.76, .45, .45), (5.40, .25, .22), (8.93, .12, .12)),
    "bonang": ((1, 1, 1), (1.52, .35, .5), (2.04, .5, .6), (2.83, .28, .35), (4.01, .18, .22), (5.43, .1, .15)),
    "kempul": ((1, 1, 1), (1.47, .45, .6), (2.0, .6, .7), (2.53, .3, .4), (3.01, .25, .35), (4.08, .12, .2)),
    "glass": ((1, 1, 1), (2.32, .45, .55), (4.25, .25, .35), (6.63, .12, .2)),
}


def bell(f, dur=3.0, seed=0, kind="bonang", ombak=4.0, decay=1.4):
    """가믈란 금속 타악: 비조화 부분음 + 옴박(쌍으로 디튠된 맥놀이)."""
    r = rng("bell", seed, float(f), kind)
    n = nsamp(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, amp, dk in BELLS[kind]:
        fr = f * ratio * r.uniform(0.996, 1.004)
        if fr > 0.45 * SR:
            continue
        e = np.exp(-t / (decay * dk))
        y += amp * e * (np.sin(TAU * fr * t + r.uniform(0, TAU)) + 0.8 * np.sin(TAU * (fr + ombak * ratio ** .5) * t + r.uniform(0, TAU)))
    y += 0.4 * bq(r.standard_normal(n), "bp", min(f * 4, 15000), 1.5) * np.exp(-t / 0.004)
    return norm(y * np.minimum(t / 0.0006, 1))


def crash(dur=2.0, seed=0, decay=0.9, bright=1.0):
    r = rng("crash", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    nz = r.standard_normal(n)
    y = bq(nz, "hp", 3200 * bright, 0.7) * np.exp(-t / decay)
    metal = sum(np.sign(np.sin(TAU * f * 1.6 * t + r.uniform(0, TAU))) for f in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0))
    y += 0.25 * bq(metal, "hp", 5200, 0.7) * np.exp(-t / (decay * 0.7))
    y += 2.0 * bq(nz, "bp", 4500, 0.8) * np.exp(-t / 0.02)
    return norm(y * np.minimum(t / 0.0005, 1))


def whoosh(dur, peak=0.55, f0=250.0, f1=3000.0, f2=None, q=1.2, seed=0, pan0=-0.6, pan1=0.6,
           att=2.2, rel=1.4, body=0.5):
    """대역 통과 노이즈 스윕 + 팬 이동."""
    r = rng("whoosh", seed)
    n = nsamp(dur)
    u = np.arange(n) / n
    f2 = f2 if f2 is not None else f0 * 1.3
    pk = min(max(peak, 0.02), 0.98)
    up = np.clip(u / pk, 0, 1)
    dn = np.clip((u - pk) / (1 - pk), 0, 1)
    fc = np.where(u < pk, f0 * (f1 / f0) ** (up ** 1.5), f1 * (f2 / f1) ** (dn ** 0.7))
    amp = np.where(u < pk, up ** att, (1 - dn) ** rel)
    y = tv_bq(white(n, r), "bp", fc, q, block=128)
    y += body * tv_bq(white(n, r), "lp", fc * 0.35, 0.7, block=128)
    y *= amp
    s = u * u * (3 - 2 * u)
    st = pan_st(y, pan0 + (pan1 - pan0) * s)
    # 살짝 디코릴레이션
    st[:, 1] = np.roll(st[:, 1], 24)
    return norm(edge_fade(st, 0.003, 0.01))


def src_wind(n, r):
    x = r.standard_normal(n)
    return 2.0 * bq(x, "bp", 1500, 0.5) + 2.0 * bq(x, "lp", 400)


def src_car(n, r, f=110.0):
    ff = f * (1 + 0.01 * smooth_rand(n, 8, r))
    y = 0.6 * osc_saw(ff) + 0.4 * osc_saw(ff * 0.5)
    y = bq(y, "lp", 1400)
    return y + 2.0 * bq(r.standard_normal(n), "bp", 900, 0.6)


def passby(src, dur, t_close, v=60.0, d=3.0, side=1.0, seed=0, mode="cross", c=343.0):
    """실제 기하로 계산한 도플러 패스바이 (가변 지연 읽기 + 1/r 감쇠 + 공기 흡수 + 팬)."""
    r = rng("passby", seed)
    n = nsamp(dur)
    pre = nsamp(0.8)
    ne = n + pre
    te = (np.arange(ne) - pre) / SR
    s = src(ne, r)
    tce = t_close - d / c
    x = v * (te - tce)
    rr = np.sqrt(x * x + d * d)
    ta = te + rr / c
    t = np.arange(n) / SR
    te_t = np.interp(t, ta, te)
    pos = (te_t - te[0]) * SR
    near = vread(s, pos)
    far = vread(bq(s, "lp", 1100, 0.7), pos)
    xr = v * (te_t - tce)
    rt = np.sqrt(xr * xr + d * d)
    w = np.clip(2.0 * d / rt, 0, 1)
    y = (near * w + far * (1 - w)) * (d / rt)
    if mode == "cross":
        pan = side * 0.85 * np.clip(xr / rt, -1, 1)
    else:
        pan = side * 0.9 * np.sqrt(d / rt)
    return norm(edge_fade(pan_st(y, pan), 0.02, 0.05))


def screech(dur, seed=0, f0=1350.0, f1=820.0):
    """타이어 스퀼: 지터가 많은 음정 + 협대역 노이즈 + 스틱-슬립 AM."""
    r = rng("screech", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    u = t / dur
    f = f0 * (f1 / f0) ** u * (1 + 0.045 * smooth_rand(n, 40, r) + 0.015 * smooth_rand(n, 7, r))
    ph = phase(f)
    y = 0.8 * np.sin(TAU * ph) + 0.45 * np.sin(TAU * 2 * ph + 0.5) + 0.2 * np.sin(TAU * 3 * ph + 1.1)
    nz = tv_bq(white(n, r), "bp", f, 10.0, block=64) * 3.0
    nz += tv_bq(white(n, r), "bp", 2 * f, 8.0, block=64) * 1.5
    am = 0.65 + 0.35 * np.sin(TAU * phase(55 + 25 * smooth_rand(n, 9, r)))
    y = (0.5 * y + nz) * am
    y += 0.4 * bq(white(n, r), "hp", 3000) * (0.5 + 0.5 * smooth_rand(n, 30, r))
    env = np.minimum(t / 0.03, 1) * np.clip((1 - u) / 0.12, 0, 1) ** 0.7 * (0.75 + 0.25 * smooth_rand(n, 5, r))
    return norm(y * env)


def crackle(dur, rate=200.0, seed=0, tau=None, f_lo=2500.0, f_hi=9000.0, decay=0.003):
    """금속성 불꽃 튀는 소리: 포아송 클릭 → 비조화 공진기 뱅크 (스테레오)."""
    r = rng("crackle", seed)
    n = nsamp(dur)
    out = np.zeros((n, 2))
    cnt = r.poisson(rate * dur)
    tm = r.exponential(tau, cnt) if tau else r.uniform(0, dur, cnt)
    tm = tm[tm < dur - 0.02]
    amps = r.exponential(1.0, len(tm)) ** 1.6
    pans = r.uniform(-0.9, 0.9, len(tm))
    ii = (tm * SR).astype(int)
    imp = np.zeros((n, 2))
    np.add.at(imp[:, 0], ii, amps * np.cos((pans + 1) * np.pi / 4))
    np.add.at(imp[:, 1], ii, amps * np.sin((pans + 1) * np.pi / 4))
    for k in range(4):
        fc = float(np.exp(r.uniform(np.log(f_lo), np.log(f_hi))))
        out += bq(imp, "bp", fc, r.uniform(6, 16))
    out += 0.4 * bq(imp, "hp", 6000)
    out = signal.lfilter([1.0], [1.0, -math.exp(-1 / (decay * SR))], out, axis=0) * 0.05 + out
    return norm(out)


def scrape(dur, seed=0):
    """발판이 아스팔트를 긁는 금속 마찰음."""
    r = rng("scrape", seed)
    n = nsamp(dur)
    nz = white(n, r)
    am = 1 + 0.8 * smooth_rand(n, 60, r)
    y = sum(bq(nz, "bp", f * r.uniform(0.97, 1.03), 18) for f in (2150, 3720, 5230, 7400))
    y = (y * 2 + 0.3 * bq(nz, "hp", 4000)) * am
    return norm(edge_fade(y, 0.02, 0.05))


def hit(seed=0, big=1.0, f_hi=62.0, f_lo=30.0):
    """임팩트: 서브 쿵 + 크랙 + 바디 (포화)."""
    r = rng("hit", seed)
    n = nsamp(0.6 + 1.4 * big)
    t = np.arange(n) / SR
    f = f_lo + (f_hi - f_lo) * np.exp(-t / 0.06)
    sub = np.sin(TAU * phase(f)) * np.exp(-t / (0.35 * big))
    nz = white(n, r)
    crack = 4.0 * bq(nz, "bp", 2500, 0.7) * np.exp(-t / 0.015)
    body = 8.0 * bq(nz, "lp", 500) * np.exp(-t / 0.12)
    y = np.tanh(1.8 * (sub + 0.5 * crack + 0.6 * body))
    return norm(y * np.minimum(t / 0.0005, 1))


def thoom(dur=0.8, seed=0, f_hi=70.0, f_lo=32.0):
    r = rng("thoom", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    f = f_lo + (f_hi - f_lo) * np.exp(-t / 0.12)
    y = np.sin(TAU * phase(f)) * env_perc(n, 0.01, dur * 0.35)
    y += 3.0 * bq(white(n, r), "lp", 160) * env_perc(n, 0.03, dur * 0.3)
    return norm(np.tanh(1.5 * y))


def glass(dur=2.0, seed=0, ngr=700, tau=0.10, fl=1800.0, fh=12000.0, tail=0.25, crack=1.0, dec_scale=1.0):
    """유리 파쇄: 수백 개의 비조화 그레인(밀도 지수 감쇠) + 크랙 트랜지언트 + 파편 낙하 꼬리."""
    r = rng("glass", seed)
    n = nsamp(dur)
    out = np.zeros((n, 2))
    nb = int(ngr * (1 - tail))
    tms = np.concatenate([r.exponential(tau, nb), 0.04 + r.uniform(0, 1, ngr - nb) ** 1.6 * (dur - 0.4)])
    burst = np.arange(len(tms)) < nb
    for tm, b in zip(tms, burst):
        if tm >= dur - 0.05:
            continue
        f = float(np.exp(r.uniform(np.log(fl), np.log(fh))))
        dec = float(np.clip(r.lognormal(np.log(0.012), 0.6) * (3000 / f) ** 0.5 * dec_scale, 0.002, 0.09))
        L = int(dec * 5 * SR) + 1
        tg = np.arange(L) / SR
        g = np.sin(TAU * f * tg + r.uniform(0, TAU))
        for ratio, a in ((r.uniform(1.4, 1.9), .6), (r.uniform(2.3, 3.1), .4), (r.uniform(3.5, 5.2), .25)):
            if f * ratio < 0.45 * SR:
                g += a * np.sin(TAU * f * ratio * tg + r.uniform(0, TAU))
        g *= np.exp(-tg / dec)
        amp = r.exponential(1.0) * (1.0 if b else 0.45) * math.exp(-tm / 1.2)
        p = r.uniform(-0.95, 0.95)
        i0 = int(tm * SR)
        i1 = min(n, i0 + L)
        th = (p + 1) * np.pi / 4
        out[i0:i1, 0] += g[: i1 - i0] * amp * math.cos(th)
        out[i0:i1, 1] += g[: i1 - i0] * amp * math.sin(th)
    t = np.arange(n) / SR
    nz = white(n, r)
    c = crack * (6.0 * bq(nz, "hp", 2200) * np.exp(-t / 0.006) + 10.0 * bq(nz, "lp", 160) * np.exp(-t / 0.05))
    c += 0.8 * bq(nz, "hp", 5000) * np.exp(-t / 0.25)
    out += pan_st(c, 0.0) * 0.5
    return norm(out)


def crunch(dur=1.5, seed=0, ngr=260):
    """콘크리트/철골 붕괴: 저역 그레인 + 노이즈 덩어리 + 휘는 금속 신음."""
    r = rng("crunch", seed)
    n = nsamp(dur)
    out = np.zeros((n, 2))
    for tm in r.exponential(0.18, ngr):
        if tm > dur - 0.1:
            continue
        f = float(np.exp(r.uniform(np.log(120), np.log(2600))))
        dec = float(np.clip(r.lognormal(np.log(0.03), 0.6), 0.006, 0.15))
        L = int(dec * 4 * SR) + 1
        tg = np.arange(L) / SR
        g = (np.sin(TAU * f * tg) + 0.6 * np.sin(TAU * f * r.uniform(1.3, 2.7) * tg)) * np.exp(-tg / dec)
        g += 0.5 * r.standard_normal(L) * np.exp(-tg / (dec * 0.4))
        i0 = int(tm * SR); i1 = min(n, i0 + L)
        p = r.uniform(-0.9, 0.9)
        th = (p + 1) * np.pi / 4
        a = r.exponential(1.0)
        out[i0:i1, 0] += g[: i1 - i0] * a * math.cos(th)
        out[i0:i1, 1] += g[: i1 - i0] * a * math.sin(th)
    t = np.arange(n) / SR
    groan_f = 70 * (1 - 0.35 * t / dur) * (1 + 0.03 * smooth_rand(n, 12, r))
    groan = bq(osc_saw(groan_f), "bp", 400, 1.5) * env_perc(n, 0.05, dur * 0.4)
    rub = 3.0 * bq(white(n, r), "lp", 900) * env_perc(n, 0.005, 0.25)
    out += pan_st(groan * 0.6 + rub, 0.0)
    return norm(out)


def static_burst(dur, seed=0, buzz=0.5):
    """TV 스태틱: 샘플&홀드 노이즈 + 59.94 Hz 버즈 + 무작위 초핑 (스테레오)."""
    r = rng("static", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    chans = []
    for c in range(2):
        k = int(r.integers(2, 7))
        held = np.repeat(r.standard_normal(n // k + 2), k)[:n]
        x = held * 0.9 + 0.5 * r.standard_normal(n)
        x = bq(bq(x, "hp", 250), "lp", 9500)
        chans.append(x)
    b = np.sign(np.sin(TAU * 59.94 * t)) + 0.5 * np.sign(np.sin(TAU * 119.88 * t + 1.0))
    b = bq(bq(b, "hp", 180), "lp", 3500) * buzz
    st = np.stack(chans, 1) + b[:, None]
    # 초핑 게이트
    g = np.zeros(n)
    pos = 0
    while pos < n:
        L = nsamp(r.uniform(0.006, 0.03))
        g[pos:pos + L] = r.choice([0.1, 0.45, 1.0, 1.0])
        pos += L
    g = uniform_filter1d(g, nsamp(0.0015))
    return norm(edge_fade(st * g[:, None], 0.002, 0.01))


def neon(dur, seed=0, chops=5):
    """네온관 점등 버즈 (100 Hz 정류 험 + 지직임) on/off 스터터."""
    r = rng("neon", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    hum = np.abs(np.sin(TAU * 50 * t))
    hum -= hum.mean()
    y = bq(hum, "hp", 140) + 0.6 * bq(white(n, r), "bp", 3000, 1.0) * (np.abs(smooth_rand(n, 80, r)) ** 3) * 3
    edges = np.sort(r.uniform(0, dur, chops * 2))
    g = np.ones(n) * 0.0
    on = True
    last = 0
    for e in list(edges) + [dur]:
        i = idx(e)
        g[last:i] = 1.0 if on else 0.08
        on = not on
        last = i
    g = uniform_filter1d(g, nsamp(0.002))
    return norm(edge_fade(y * g, 0.002, 0.02))


def heartbeat(seed=0, size=1.0):
    """럽-덥 심장박동 (저역 쿵 2회)."""
    r = rng("hb", seed)
    n = nsamp(0.8)
    y = np.zeros(n)
    for t0, fh, fl, dec, a in ((0.0, 80, 44, 0.10, 1.0), (0.19, 95, 52, 0.075, 0.62)):
        i0 = idx(t0)
        m = n - i0
        tt = np.arange(m) / SR
        f = fl + (fh - fl) * np.exp(-tt / 0.025)
        s = np.sin(TAU * phase(f)) * env_perc(m, 0.004, dec * size)
        s += 4.0 * bq(r.standard_normal(m), "lp", 140) * env_perc(m, 0.002, 0.035)
        y[i0:] += a * s
    y = bq(np.tanh(1.8 * y), "lp", 700)
    return norm(y)


def chime(f, seed=0, dur=2.2, decay=0.8, kind="glass", beat=1.6):
    return bell(f, dur, seed, kind, beat, decay)


def glint(seed=0, dur=0.7, f=4800.0):
    """반짝 '샹': 고역 부분음 + 빠른 트레몰로."""
    r = rng("glint", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    trem = 0.6 + 0.4 * np.sin(TAU * 27 * t)
    y = np.zeros(n)
    for k, a in ((1, 1), (1.5, 0.6), (2.01, 0.45), (2.76, 0.3)):
        y += a * np.sin(TAU * f * k * (1 + 0.03 * t) * t + r.uniform(0, TAU))
    y = y * trem * env_perc(n, 0.015, 0.16)
    y += 0.5 * bq(white(n, r), "hp", 7000) * env_perc(n, 0.04, 0.09)
    return norm(edge_fade(y, 0.001, 0.05))


def tick(seed=0, f=1800.0, dur=0.05):
    r = rng("tick", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    y = 3.0 * bq(white(n, r), "bp", f, 2.0) * np.exp(-t / 0.004) + 0.3 * np.sin(TAU * 140 * t) * np.exp(-t / 0.015)
    return norm(y)


def rev_swell(dur, seed=0, root=50, tonal=1.0, crashy=1.0, bright=1.0, curve=2.2, room="hall"):
    """역재생 스웰: (크래시 + 화음 + 잔향)을 거꾸로 → 끝에서 정점."""
    r = rng("rsw", seed)
    n = nsamp(dur)
    m = n + nsamp(0.25)
    t = np.arange(m) / SR
    fwd = np.zeros(m)
    if crashy:
        fwd += crashy * crash(m / SR, seed, decay=0.7, bright=bright)
    if tonal:
        for k, iv in enumerate((0, 7, 12, 15, 19, 24)):
            fwd += tonal * 0.22 * osc_saw(mtof(root + iv) * (1 + 0.003 * (k - 2.5)), m, r.random()) * np.exp(-t / 1.0)
        fwd = bq(fwd, "lp", 2500 * bright, 0.8)
    wet = reverb(fwd, room)
    st = (0.35 * fwd[:, None] + wet)[:n][::-1].copy()
    u = np.linspace(0, 1, n)
    st *= (u ** curve)[:, None]
    return norm(edge_fade(st, 0.01, 0.003))


def boom(dur=3.2, seed=0, f_hi=90.0, f_lo=28.0, glide=1.0, crack=1.0):
    """초대형 폭발: 서브 드롭 + 왜곡 노이즈 럼블 + 크랙 (스테레오)."""
    r = rng("boom", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    f = f_lo + (f_hi - f_lo) * np.exp(-t / (glide / 3))
    sub = np.sin(TAU * phase(f)) * env_perc(n, 0.004, dur * 0.38)
    chans = []
    for c in range(2):
        br = bq(brown(n, r), "lp", 240)
        br *= 1 + 0.6 * smooth_rand(n, 7, r)
        br = np.tanh(2.6 * br)
        br = bq(br, "lp", 1400, 0.7) * env_perc(n, 0.035, dur * 0.26)
        mid = np.tanh(3 * bq(white(n, r), "bp", 320, 0.8)) * env_perc(n, 0.01, 0.45)
        chans.append(br + 0.35 * mid)
    rum = np.stack(chans, 1)
    nz = white(n, r)
    cr = crack * (2.5 * bq(nz, "bp", 2400, 0.5) * np.exp(-t / 0.025) + 2.0 * bq(nz, "hp", 5000) * np.exp(-t / 0.06))
    out = pan_st(1.15 * sub + cr, 0.0) + 0.9 * rum
    return norm(out * np.minimum(t / 0.0006, 1)[:, None])


def choir(notes, dur, seed=0, glide=None, nv=4, vowel="a", breath=0.15):
    """포먼트 콰이어 '아': 디튠·비브라토 톱니 성부 → 포먼트 필터 (스테레오)."""
    r = rng("choir", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    L = np.zeros(n); Rr = np.zeros(n)
    for k, f in enumerate(notes):
        gl = 1.0 if glide is None or glide[k] is None else 2.0 ** (glide[k] / 12.0)
        for v in range(nv):
            det = 2 ** (r.normal(0, 10) / 1200)
            vib = 1 + 0.006 * np.sin(TAU * r.uniform(4.6, 5.8) * t + r.uniform(0, TAU)) * np.clip(t / 0.5, 0, 1)
            vib *= 1 + 0.003 * smooth_rand(n, 1.5, r)
            s = osc_saw(f * det * vib * gl, ph0=r.random())
            p = r.uniform(-0.9, 0.9)
            L += s * math.cos((p + 1) * np.pi / 4)
            Rr += s * math.sin((p + 1) * np.pi / 4)
    st = np.stack([L, Rr], 1)
    st += breath * len(notes) * np.stack([white(n, r), white(n, r)], 1)
    return norm(formant(st, vowel))


def brass(midis, dur=2.0, seed=0, peak_fc=2600.0, end_fc=420.0, sustain=0.55):
    """저음 금관 스탭: 디튠 톱니 + 피치 스쿱 + 필터 엔벨로프 + 포화."""
    r = rng("brass", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    x = np.zeros(n)
    scoop = 1 - 0.03 * np.exp(-t / 0.05)
    for m in midis:
        for v in range(3):
            x += osc_saw(float(mtof(m)) * 2 ** ((v - 1) * 7 / 1200) * scoop, ph0=r.random())
    fc = end_fc + (peak_fc - end_fc) * (1 - np.exp(-t / 0.018)) * np.exp(-t / 0.33)
    y = tv_bq(x, "lp", fc, 1.0, stages=2, block=64)
    amp = np.minimum(t / 0.012, 1) * (sustain + (1 - sustain) * np.exp(-t / 0.16)) * np.exp(-t / (0.25 + 1.5 * sustain))
    return norm(np.tanh(1.4 * norm(y) * 1.5) * amp)


def mbox_note(f, dur=1.3, seed=0):
    """오르골 빗살: 기본음 + 높은 6배 부근 모드 + 클릭."""
    r = rng("mbox", seed, float(f))
    n = nsamp(dur)
    t = np.arange(n) / SR
    y = np.sin(TAU * f * t) * np.exp(-t / 0.75)
    y += 0.06 * np.sin(TAU * 2 * f * t) * np.exp(-t / 0.3)
    if 5.95 * f < 0.45 * SR:
        y += 0.22 * np.sin(TAU * 5.95 * f * t + 0.4) * np.exp(-t / 0.10)
    y += 0.25 * bq(r.standard_normal(n), "hp", 4000) * np.exp(-t / 0.0015)
    return y * np.minimum(t / 0.0004, 1)


def epiano(f, dur=1.4, seed=0):
    """DX 계열 FM 일렉 피아노 (80년대 아이돌 팝 색)."""
    t = np.arange(nsamp(dur)) / SR
    idx_ = 1.6 * np.exp(-t / 0.45) + 0.25
    y = np.sin(TAU * f * t + idx_ * np.sin(TAU * f * t))
    y += 0.15 * np.sin(TAU * f * t + 2.0 * np.exp(-t / 0.02) * np.sin(TAU * 14 * f * t)) * np.exp(-t / 0.08)
    return y * np.exp(-t / 1.0) * np.minimum(t / 0.002, 1)


def engine(rpm, seed=0, cyl=4, bright=1.0):
    """엔진: rpm → 점화 주파수 펄스열(사이클별 랜덤) + 배기 공명 + 포화."""
    r = rng("engine", seed)
    n = len(rpm)
    ff = rpm / 60.0 * cyl / 2.0 * (1 + 0.004 * smooth_rand(n, 25, r))
    ph = phase(ff)
    cyc = np.floor(ph).astype(np.int64)
    amp_c = r.normal(1.0, 0.18, int(cyc.max()) + 2)
    frac = ph - cyc
    pulse = np.exp(-frac / 0.08) * (1 - np.exp(-frac / 0.01)) * amp_c[cyc]
    x = pulse - pulse.mean() + 0.25 * np.sin(TAU * ph * 0.5 + 0.3)
    y = bq(x, "peak", 160, 1.0, 8)
    y = bq(y, "peak", 620, 1.5, 5)
    y = np.tanh(2.2 * norm(y) * 1.6)
    y = bq(y, "lp", 3800 * bright, 0.7)
    mech = bq(white(n, r), "bp", 2600, 1.2) * (0.4 + 0.6 * pulse)
    return norm(y + 0.25 * mech)


def seq_bass(steps, step, n, seed=0, fc_hi=1700.0, fc_lo=170.0):
    """16분 펄스 신스 베이스: steps = [(start_sec, midi or None)]."""
    r = rng("bass", seed)
    t = np.arange(n) / SR
    f = np.full(n, float(mtof(38)))
    amp = np.zeros(n)
    fc = np.full(n, fc_lo)
    for st, m in steps:
        i0, i1 = idx(st), min(n, idx(st + step))
        if m is None or i0 >= n:
            continue
        tt = t[i0:i1] - st
        f[i0:i1] = mtof(m)
        gate = np.clip((step * 0.88 - tt) / 0.004, 0, 1)
        amp[i0:i1] = np.minimum(tt / 0.002, 1) * (0.45 + 0.55 * np.exp(-tt / (step * 0.7))) * gate
        fc[i0:i1] = fc_lo + (fc_hi - fc_lo) * np.exp(-tt / 0.045)
    x = 0.6 * osc_saw(f) + 0.6 * osc_saw(f * 1.006, ph0=0.3) + 0.6 * osc_sq(f * 0.5)
    y = tv_bq(x, "lp", fc, 1.5, stages=2, block=32)
    return norm(np.tanh(1.6 * norm(y)) * amp)


def saw_pad(notes, n, seed=0, nv=3, cents=9.0):
    """디튠 톱니 패드: notes = [(midi, t_on, gain)] (스테레오)."""
    r = rng("pad", seed)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for m, ton, g in notes:
        e = np.clip((t - ton) / 0.7, 0, 1) ** 2
        for v in range(nv):
            det = (v - (nv - 1) / 2) * cents + r.normal(0, 2)
            drift = 1 + 0.0015 * smooth_rand(n, 0.7, r)
            s = osc_saw(float(mtof(m)) * 2 ** (det / 1200) * drift, ph0=r.random()) * e * g
            out += pan_st(s, r.uniform(-0.85, 0.85))
    return out


def siren(dur, seed=0):
    r = rng("siren", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    f = 880 + 300 * np.sin(TAU * 0.42 * t - 1.2) + 6 * np.sin(TAU * 5.5 * t)
    ph = phase(f)
    y = np.sin(TAU * ph) + 0.35 * np.sin(TAU * 2 * ph) + 0.15 * np.sin(TAU * 3 * ph)
    y = bq(y, "lp", 1500)
    return norm(y * (0.8 + 0.2 * smooth_rand(n, 0.8, r)))


def projector(dur, seed=0, clatter=1.0):
    """영사기: 24 Hz 클로 클래터 + 셔터 바람 + 모터 화인."""
    r = rng("proj", seed)
    n = nsamp(dur)
    t = np.arange(n) / SR
    per = SR / 24.0
    k = np.arange(int(dur * 24) + 2)
    imp = np.zeros(n)
    i1 = (k * per).astype(int); i2 = (k * per + per * 0.42).astype(int)
    a1 = r.normal(1.0, 0.15, len(k)); a2 = r.normal(0.45, 0.1, len(k))
    ok1 = i1 < n; ok2 = i2 < n
    imp[i1[ok1]] += a1[ok1]
    imp[i2[ok2]] += a2[ok2]
    cl = 2.0 * bq(imp, "bp", 1900, 3.0) + 1.2 * bq(imp, "bp", 3400, 4.0) + 0.8 * bq(imp, "bp", 650, 2.0)
    shutter = bq(white(n, r), "bp", 420, 0.8) * (0.4 + 0.6 * np.abs(np.sin(TAU * 24 * t)))
    motor = (np.sin(TAU * 117 * t) + 0.5 * np.sin(TAU * 234 * t + 1) + 0.25 * np.sin(TAU * 351 * t)) * (1 + 0.05 * np.sin(TAU * 0.7 * t))
    return clatter * cl + 0.25 * shutter + 0.05 * motor


# ═══════════════════════════ 타임라인 ═══════════════════════════
def _timeline():
    """[(kind, name, t0, t1, [(t, label), ...])] (초)."""
    try:
        try:
            from .scenes import TIMELINE, starts
        except ImportError:
            from retro_anime.scenes import TIMELINE, starts
        st = [int(s) for s in starts()]
        names = [str(getattr(m, "NAME", "")) for m in TIMELINE]
        frames = [int(getattr(m, "FRAMES", 0)) for m in TIMELINE]
        mods = list(TIMELINE)
    except Exception:
        names = list(DEFAULT_NAMES)
        frames = list(DEFAULT_FRAMES)
        st = list(np.cumsum([0] + frames[:-1]))
        mods = [None] * len(names)
    out = []
    for i, (nm, s, fr, m) in enumerate(zip(names, st, frames, mods)):
        kind = next((k for k in KINDS if k in nm), KINDS[min(i, len(KINDS) - 1)])
        evs = []
        try:
            for item in (getattr(m, "EVENTS", None) or []):
                try:
                    evs.append(((s + float(item[0])) / FPS, str(item[1]).strip().lower()))
                except Exception:
                    continue
        except Exception:
            evs = []
        evs.sort()
        out.append((kind, nm, s / FPS, (s + fr) / FPS, evs))
    return out


def _first(ev, label, default):
    ts = [t for t, l in ev if l == label]
    return ts[0] if ts else default


def _times(ev, label):
    return [t for t, l in ev if l == label]


# ═══════════════════════════ 장면별 사운드 ═══════════════════════════
def sec_city(s0, s1, ev):
    bus = Bus(s0, s1 + 0.1)
    D = s1 - s0
    n = nsamp(D)
    t = np.arange(n) / SR
    u = t / D
    r = rng("city")
    # 1) 어두운 패드: Dm(add9) → 끝에 Eb(b9) 긴장음, LPF 가 서서히 열린다
    notes = [(38, 0.0, 1.0), (45, 0.0, 0.8), (50, 0.12 * D, 0.7), (53, 0.25 * D, 0.6),
             (57, 0.4 * D, 0.45), (60, 0.55 * D, 0.32), (63, 0.68 * D, 0.30), (64, 0.82 * D, 0.22)]
    pad = saw_pad(notes, n, seed=1)
    fc = 110.0 * (4200 / 110) ** (u ** 1.25)
    pad = tv_bq(pad, "lp", fc, 1.3, stages=2, block=128)
    pad *= (0.08 + 0.92 * u ** 1.6)[:, None]
    bus.add(norm(pad), s0, db(-6), send={"hall": 0.55})
    sub = osc_sin(float(mtof(26)), n) * u ** 1.3
    bus.add(sub, s0, db(-13))
    # 2) 도시 험(교통 럼블 + 50 Hz 전원 험) + 끝으로 갈수록 고속도로 흐름
    hum = np.stack([bq(brown(n, r), "lp", 170) + 0.35 * bq(pink(n, r), "bp", 450, 0.5) for _ in range(2)], 1)
    mains = 0.5 * np.sin(TAU * 50 * t) + 0.25 * np.sin(TAU * 100 * t) + 0.12 * np.sin(TAU * 150 * t)
    hum = norm(hum) * (0.6 + 0.4 * u)[:, None] + 0.08 * mains[:, None]
    bus.add(hum, s0, db(-22), send={"hall": 0.15})
    river = np.stack([bq(pink(n, r), "bp", 700, 0.45) for _ in range(2)], 1) * (np.clip((u - 0.55) / 0.45, 0, 1) ** 2)[:, None]
    bus.add(norm(river), s0, db(-20), send={"room": 0.2})
    # 3) 먼 사이렌 (오른쪽, 큰 잔향)
    sir = siren(D, 3) * np.clip((t - 0.25) / 1.0, 0, 1)
    bus.add(sir, s0, db(-33), pan=0.6, send={"huge": 1.4})
    # 4) 이벤트
    k_pulse = k_blink = k_pass = 0
    for te, lab in ev:
        if lab == "pulse":
            m = (62, 57, 63, 60)[k_pulse % 4]
            b = bell(float(mtof(m)), 3.0, seed=("cp", k_pulse), kind="kempul", decay=1.5) \
                + 0.35 * bell(float(mtof(m + 19)), 3.0, seed=("cpb", k_pulse), kind="gender", decay=0.8)
            bus.add(norm(b), te, db(-14 + 6 * (te - s0) / D), pan=(-0.3, 0.3, 0.0, -0.15)[k_pulse % 4], send={"hall": 0.8})
            k_pulse += 1
        elif lab == "blink":
            m = (81, 82, 86)[k_blink % 3]
            bus.add(bell(float(mtof(m)), 1.6, seed=("bk", k_blink), kind="bonang", decay=0.5), te,
                    db(-22), pan=(0.55, -0.55)[k_blink % 2], send={"hall": 0.9})
            k_blink += 1
        elif lab == "glitch":
            bus.add(neon(0.22, seed=("cg", te)), te, db(-23), pan=float(rng("cgp", te).uniform(-0.6, 0.6)),
                    send={"room": 0.3})
        elif lab == "whoosh":
            bus.add(whoosh(1.1, 0.6, 250, 2200, seed=("cw", te), pan0=-0.5, pan1=0.5), te - 0.66, db(-21),
                    send={"hall": 0.4})
        elif lab == "passby":
            side = (1.0, -1.0)[k_pass % 2]
            sig = passby(lambda m_, r_: src_car(m_, r_, 95.0), 2.0, 1.0, v=30.0, d=11.0, side=side, seed=("cpb", te))
            bus.add(sig, te - 1.0, db(-10 + 2 * k_pass), send={"hall": 0.35})
            k_pass += 1
        elif lab == "swell":
            if te - s0 < 0.25 * D:
                # 오프닝: 첫 벨까지 부드러운 역재생 공
                tgt = next((x for x, l in ev if l == "pulse" and x > te + 0.15), te + 0.45)
                bus.add(rev_swell(tgt - te, seed="c_open", root=38, tonal=0.6, crashy=0.25, bright=0.5),
                        te, db(-20))
            else:
                # 끝: 컷(3.0 s)으로 빨려 들어가는 라이저
                t_start = max(s0, min(te, s1 - 0.25) - 0.75)
                bus.add(rev_swell(s1 - t_start, seed="c_riser", root=50, tonal=0.7, crashy=1.0, curve=2.6),
                        t_start, db(-8))
                rs = whoosh(s1 - t_start, 0.995, 300, 6000, seed="c_rw", pan0=-0.3, pan1=0.3, att=3.0)
                bus.add(rs, t_start, db(-14))
    return bus.render(), bus.i0


def sec_chase(s0, s1, ev):
    bus = Bus(s0, s1 + 0.1)
    D = s1 - s0
    n = nsamp(D)
    t = np.arange(n) / SR
    t_imp = _first(ev, "impact", s0 + 0.806 * D)
    t_skid = _first(ev, "skid", t_imp - 0.75)
    t_cutB = s0 + D * 34 / 72
    step = 60.0 / 132 / 4
    K = int((t_imp - 0.05 - s0) / step) + 1
    fill0 = int(math.ceil((t_skid - s0) / step))
    # 1) 드럼 (타이코/시메/카)
    DON = {0: 1.0, 3: 0.7, 6: 0.55, 8: 0.95, 10: 0.45, 11: 0.75, 14: 0.8}
    MID = {4: 0.9, 7: 0.35, 12: 0.95, 15: 0.4}
    for k in range(K):
        ts = s0 + k * step
        b = k % 16
        if k >= fill0:
            w = (k - fill0 + 1) / max(1, K - fill0)
            if k % 2 == 0:
                bus.add(taiko(68, 1.0, 0.4, seed=("fd", k)), ts, db(-5) * (0.55 + 0.45 * w), send={"room": 0.35})
            else:
                bus.add(taiko(132, 0.6, 0.22, seed=("fm", k), slap=0.8), ts, db(-7) * (0.55 + 0.45 * w),
                        pan=0.25, send={"room": 0.35})
        else:
            if b in DON:
                bus.add(taiko(68, 1.2, 0.42, seed=("d", k)), ts, db(-5) * DON[b], send={"room": 0.4})
            if b in MID:
                bus.add(taiko(132, 0.7, 0.22, seed=("m", k), slap=0.8), ts, db(-8) * MID[b], pan=(0.3, -0.3)[(k // 4) % 2],
                        send={"room": 0.4})
        kv = 0.85 if b in (2, 6, 10, 14) else (0.5 if b % 2 == 0 else 0.28)
        bus.add(ka(seed=("k", k), f=2600 if b % 2 else 2300), ts, db(-18) * kv, pan=-0.35, send={"room": 0.2})
        if b in (0, 8) and k < fill0:
            bus.add(taiko(290, 0.25, 0.08, seed=("sh", k), slap=1.0, thump=0.2), ts + step * 0.5, db(-17), pan=0.4)
    # 2) 펄스 베이스 (16분, 옥타브 점프)
    bar_a = [38, 38, 50, 38, 38, 50, 38, 48, 38, 38, 50, 38, 41, 38, 48, 50]
    bar_b = [34, 34, 46, 34, 36, 36, 48, 36, 38, 38, 50, 38, 38, 50, 38, 50]
    steps = []
    for k in range(K):
        m = (bar_a if (k // 16) % 2 == 0 else bar_b)[k % 16]
        if k >= fill0:
            m = (38, 50)[k % 2]
        steps.append((k * step, m))
    bass = seq_bass(steps, step, n, seed=2)
    bus.add(bass, s0, db(-10), send={"room": 0.1})
    # 3) 엔진 (rpm 곡선: 변속, 스키드 중 다운시프트 블립, 정지 후 아이들)
    a, b = t_skid - s0, t_imp - s0
    pts = sorted([(0.0, 10400), (0.5, 11900), (0.55, 9400), (a * 0.82, 12000), (a * 0.82 + 0.05, 9900), (a, 10900),
                  (a + 0.04, 12700), (a + 0.12, 9900), (a + 0.28, 12300), (a + 0.36, 9300), (a + 0.5, 11600),
                  (a + 0.58, 8600), (b - 0.02, 7200), (b + 0.12, 3200), (b + 0.35, 1700), (D + 0.1, 1450)])
    rpm = np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
    eng = engine(rpm, seed=5)
    eamp = np.where(t < b, 1.0, 0.3 + 0.7 * np.exp(-(t - b) / 0.15))
    bus.add(eng * eamp, s0, db(-14), send={"room": 0.3})
    # 추격자 엔진 (f12~컷B, 왼쪽 뒤에서 다가옴)
    p0 = s0 + D * 12 / 72
    if t_cutB > p0 + 0.1:
        m_ = nsamp(t_cutB - p0)
        tt = np.arange(m_) / SR
        prpm = 9000 + 1500 * tt / tt[-1] + 300 * np.sin(TAU * 1.3 * tt)
        pe = engine(prpm, seed=6, cyl=2, bright=0.7) * (tt / tt[-1]) ** 1.5
        bus.add(edge_fade(pe, 0.01, 0.03), p0, db(-17), pan=-0.55, send={"room": 0.2})
    # 4) 이벤트
    k_pass = 0
    spark_ts = _times(ev, "spark")
    for te, lab in ev:
        if lab == "whoosh":
            if te - s0 < 0.03:
                bus.add(whoosh(0.5, 0.08, 400, 4500, seed=("w0", te), pan0=0.0, pan1=0.0, att=1.0), te, db(-9))
            else:
                bus.add(whoosh(0.8, 0.55, 200, 2600, seed=("w", te), pan0=0.4, pan1=-0.4), te - 0.44, db(-11))
        elif lab == "passby":
            if k_pass % 2 == 0:
                sig = passby(src_wind, 1.0, 0.5, v=62.0, d=2.2, side=1.0, seed=("pl", te), mode="side")
                bus.add(sig, te - 0.5, db(-7), send={"room": 0.1})
            else:
                sig = passby(lambda m_, r_: src_car(m_, r_, 120.0), 1.0, 0.5, v=85.0, d=3.5, side=-1.0,
                             seed=("pc", te), mode="side")
                bus.add(sig, te - 0.5, db(-7), send={"room": 0.1})
            k_pass += 1
        elif lab == "flash":
            bus.add(crash(2.0, seed=("fc", te)), te, db(-12), pan=-0.2, send={"room": 0.2})
            bus.add(glint(seed=("fg", te), f=5200), te, db(-18), pan=-0.4)
        elif lab == "skid":
            dur = t_imp + 0.06 - te
            bus.add(screech(dur, seed=("sk", te)), te, db(-4), pan=0.25, send={"room": 0.2})
            smoke = bq(white(nsamp(dur + 0.4), rng("smk", te)), "lp", 2500) * env_perc(nsamp(dur + 0.4), 0.2, 0.4)
            bus.add(norm(smoke), te, db(-22), pan=0.1)
        elif lab == "spark":
            bus.add(crackle(0.35, 300, seed=("sp", te), tau=0.09), te, db(-2), send={"room": 0.15})
        elif lab == "impact":
            bus.add(hit(seed=("ih", te), big=1.2), te, db(0), send={"room": 0.3, "hall": 0.25})
            bus.add(taiko(50, 2.0, 0.75, seed=("it", te), drive=2.0), te, db(-2), send={"hall": 0.3})
            bus.add(crash(2.5, seed=("ic", te), decay=1.1), te, db(-9), send={"hall": 0.2})
        elif lab == "shockwave":
            bus.add(whoosh(0.9, 0.06, 2400, 2400, 180, seed=("sw", te), pan0=-0.8, pan1=0.8, att=1.0, rel=1.2),
                    te, db(-12))
            bus.add(thoom(0.9, seed=("st", te)), te, db(-8))
        elif lab == "swell":
            if s1 - te > 0.15:
                bus.add(rev_swell(s1 - te, seed="ch_sw", root=50, tonal=0.8, crashy=1.0, curve=2.0), te, db(-6))
        elif lab == "blink":
            bus.add(glint(seed=("bl", te), f=5600, dur=0.5), te, db(-15), pan=0.3, send={"room": 0.3})
    if spark_ts:
        ts0 = spark_ts[0] - 0.02
        bus.add(scrape(t_imp - ts0 + 0.03, seed=7) * np.linspace(0.6, 1.0, nsamp(t_imp - ts0 + 0.03)),
                ts0, db(-13), pan=0.15)
    return bus.render(), bus.i0


def sec_eye(s0, s1, ev):
    bus = Bus(s0, s1 + 0.6)
    D = s1 - s0
    n = nsamp(D)
    t = np.arange(n) / SR
    r = rng("eye")
    t_imp = _first(ev, "impact", s0 + D * 20 / 60)
    t_sw = _first(ev, "swell", s0 + D * 45 / 60)
    # 숨결 노이즈 패드 (핑크/블루 = 좌/우 디코릴레이션), 느리게 호흡
    br = np.stack([formant(pink(n, r), "h") for _ in range(2)], 1)
    breath = 0.5 + 0.5 * np.sin(TAU * 0.42 * t - np.pi / 2)
    swell = 1 + 1.6 * np.clip((t - (t_sw - s0)) / (s1 - t_sw + 1e-3), 0, 1) ** 2
    bus.add(norm(br) * (breath * np.minimum(t / 0.6, 1) * swell)[:, None], s0, db(-29), send={"plate": 0.6})
    # 이명: 7.35 kHz 두 사인(맥놀이), 충격 뒤 커진다
    ti = np.sin(TAU * 7350 * t) + 0.6 * np.sin(TAU * 7356.5 * t)
    ji = t_imp - s0
    tenv = 0.22 + np.where(t > ji, 0.9 * np.exp(-(t - ji) / 1.1), 0) * np.clip((t - ji) / 0.15, 0, 1)
    bus.add(norm(ti) * tenv * np.minimum(t / 0.3, 1), s0, db(-40), send={"plate": 0.2})
    # 아주 낮은 드론
    dr = (osc_sin(float(mtof(26)), n) + 0.5 * osc_sin(float(mtof(33)) * 1.002, n)) * np.minimum(t / 1.0, 1)
    bus.add(norm(dr), s0, db(-30))
    # 첫 박동 (이벤트 이전, 아주 작게)
    hb_first = _first(ev, "heartbeat", s1)
    if hb_first - s0 > 0.9:
        bus.add(heartbeat(seed="e0", size=0.9), s0 + 0.38, db(-12), send={"plate": 0.1})
    k_sp = 0
    for te, lab in ev:
        if lab == "blink":
            bus.add(tick(seed=("eb", te), f=1500), te, db(-23), send={"plate": 0.3})
        elif lab == "pulse":
            bus.add(neon(0.3, seed=("ep", te), chops=4), te, db(-24), pan=-0.4, send={"plate": 0.4})
        elif lab == "passby":
            sig = passby(lambda m_, r_: src_car(m_, r_, 90.0), 1.6, 0.8, v=22.0, d=16.0, side=-1.0, seed=("epb", te))
            sig = bq(sig, "lp", 900)
            bus.add(norm(sig), te - 0.8, db(-26), send={"plate": 0.4})
        elif lab == "impact":
            m_ = nsamp(0.9)
            tt = np.arange(m_) / SR
            cl = sum(osc_saw(float(mtof(mm)) * (1 + 0.004 * j), m_, j * 0.3) for j, mm in enumerate((86, 87, 92, 93)))
            cl = bq(cl, "bp", 2600, 0.9) * env_perc(m_, 0.003, 0.28)
            bus.add(norm(cl), te, db(-16), send={"plate": 0.9})
            bus.add(bq(hit(seed=("ej", te), big=0.6), "lp", 220), te, db(-12))
            sk = whoosh(0.4, 0.98, 300, 5000, seed=("es", te), pan0=0.3, pan1=-0.1, att=3.0)
            bus.add(sk, te - 0.4, db(-22))
        elif lab == "flash":
            m_ = nsamp(0.06)
            bus.add(norm(bq(white(m_, rng("ef", te)), "hp", 6000) * env_perc(m_, 0.001, 0.015)), te, db(-26))
        elif lab == "shockwave":
            m_ = nsamp(0.6)
            tt = np.arange(m_) / SR
            vw = osc_sin(70 + 350 * np.exp(-tt / 0.12), m_) * env_perc(m_, 0.005, 0.2)
            bus.add(vw, te, db(-20), send={"plate": 0.5})
        elif lab == "tear":
            bus.add(chime(float(mtof(93)), seed=("et", te), dur=2.5, decay=0.9), te, db(-21), pan=0.2,
                    send={"plate": 1.0})
        elif lab == "heartbeat":
            bus.add(heartbeat(seed=("eh", te)), te, db(-6), send={"plate": 0.12})
        elif lab == "spark":
            m = (100, 98)[k_sp % 2]
            bus.add(chime(float(mtof(m)), seed=("esp", te), dur=1.4, decay=0.4), te, db(-27), pan=0.35,
                    send={"plate": 1.0})
            k_sp += 1
        elif lab == "swell":
            if s1 - te > 0.2:
                # 다음 컷의 오르골을 예고하는 역재생 음 + 낮은 드론 상승
                m_ = nsamp(s1 - te)
                note = mbox_note(float(mtof(84)), 1.6, seed="eye_rev")
                st = reverb(note, "plate")[:m_]
                if len(st) < m_:
                    st = np.concatenate([st, np.zeros((m_ - len(st), 2))])
                st = st[::-1] * (np.linspace(0, 1, m_) ** 2)[:, None]
                bus.add(norm(edge_fade(st, 0.01, 0.004)), te, db(-24))
    return bus.render(), bus.i0


def sec_mirror(s0, s1, ev):
    bus = Bus(s0, s1 + 0.1)
    D = s1 - s0
    n = nsamp(D)
    t = np.arange(n) / SR
    r = rng("mirror")
    t_crack = _first(ev, "impact", s0 + D * 18 / 60)
    t_shat = _first(ev, "shatter", s0 + D * 30 / 60)
    t_sw = _first(ev, "swell", s0 + D * 52 / 60)
    glitches = _times(ev, "glitch")
    # 분장실 형광등/전구 험
    hum = (np.sin(TAU * 120 * t) + 0.5 * np.sin(TAU * 240 * t) + 0.3 * np.sin(TAU * 360 * t))
    hum = hum + 0.3 * bq(white(n, r), "bp", 3000, 2.0)
    bus.add(norm(hum) * np.minimum(t / 0.2, 1), s0, db(-38))
    # ── 아이돌 팝 오르골 조각 (F장조) — 점점 어긋난다
    L = nsamp(D + 0.6)
    idol = np.zeros((L, 2))
    step = 0.19
    mel = [84, 81, 77, 81, 82, 81, 79, 77, 76, 79, 84, 86, 84, 81, 79, 77]
    acc = [65, 72, 69, 72, 62, 69, 65, 69, 58, 65, 62, 65, 60, 67, 64, 67]
    chords = [(53, 57, 60), (50, 53, 57), (46, 50, 53), (48, 52, 55)]
    span = max(t_shat - s0, 0.3)

    def put(sig, tl, g, p):
        a = idx(tl)
        b_ = min(L, a + len(sig))
        if b_ > a:
            idol[a:b_] += pan_st(sig[: b_ - a], p) * g

    for k, m in enumerate(mel):
        tl = 0.03 + k * step
        if tl > D:
            break
        w = min(1.0, tl / span)
        cents = r.normal(0, 4 + 45 * w ** 1.5)
        if s0 + tl > t_crack and r.random() < 0.35:
            m += int(r.choice([-1, 1]))
        put(mbox_note(float(mtof(m)) * 2 ** (cents / 1200), seed=("mb", k)), tl, 1.0, r.uniform(-0.25, 0.25))
        put(mbox_note(float(mtof(acc[k])) * 2 ** (r.normal(0, 4 + 30 * w) / 1200), seed=("ma", k)), tl, 0.4,
            r.uniform(-0.5, 0.5))
        if k % 4 == 0:
            for j, cm in enumerate(chords[(k // 4) % 4]):
                put(epiano(float(mtof(cm)) * 2 ** (r.normal(0, 3 + 25 * w) / 1200), 1.0, seed=("ep", k, j)), tl,
                    0.22, (-0.4, 0.0, 0.4)[j])
    clean = idol.copy()
    tl = np.arange(L) / SR
    ts = t_shat - s0
    wv = np.clip(tl / span, 0, 1)
    wow = (0.0003 + 0.0042 * wv ** 2) * (0.7 * np.sin(TAU * 1.3 * tl) + 0.3 * smooth_rand(L, 6, r))
    flutter = 0.00012 * wv * np.sin(TAU * 9.0 * tl)
    pos = tl - wow - flutter - 0.006
    # 파쇄 순간 테이프 스톱
    after = tl > ts
    rate = np.clip(1 - (tl - ts) / 0.4, 0, 1) ** 1.4
    p_ts = np.interp(ts, tl, pos)
    cum = np.cumsum(np.where(after, rate, 0.0)) / SR
    pos = np.where(after, p_ts + cum, pos)
    y = vread(idol, pos * SR)
    y *= np.where(after, np.clip(1 - (tl - ts) / 0.42, 0, 1) ** 0.6, 1.0)[:, None]
    # 파편 반사들: 깨끗한 원본 조각을 다른 피치/역재생/팬으로
    ft = ts + 0.2
    while ft < (t_sw - s0) + 0.15 and ft < D - 0.1:
        fr = 2 ** (float(r.choice([-7, -5, -1, 2, 5, -12])) / 12)
        src0 = r.uniform(0.05, max(0.1, ts - 0.45))
        dur = r.uniform(0.18, 0.34)
        m_ = nsamp(dur)
        seg = vread(clean, (src0 + np.arange(m_) / SR * fr) * SR)
        if r.random() < 0.35:
            seg = seg[::-1]
        seg = edge_fade(seg, 0.008, 0.03) * 0.75
        a = idx(ft)
        b_ = min(L, a + m_)
        y[a:b_] += pan_st(seg[: b_ - a], r.uniform(-0.85, 0.85))
        ft += r.uniform(0.12, 0.24)
    # 글리치 스터터 (버퍼 반복)
    for tg in glitches:
        a = idx(tg - s0)
        if a < 0 or a >= L - 100:
            continue
        rr = rng("stut", tg)
        sl = nsamp(float(rr.choice([0.03, 0.045, 0.06])))
        reps = int(rr.integers(3, 6))
        seg = edge_fade(y[a:a + sl].copy(), 0.002, 0.003)
        for q in range(reps):
            c0 = a + q * sl
            c1 = min(L, c0 + sl)
            if c0 >= L:
                break
            y[c0:c1] = seg[: c1 - c0]
        e = min(L, a + reps * sl)
        f_ = min(L - e, nsamp(0.004))
        y[e:e + f_] *= np.linspace(0, 1, f_)[:, None]
    bus.add(y, s0, db(-12), send={"plate": 0.35})
    # ── 이벤트
    for te, lab in ev:
        rr = rng("mev", te, lab)
        if lab == "heartbeat":
            bus.add(heartbeat(seed=("mh", te)), te, db(-9), send={"room": 0.15})
        elif lab == "pulse":
            bus.add(thoom(0.7, seed=("mp", te), f_hi=60, f_lo=45), te, db(-16), send={"plate": 0.3})
        elif lab == "blink":
            bus.add(tick(seed=("mbk", te), f=2200), te, db(-24), pan=0.2, send={"plate": 0.4})
        elif lab == "glitch":
            big = abs((te - s0) * FPS - 44) < 0.6
            dur = 0.3 if big else float(rr.uniform(0.07, 0.16))
            bus.add(static_burst(dur, seed=("st", te)), te, db(-8 if big else -12), send={"room": 0.15})
        elif lab == "impact":
            bus.add(glass(0.9, seed=("crk", te), ngr=140, tau=0.05, tail=0.05, crack=1.2), te, db(-3),
                    send={"room": 0.3})
            bus.add(bq(hit(seed=("mih", te), big=0.7), "lp", 300), te, db(-8))
        elif lab == "flash":
            if t_sw - 0.05 <= te:
                m_ = nsamp(max(0.05, s1 - te))
                hiss_ = bq(white(m_, rr), "hp", 4500) * np.linspace(0.2, 1, m_) ** 2
                bus.add(norm(edge_fade(hiss_, 0.01, 0.003)), te, db(-20))
            else:
                bus.add(whoosh(0.18, 0.15, 6000, 9000, 3000, seed=("mf", te), pan0=-0.2, pan1=0.2, att=1.0),
                        te, db(-18))
        elif lab == "spark":
            for q in range(3):
                tq = te + q * float(rr.uniform(0.02, 0.07))
                m_ = nsamp(0.08)
                pop = 6.0 * bq(white(m_, rr), "lp", 1500) * env_perc(m_, 0.0005, 0.01)
                bus.add(norm(pop), tq, db(-17), pan=float(rr.uniform(-0.8, 0.8)), send={"room": 0.3})
            bus.add(glass(0.6, seed=("bulb", te), ngr=40, tau=0.05, tail=0.1, crack=0.2), te + 0.01, db(-16),
                    send={"room": 0.3})
        elif lab == "shatter":
            bus.add(glass(2.2, seed=("sh", te), ngr=1000, tau=0.13, tail=0.3, crack=1.0), te, db(3),
                    send={"room": 0.25, "hall": 0.3})
            bus.add(hit(seed=("msh", te), big=0.8), te, db(-6), send={"hall": 0.2})
        elif lab == "shockwave":
            bus.add(thoom(1.0, seed=("msw", te), f_hi=80, f_lo=34), te, db(-9))
        elif lab == "whoosh":
            bus.add(whoosh(0.9, 0.45, 300, 5000, seed=("mw", te), pan0=-0.8, pan1=0.8), te - 0.1, db(-13),
                    send={"plate": 0.2})
        elif lab == "passby":
            def src_shard(m_, r_):
                tt_ = np.arange(m_) / SR
                return src_wind(m_, r_) + 0.5 * np.sin(TAU * 2600 * tt_) + 0.3 * np.sin(TAU * 4100 * tt_)
            sig = passby(src_shard, 1.2, 0.6, v=40.0, d=1.8, side=1.0, seed=("mpb", te))
            bus.add(sig, te - 0.6, db(-11), send={"plate": 0.3})
        elif lab == "swell":
            t_st = max(s0, te - 0.7)
            bus.add(rev_swell(s1 - t_st, seed="m_sw", root=53, tonal=1.0, crashy=1.0, curve=2.4), t_st, db(6))
    return bus.render(), bus.i0


def sec_blast(s0, s1, ev):
    bus = Bus(s0, s1 + 0.1)
    D = s1 - s0
    n = nsamp(D)
    r = rng("blast")
    booms = _times(ev, "boom") or [s0 + D * 14 / 60]
    t_b1 = booms[0]
    impacts = _times(ev, "impact")
    t_final = max([x for x in impacts if x > t_b1 + 0.3] or [s0 + D * 40 / 60])
    swells = _times(ev, "swell")
    t_white = max(swells) if swells and max(swells) > t_b1 + 0.5 else s0 + D * 47 / 60
    # 콰이어 '아': 첫 폭발 뒤 → 컷까지 상승, 마지막엔 윗성부가 끌어 올라감
    c0 = t_b1 + 0.05
    cd = s1 - c0
    if cd > 0.3:
        notes = [float(mtof(m)) for m in (50, 57, 62, 65, 69, 74)]
        m_ = nsamp(cd)
        tt = np.arange(m_) / SR
        rise = np.clip((tt - (t_white - c0)) / max(0.1, s1 - t_white), 0, 1) ** 1.5
        glide = [None, None, None, 2.0 * rise, 3.0 * rise, 5.0 * rise]
        ch = choir(notes, cd, seed=9, glide=glide)
        ch *= (0.15 + 0.85 * (tt / cd) ** 2.2)[:, None]
        bus.add(ch, c0, db(-4), send={"huge": 0.6})
    # 이벤트
    k_sw = 0
    for te, lab in ev:
        rr = rng("bev", te, lab)
        if lab == "flash":
            if te - s0 < 0.03:
                m_ = nsamp(0.9)
                tt = np.arange(m_) / SR
                ring = sum(np.sin(TAU * f * tt) for f in (3520.0, 3527.0, 4699.0, 6272.0)) * env_perc(m_, 0.002, 0.25)
                bus.add(norm(ring), te, db(-27), send={"hall": 0.5})
            else:
                m_ = nsamp(0.15)
                bus.add(norm(bq(white(m_, rr), "hp", 3000) * env_perc(m_, 0.001, 0.03)), te, db(-16))
        elif lab == "swell":
            if te < t_b1 - 0.1:
                if k_sw == 0:
                    # 빛의 씨앗으로 수렴: 상승 사인 글라이드
                    m_ = nsamp(t_b1 - 0.04 - te)
                    tt = np.arange(m_) / SR
                    u = tt / tt[-1]
                    f = 220 * 4 ** (u ** 1.3)
                    tone = (osc_sin(f) + 0.5 * osc_sin(f * 1.5) + 0.3 * osc_sin(f * 2.01)) * (u ** 1.5)
                    tone *= 1 + 0.5 * u * np.sin(TAU * (6 + 20 * u) * tt)
                    bus.add(norm(edge_fade(tone, 0.02, 0.006)), te, db(-25), send={"hall": 0.4})
                else:
                    # 어둠 속 들숨: 역재생 노이즈 → 폭발 직전 정적
                    t_st = max(s0, min(te, t_b1 - 0.3) - 0.25)
                    sk = whoosh(t_b1 - 0.03 - t_st, 0.985, 200, 7000, seed=("inh", te), pan0=0.6, pan1=0.0, att=3.2)
                    bus.add(sk, t_st, db(-19))
                k_sw += 1
            elif s1 - te > 0.15:
                # 화이트아웃 바람
                m_ = nsamp(s1 - te)
                u = np.linspace(0, 1, m_)
                wh = np.stack([bq(white(m_, rr), "hp", 2500) for _ in range(2)], 1) * (u ** 2)[:, None]
                bus.add(norm(edge_fade(wh, 0.02, 0.003)), te, db(-12))
        elif lab == "heartbeat":
            bus.add(thoom(1.0, seed=("bh", te), f_hi=58, f_lo=36), te, db(-19), send={"huge": 0.3})
            bus.add(heartbeat(seed=("bhh", te), size=1.2), te, db(-21))
        elif lab == "boom":
            big = 1.0 if te == t_b1 else 0.85
            bus.add(boom(3.2 if te == t_b1 else 2.6, seed=("bm", te), crack=big), te, db(0 if te == t_b1 else -1.5),
                    send={"huge": 0.18})
        elif lab == "impact":
            bus.add(hit(seed=("bi", te), big=1.0), te, db(-6), send={"huge": 0.2})
        elif lab == "shockwave":
            bus.add(whoosh(1.4, 0.12, 3200, 3200, 220, seed=("bsw", te), pan0=-0.9, pan1=0.9, att=1.0, rel=1.0,
                           body=0.9), te, db(-7), send={"huge": 0.1})
        elif lab == "shatter":
            bus.add(crunch(1.5, seed=("bc", te)), te, db(-8), pan=float(rr.uniform(-0.5, 0.5)), send={"hall": 0.3})
        elif lab == "whoosh":
            bus.add(whoosh(0.9, 0.5, 200, 2500, seed=("bw", te), pan0=0.7, pan1=-0.7), te - 0.45, db(-11))
        elif lab == "passby":
            def src_debris(m_, r_):
                return src_wind(m_, r_) + 0.7 * bq(r_.standard_normal(m_), "lp", 300)
            sig = passby(src_debris, 1.0, 0.5, v=90.0, d=2.5, side=float(rr.choice([-1.0, 1.0])), seed=("bp", te))
            bus.add(sig, te - 0.5, db(-9))
    # 충격파가 카메라에 도달한 뒤의 폭풍 벽
    if s1 - t_final > 0.2:
        m_ = nsamp(s1 - t_final)
        u = np.linspace(0, 1, m_)
        ch_ = []
        for c in range(2):
            x = pink(m_, r)
            x = tv_bq(x, "lp", 1800 * (6.0 ** u), 0.8, block=128)
            ch_.append(np.tanh(2.5 * norm(x)))
        wall = np.stack(ch_, 1) * (np.minimum(u / 0.04, 1) * (0.7 + 0.3 * u))[:, None]
        bus.add(norm(edge_fade(wall, 0.005, 0.003)), t_final, db(-9), send={"huge": 0.1})
    # 정전 파도 (창문이 꺼짐): 험 피치 다운 + 릴레이 '턱' 소리, 폭발 전
    p0, p1 = s0 + D * 8 / 60, t_b1 - 0.05
    if p1 - p0 > 0.15:
        m_ = nsamp(p1 - p0)
        tt = np.arange(m_) / SR
        u = tt / tt[-1]
        hf = 50 * (1 - 0.6 * u)
        ph_ = phase(hf)
        hm = (np.sin(TAU * ph_) + 0.6 * np.sin(TAU * 2 * ph_) + 0.4 * np.sin(TAU * 3 * ph_)) * (1 - u) ** 0.7
        bus.add(norm(hm), p0, db(-27))
        for q in range(6):
            tq = p0 + (q + 0.3) / 6.5 * (p1 - p0)
            bus.add(bq(tick(seed=("rel", q), f=300, dur=0.12), "lp", 900), tq, db(-25), pan=(-1) ** q * 0.15 * (q + 1))
    out = bus.render()
    # 화이트아웃: 마지막 구간에서 저역을 걷어내고 밝게 포화
    lt = bus.local_t()
    w = np.clip((lt - t_white) / max(0.1, s1 - t_white), 0, 1)
    if w.max() > 0:
        pk = np.max(np.abs(out)) + 1e-9
        out = tv_bq(out, "hp", 20 + 480 * w ** 1.5, 0.7, stages=2, block=128)
        sat = np.tanh(2.2 * bq(out, "hp", 1500) / pk) * pk
        out = out * (1 - 0.35 * w[:, None]) + 0.6 * sat * w[:, None]
    return out, bus.i0


def sec_title(s0, s1, ev):
    bus = Bus(s0, s1 + 0.05)
    D = s1 - s0
    t_run = _first(ev, "runout", s0 + D * 28 / 36)
    t_burn = _first(ev, "tear", s0 + D * 32 / 36)
    dropouts = []
    for te, lab in ev:
        rr = rng("tev", te, lab)
        if lab == "title_hit":
            bus.add(taiko(46, 2.2, 0.6, seed=("odaiko", te), drive=2.2, slap=0.8), te, db(0), send={"huge": 0.2})
            bus.add(bq(gong(62.0, 3.5, seed=("tg", te), bloom=0.35), "hp", 110), te, db(-3), send={"huge": 0.7})
            bus.add(brass((26, 38, 45, 50, 53, 57), 2.5, seed=("tb", te), sustain=0.12), te, db(-3), send={"hall": 0.6})
            bus.add(thoom(1.0, seed=("tt", te), f_hi=60, f_lo=29), te, db(-6))
        elif lab == "impact":
            bus.add(crash(2.5, seed=("tc", te), decay=1.2), te, db(-10), send={"huge": 0.3})
        elif lab == "flash":
            if te < t_run:
                bus.add(glint(seed=("tf", te), f=6200, dur=0.4), te, db(-20), send={"huge": 0.3})
        elif lab == "shockwave":
            bus.add(whoosh(0.9, 0.08, 3000, 3000, 300, seed=("tsw", te), pan0=-0.9, pan1=0.9, att=1.0), te, db(-15))
        elif lab == "pulse":
            if te - s0 < 0.4:
                bus.add(ka(seed=("tp", te), f=2800), te, db(-14), send={"huge": 0.4})
                bus.add(bell(float(mtof(86)), 2.0, seed=("tpb", te), kind="gender", decay=0.7), te, db(-20),
                        pan=0.3, send={"huge": 0.5})
            else:
                bus.add(bq(taiko(60, 1.5, 0.5, seed=("tep", te)), "lp", 400), te, db(-12), send={"huge": 1.0})
        elif lab == "blink":
            bus.add(neon(0.06, seed=("tbk", te), chops=1), te, db(-17), pan=0.4)
        elif lab == "swell":
            cd = min(1.4, s1 - te)
            if cd > 0.2:
                ch = choir([float(mtof(m)) for m in (38, 45, 50, 57)], cd, seed=11, vowel="o", nv=3)
                m_ = len(ch)
                ch *= np.sin(np.linspace(0, np.pi, m_))[:, None] ** 1.5
                bus.add(ch, te, db(-16), send={"huge": 0.5})
        elif lab == "spark":
            bus.add(glint(seed=("tsp", te), f=5000), te, db(-17), pan=0.35, send={"huge": 0.4})
        elif lab == "glitch":
            bus.add(norm(projector(0.12, seed=("flap", te)) + 0.5 * white(nsamp(0.12), rr)), te, db(-12))
            dropouts.append(te)
    out = bus.render()
    lt = bus.local_t()
    # 런아웃: 24 Hz 플러터 + 회전이 느려지는 와우 + 프레임 슬립 드롭아웃
    if t_run < s1:
        w = np.clip((lt - t_run) / max(0.1, s1 - t_run), 0, 1)
        rate = 1 - 0.35 * w ** 1.5
        pos = np.where(lt < t_run, lt - s0, 0.0)
        cum = np.cumsum(np.where(lt >= t_run, rate, 0.0)) / SR
        pos = np.where(lt >= t_run, (t_run - s0) + cum, pos)
        out = vread(out, pos * SR)
        out *= (1 - 0.6 * w * (0.5 + 0.5 * np.sin(TAU * 24 * lt)) ** 2)[:, None]
    for tdp in dropouts:
        g = 1 - 0.85 * np.exp(-((lt - tdp - 0.02) / 0.02) ** 2)
        out *= g[:, None]
    # 필름 번(타는 소리)으로 음악이 녹아 사라짐
    fade = np.clip(1 - (lt - t_burn) / max(0.05, s1 - 0.03 - t_burn), 0, 1) ** 1.5
    out *= fade[:, None]
    # 영사기 클래터 (런아웃부터 크게)
    a0 = max(s0, t_run - 0.05)
    m_ = nsamp(s1 - a0)
    pj = projector(s1 - a0, seed="runout", clatter=1.0)
    u = np.linspace(0, 1, m_)
    pj *= np.minimum(u / 0.15, 1) * (1 - u) ** 0.5
    bus2 = Bus(s0, s1 + 0.05)
    bus2.add(norm(pj), a0, db(-15), pan=0.1, send={"room": 0.3})
    t_fl = max([x for x, l in ev if l == "flash" and x > t_run] or [t_burn])
    bd = s1 - t_fl
    if bd > 0.05:
        cr = crackle(bd, 900, seed="burn", f_lo=1200, f_hi=6000)
        m2 = len(cr)
        uu = np.linspace(0, 1, m2)
        sizzle = np.stack([bq(white(m2, rng("sz", c)), "bp", 3500, 0.6) for c in range(2)], 1)
        cr = (norm(cr) + 0.6 * norm(sizzle)) * (np.minimum(uu / 0.1, 1) * (1 - uu) ** 0.8)[:, None]
        bus2.add(norm(cr), t_fl, db(-13))
    out = out + bus2.render()
    return out, bus.i0


def bed(T):
    """전 구간 아주 작은 영사기 소리 + 레코드 히스/크래클."""
    r = rng("bed")
    n = nsamp(T)
    hiss = np.stack([bq(bq(white(n, r), "hp", 2500), "lp", 11000) for _ in range(2)], 1)
    imp = np.zeros((n, 2))
    cnt = r.poisson(6 * T)
    ii = r.integers(0, n, cnt)
    np.add.at(imp[:, 0], ii, r.exponential(1, cnt) ** 2 * r.choice([-1, 1], cnt))
    ii = r.integers(0, n, cnt)
    np.add.at(imp[:, 1], ii, r.exponential(1, cnt) ** 2 * r.choice([-1, 1], cnt))
    pops = bq(imp, "hp", 900) + 0.3 * bq(imp, "bp", 300, 1.0)
    pj = projector(T, seed="bed", clatter=0.35)
    pj = np.stack([pj, np.roll(pj, 37)], 1)
    return db(-56) * norm(hiss) * 4 + db(-44) * norm(pops) + db(-50) * norm(pj) * 2


# ═══════════════════════════ 마스터 ═══════════════════════════
SECTIONS = {"city": sec_city, "chase": sec_chase, "eye": sec_eye,
            "mirror": sec_mirror, "blast": sec_blast, "title": sec_title}
# (목표 RMS dBFS, 페이드인 s, 페이드아웃 s, 꼬리 허용 s)
SEC_MIX = {"city": (-21.0, 0.004, 0.006, 0.0), "chase": (-12.8, 0.002, 0.008, 0.0),
           "eye": (-31.0, 0.004, 0.25, 0.25), "mirror": (-16.0, 0.002, 0.004, 0.0),
           "blast": (-12.5, 0.002, 0.004, 0.0), "title": (-15.0, 0.002, 0.03, 0.0)}


def _gate(N, i0, n, s0, s1, fi, fo, tail):
    """구간 [s0, s1(+tail)] 창 (마스터 좌표), 짧은 코사인 페이드."""
    t = (i0 + np.arange(n)) / SR
    g = np.clip((t - s0) / max(fi, 1e-4), 0, 1)
    end = s1 + tail
    g *= np.clip((end - t) / max(fo, 1e-4), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * g)


def softclip(x, thr=0.55):
    a = np.abs(x)
    over = a > thr
    y = x.copy()
    y[over] = np.sign(x[over]) * (thr + (1 - thr) * np.tanh((a[over] - thr) / (1 - thr)))
    return y


def limiter(x, ceil_db=-1.3, win=0.012):
    c = db(ceil_db)
    pk = np.max(np.abs(x), axis=1)
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = nsamp(win)
    gm = minimum_filter1d(g, size=2 * L + 1, mode="nearest")
    gs = uniform_filter1d(gm, size=L, mode="nearest")
    y = x * np.minimum(gs, g + 1.0)[:, None]
    return np.clip(y, -c, c)


@lru_cache(maxsize=2)
def _master(total_frames=None):
    tl = _timeline()
    T = tl[-1][3] if tl else 15.0
    if total_frames:
        T = max(T, total_frames / FPS)
    N = nsamp(T)
    mix = np.zeros((N, 2))
    stats = []
    hp = signal.butter(2, 24, "hp", fs=SR, output="sos")
    for kind, name, s0, s1, ev in tl:
        fn = SECTIONS.get(kind)
        if fn is None or s1 <= s0:
            continue
        sec, i0 = fn(s0, s1, ev)
        sec = signal.sosfilt(hp, sec, axis=0)      # 서브소닉/DC 제거는 게이트 전에 (컷 뒤 링잉 방지)
        tgt, fi, fo, tail = SEC_MIX[kind]
        g = _gate(N, i0, len(sec), s0, s1, fi, fo, tail)
        sec = sec * g[:, None]
        a, b = max(i0, 0), min(i0 + len(sec), N)
        sec = sec[a - i0: b - i0]
        i_s, i_e = idx(s0) - a, idx(s1) - a
        rms = math.sqrt(float(np.mean(sec[max(i_s, 0):max(i_e, 1)] ** 2)) + 1e-12)
        gain = float(np.clip(db(tgt) / rms, db(-30), db(30)))
        mix[a:b] += sec * gain
        stats.append((name, s0, s1, 20 * math.log10(rms), 20 * math.log10(gain)))
    mix += signal.sosfilt(hp, bed(N / SR)[:N], axis=0)
    # 마스터: 소프트 클립 → 룩어헤드 리미터 → 페이드 → TPDF 디더
    mix = softclip(mix, 0.55)
    mix = limiter(mix, -1.3)
    t = np.arange(N) / SR
    fade = np.clip(t / 0.12, 0, 1) * np.clip((T - t) / 0.06, 0, 1)
    mix *= fade[:, None]
    mix -= mix.mean(axis=0, keepdims=True) * fade[:, None]
    r = rng("dither")
    d = (r.random((N, 2)) - r.random((N, 2))) / 32768.0
    out = np.clip(np.round((mix + d) * 32767.0), -32767, 32767).astype(np.int16)
    _master.stats = stats
    return out


def render_audio(path, start_frame=0, end_frame=360, fps=24):
    """[start_frame, end_frame) 구간의 48 kHz 16-bit 스테레오 WAV 를 path 에 쓴다."""
    try:
        from .scenes import total_frames
        tf = int(total_frames())
    except Exception:
        tf = sum(DEFAULT_FRAMES)
    full = _master(max(tf, int(end_frame * FPS / fps + 0.999)))
    n = int(round((end_frame - start_frame) / fps * SR))
    a = int(round(start_frame / fps * SR))
    seg = full[a:a + n]
    if len(seg) < n:
        seg = np.concatenate([seg, np.zeros((n - len(seg), 2), np.int16)])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.ascontiguousarray(seg).tobytes())
    return path
