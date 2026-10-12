"""35mm 필름 프린트 룩 후처리.

장면이 만든 '깨끗한 셀 촬영 결과'에 필름 고유의 결함을 입힌다:
할레이션 · 블룸 · 색 바램/그레이딩 · 색수차 · 소프트니스 · 게이트 위브(프레임 흔들림)
· 플리커 · 그레인 · 먼지/머리카락/스크래치 · 비네트.
모든 무작위성은 전체 프레임 번호로 결정되므로 병렬 렌더해도 결과가 같다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .core import H, W, blur, hash01, wobble

FILM = dict(
    halation=0.55,        # 밝은 부분 주위의 붉은 번짐 (필름 베이스 반사)
    halation_thresh=0.72,
    bloom=0.10,
    lift=(0.030, 0.026, 0.045),   # 블랙 들뜸 (푸른 기)
    gain=(1.00, 0.985, 0.95),     # 하이라이트 약간 따뜻하게
    saturation=0.92,
    contrast=1.06,
    aberration=1.4,       # px (가장자리 기준)
    soften=0.65,
    weave=1.6,            # px
    flicker=0.022,
    grain=0.050,
    dust=1.0,
    vignette=0.30,
)


def _tonemap(img):
    """1.0 초과 HDR 을 부드럽게 어깨 압축 (필름 숄더)."""
    knee = 0.80
    over = np.maximum(img - knee, 0)
    return np.minimum(img, knee) + (1 - knee) * (1 - np.exp(-over / (1 - knee)))


def _vignette_map():
    if not hasattr(_vignette_map, "m"):
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot((xs - W / 2) / (W / 2), (ys - H / 2) / (H / 2)) / math.sqrt(2)
        _vignette_map.m = (1 - np.clip(d - 0.35, 0, 1) ** 1.6 * 1.25).astype(np.float32)
    return _vignette_map.m


def _aberration(img, px):
    if px <= 0:
        return img
    out = img.copy()
    for ch, s in ((0, 1 + px / W), (2, 1 - px / W)):
        M = np.array([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]], np.float32)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
    return out


def _grain(frame, amount):
    rng = np.random.default_rng([7, frame])
    gh, gw = int(H / 1.6), int(W / 1.6)
    g = rng.standard_normal((gh, gw, 1)).astype(np.float32)
    c = rng.standard_normal((gh, gw, 3)).astype(np.float32) * 0.35
    g = cv2.resize(g + c, (W, H), interpolation=cv2.INTER_LINEAR)
    return g * amount


def _dust(img, frame, amount):
    """먼지 반점, 머리카락, 세로 스크래치."""
    if amount <= 0:
        return img
    rng = np.random.default_rng([31, frame])
    m_dark = np.zeros((H, W), np.uint8)
    m_lite = np.zeros((H, W), np.uint8)
    n = rng.poisson(3.0 * amount)
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(1.0, 4.5)
        tgt = m_lite if rng.random() < 0.6 else m_dark
        pts = []
        k = rng.integers(5, 9)
        for i in range(k):
            a = i / k * 2 * np.pi
            rr = r * rng.uniform(0.5, 1.3)
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
        cv2.fillPoly(tgt, [np.round(np.array(pts) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    if rng.random() < 0.10 * amount:      # 머리카락
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a = rng.uniform(0, 2 * np.pi)
        pts = []
        for i in range(30):
            a += rng.normal(0, 0.18)
            x += math.cos(a) * 5
            y += math.sin(a) * 5
            pts.append((x, y))
        cv2.polylines(m_dark, [np.round(np.array(pts) * 16).astype(np.int32)], False, 255, 2,
                      cv2.LINE_AA, 4)
    # 스크래치: 몇 프레임 동안 지속되는 세로선
    block = frame // 9
    if hash01(77, block) < 0.35 * amount:
        x0 = hash01(78, block) * W
        x = x0 + (hash01(79, frame) - 0.5) * 6
        y0 = hash01(80, frame) * H * 0.3
        y1 = H - hash01(81, frame) * H * 0.3
        cv2.line(m_lite, (int(x * 16), int(y0 * 16)), (int((x + 2) * 16), int(y1 * 16)), 120, 1,
                 cv2.LINE_AA, 4)
    lite = cv2.GaussianBlur(m_lite, (0, 0), 0.7).astype(np.float32)[..., None] / 255.0
    dark = cv2.GaussianBlur(m_dark, (0, 0), 0.7).astype(np.float32)[..., None] / 255.0
    img = img * (1 - dark * 0.85)
    img = img + lite * 0.55 * (1.05 - img)
    return img


def film(img: np.ndarray, frame: int, p: dict | None = None) -> np.ndarray:
    """장면 출력(float32 HDR) → 필름 룩 적용된 float32 0..1."""
    p = {**FILM, **(p or {})}
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=4.0, neginf=0.0)
    img = np.clip(img, 0, 8)

    # 할레이션: 밝은 부분 → 크게 번진 붉은 주황 테
    if p["halation"] > 0:
        lum = img.max(axis=2)
        hot = np.clip(lum - p["halation_thresh"], 0, None)
        hal = blur(hot, 10) * 0.6 + blur(hot, 28) * 0.4
        img = img + hal[..., None] * np.array([1.0, 0.32, 0.12], np.float32) * p["halation"]
    if p["bloom"] > 0:
        img = img + blur(np.clip(img - 0.5, 0, None), 36) * p["bloom"]

    img = _tonemap(img)

    # 색 그레이딩 (바랜 프린트)
    g = img.mean(axis=2, keepdims=True)
    img = g + (img - g) * p["saturation"]
    img = (img - 0.5) * p["contrast"] + 0.5
    img = img * np.array(p["gain"], np.float32)
    lift = np.array(p["lift"], np.float32)
    img = lift + img * (1 - lift)

    img = _aberration(np.clip(img, 0, 1.2), p["aberration"])
    if p["soften"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), p["soften"])

    # 게이트 위브: 프레임 전체가 미세하게 흔들림
    if p["weave"] > 0:
        t = frame / 24.0
        dx = p["weave"] * (0.6 * wobble(t, 11, (0.7, 1.9, 4.3)) + 0.4 * (hash01(12, frame) - 0.5))
        dy = p["weave"] * 0.8 * (0.6 * wobble(t, 13, (0.5, 1.3, 3.7)) + 0.4 * (hash01(14, frame) - 0.5))
        rot = 0.02 * wobble(t, 15)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.003)
        M[0, 2] += dx
        M[1, 2] += dy
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 플리커
    img = img * (1 + p["flicker"] * (hash01(21, frame) - 0.5) * 2)

    img = img * (_vignette_map()[..., None] * p["vignette"] + (1 - p["vignette"]))

    # 그레인 (중간톤에서 가장 강하게)
    if p["grain"] > 0:
        L = np.clip(img.mean(axis=2, keepdims=True), 0, 1)
        resp = 0.45 + 2.2 * L * (1 - L)
        img = img + _grain(frame, p["grain"]) * resp

    img = _dust(img, frame, p["dust"])
    return np.clip(img, 0, 1)
