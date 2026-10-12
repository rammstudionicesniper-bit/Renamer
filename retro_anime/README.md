# 残像 ZANZŌ — 80년대 셀 애니메이션 스타일 프로시저럴 단편

70~80년대 극장판 셀 애니메이션(AKIRA, 퍼펙트 블루 계열)의 질감과 연출 문법을
**이미지 소스 없이 코드만으로** 그려 낸 약 15초 분량(1920×1080, 24fps, 360프레임) 애니메이션입니다.
배경 미술, 셀, 투과광, 필름 질감, 사운드트랙까지 모두 Python(numpy·OpenCV·Pillow·SciPy)으로
프레임마다 계산해서 만듭니다.

- 완성 영상: [`output/zanzou.mp4`](output/zanzou.mp4)
- 스타일 바이블 & 콘티: [`STYLE.md`](STYLE.md)

> 모든 캐릭터·메카·간판·대사는 오리지널 디자인입니다. 기존 작품은 연출 문법만 오마주했습니다.

## 구성 (총 15초)

| 시간 | 장면 | 내용 |
|---|---|---|
| 0.0–3.0s | `s1_city` | 네온 메가시티 설정 샷 — 멀티플레인 패럴랙스, 서치라이트, 네온 간판, 고가도로 불빛의 강 |
| 3.0–6.0s | `s2_chase` | 고속도로 바이크 추격 — 테일라이트 잔상(残像), 유선, 파워 슬라이드, 불꽃 |
| 6.0–8.5s | `s3_eye` | 눈 초접사 — 핑크/블루 분할 조명, 홍채 속 네온 반사, 동공 수축, 낯선 실루엣 |
| 8.5–11.0s | `s4_mirror` | 분장실 거울 파쇄 — 어긋난 반사 파편, TV 글리치, 「本当の私は、どこ？」 |
| 11.0–13.5s | `s5_blast` | 초능력 대폭발 — 순백 에너지 구체, 충격파 링, 반전·적색 임팩트 프레임 |
| 13.5–15.0s | `s6_title` | 타이틀 「残像」 슬램 인 → 필름 번/런아웃 |

## 셀 애니메이션 룩을 만드는 방법

- **셀(캐릭터·메카·이펙트)**: 플랫 베이스색과 경계가 딱딱한 그림자 1톤, 강약 있는 붓펜 잉크 주선.
  동화가 바뀔 때만 선이 미세하게 떨리는 *line boil* 을 넣었습니다.
- **2코마/3코마 촬영**: 캐릭터는 `ctx.on(2)` 로 12장/초, 카메라 팬은 24장/초로 섞었습니다.
- **배경 미술**: 그라디언트와 FFT 프랙탈 노이즈로 포스터컬러·에어브러시 질감을 냈습니다.
- **투과광(透過光)**: 네온·빔·폭발은 1.0 을 넘는 HDR 값으로 칠하고 다중 스케일 글로우로 번지게 했습니다.
- **35mm 필름 후처리**(`post.py`): 할레이션, 블룸, 바랜 색감, 색수차, 게이트 위브, 플리커, 그레인,
  먼지·머리카락·스크래치, 비네트를 프레임 번호 기반으로 결정적으로 넣습니다.
- **사운드트랙**(`audio.py`): 각 장면이 내보내는 `EVENTS`(충격·파쇄·폭발 등 프레임)에 맞춰
  태고, 신스 패드, 엔진, 유리 파쇄, 서브 붐, 공 같은 소리를 합성합니다.

## 실행

```bash
pip install -r retro_anime/requirements.txt     # ffmpeg 도 필요

# 전체 렌더 (멀티프로세스, 사운드 포함) → retro_anime/output/zanzou.mp4
python -m retro_anime.render

# 일부 구간만 / 사운드 없이
python -m retro_anime.render --start 72 --end 144 --out /tmp/clip.mp4 --no-audio

# 장면 미리보기 (JPG + 컨택트 시트)
python -m retro_anime.preview s2_chase --every 6 --out /tmp/preview
python -m retro_anime.preview s2_chase --strip 40:48      # 연속 프레임으로 동작 확인
python -m retro_anime.preview s2_chase --raw               # 필름 후처리 없이
```

일본어 글자는 시스템의 IPA 고딕(`fonts-ipafont-gothic`)과 WenQuanYi Zen Hei 폰트를 사용합니다.

## 파일 구조

```
retro_anime/
├── core.py      드로잉 엔진 (그라디언트·노이즈·안티앨리어스 다각형·잉크선·셀·글로우·효과선·텍스트)
├── post.py      35mm 필름 프린트 룩 후처리
├── render.py    병렬 렌더 → ffmpeg(H.264) + 사운드 합성
├── preview.py   장면 미리보기 / 컨택트 시트
├── audio.py     프로시저럴 사운드트랙
├── scenes/      장면 모듈 6개 (NAME, FRAMES, EVENTS, render(ctx))
└── output/      완성 영상
```
