"""연번 리네이머 핵심 로직 (GUI 비의존).

이 모듈은 Qt 등 GUI 라이브러리에 의존하지 않는 순수 파이썬 로직만 담는다.
GUI(`gui.py`)와 테스트(`tests/`)가 이 모듈을 공유한다.

핵심 규칙 (명세서 §5.2, §5.3):
  1. 파일명(확장자 제외) 끝의 윈도우식 `(n)` / `복사본` / `Copy` 표기를 먼저 제거한다.
  2. 정리된 이름을 그룹 기준으로 삼아 `_01`, `_02` … 연번을 붙인다.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# 전처리 정규식 (명세서 §5.2)  — 반드시 이름 '끝($)'에서만 매칭한다.
# ---------------------------------------------------------------------------

# ' (4)', '(5)', ' (123)'  — 공백 유무 모두. 윈도우 복사/다운로드 중복 표기.
WIN_PAREN_SUFFIX = re.compile(r"\s*\(\d+\)$")

# ' - 복사본', ' - 복사본 (2)', ' - Copy', ' - Copy (2)'
COPY_SUFFIX = re.compile(r"\s*-\s*(?:복사본|Copy)(?:\s*\(\d+\))?$", re.IGNORECASE)

# 이미 이 툴이 붙였던 '_01' 형태의 연번 접미사. 재처리 시 '_01_01' 방지용.
SEQ_SUFFIX = re.compile(r"_\d+$")


def clean_stem(stem: str, *, strip_existing_seq: bool = True) -> str:
    """파일명(확장자 제외)에서 윈도우식 중복 표기를 제거한다.

    - ` (n)` / `(n)` / ` - 복사본 (n)` / ` - Copy (n)` 를 끝에서 반복 제거.
    - `strip_existing_seq=True` 면 이 툴이 이전에 붙였던 `_NN` 연번도 1회 제거해
      재처리 시 `_01_01` 이 되는 것을 막는다.
    - 이름 '중간'의 괄호는 끝 앵커(`$`) 덕분에 보존된다.
    """
    prev = None
    while prev != stem:
        prev = stem
        stem = WIN_PAREN_SUFFIX.sub("", stem)
        stem = COPY_SUFFIX.sub("", stem)
        if strip_existing_seq:
            # 루프 안에서 처리해야 'Hum (4)_01' 처럼 연번 뒤에 숨은 (4)도 벗겨진다
            # (v1 버그 산출물 'Hum(4)_01' 재처리 대비).
            stem = SEQ_SUFFIX.sub("", stem)
    return stem.strip()


# ---------------------------------------------------------------------------
# 자연 정렬 (명세서 §5.4)  — '(2)'가 '(10)'보다 앞에 오도록.
# ---------------------------------------------------------------------------

_NAT_SPLIT = re.compile(r"(\d+)")


def natural_key(text: str):
    """자연 정렬용 키. 숫자 구간은 정수로 비교해 문자열 정렬 문제를 피한다."""
    parts = _NAT_SPLIT.split(text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class SortOrder(str, Enum):
    """미리보기/번호 부여 시 파일 정렬 순서 (명세서 §5.4)."""

    NAME = "name"      # 파일명 자연 정렬 (기본)
    MTIME = "mtime"    # 수정일시순
    DROPPED = "dropped"  # 드롭(추가)한 순서


class RenameStatus(str, Enum):
    OK = "ok"              # 정상 변경 대상
    NO_CHANGE = "nochange"  # 이미 원하는 이름 → 변경 불필요
    CONFLICT = "conflict"  # 대상 이름 충돌 → 건너뜀
    ERROR = "error"        # 실행 실패


@dataclass
class RenamePlanItem:
    """리네임 계획 한 줄 (미리보기 표의 한 행)."""

    src: Path
    new_name: str
    status: RenameStatus = RenameStatus.OK
    message: str = ""
    # 실행 결과 (execute 이후 채워짐)
    done: bool = False

    @property
    def original_name(self) -> str:
        return self.src.name

    @property
    def dst(self) -> Path:
        return self.src.with_name(self.new_name)

    @property
    def is_renameable(self) -> bool:
        return self.status == RenameStatus.OK


@dataclass
class RenameOptions:
    """리네임 옵션 (UI 컨트롤과 1:1 대응)."""

    clean_windows_suffix: bool = True   # (숫자)·복사본 제거 (기본 ON)
    digits: int = 2                     # 연번 자릿수 (2~4)
    start: int = 1                      # 시작 번호
    sort_order: SortOrder = SortOrder.NAME
    per_folder_group: bool = False      # False=파일명 기준 통합 연번(기본), True=폴더별 독립


# ---------------------------------------------------------------------------
# 파일 수집 (명세서 §5.1)
# ---------------------------------------------------------------------------

def parse_ext_filter(raw: str) -> set[str] | None:
    """'wav, mp3' 같은 입력을 {'.wav', '.mp3'} 로 정규화. 비면 None(전체)."""
    if not raw or not raw.strip():
        return None
    exts: set[str] = set()
    for tok in re.split(r"[,\s;]+", raw.strip()):
        if not tok:
            continue
        tok = tok.lower()
        if not tok.startswith("."):
            tok = "." + tok
        exts.add(tok)
    return exts or None


def collect_files(
    paths: Iterable[str | os.PathLike],
    *,
    include_subfolders: bool = True,
    ext_filter: set[str] | None = None,
) -> list[Path]:
    """드롭된 파일/폴더 경로들에서 실제 대상 파일 목록을 만든다.

    - 파일은 그대로 포함, 폴더는 내부를 탐색.
    - `include_subfolders` 로 재귀 여부 결정 (기본 ON).
    - `ext_filter` (소문자, 점 포함) 로 확장자 필터. 대소문자 무시.
    - 드롭 순서를 최대한 유지하고 중복은 제거한다.
    """
    result: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp in seen:
            return
        if ext_filter is not None and p.suffix.lower() not in ext_filter:
            return
        seen.add(rp)
        result.append(p)

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if include_subfolders else p.glob("*")
            # 폴더 내부는 자연 정렬로 안정적 순서 보장
            for child in sorted((c for c in it if c.is_file()), key=lambda c: natural_key(str(c))):
                _add(child)
        elif p.is_file():
            _add(p)
        # 존재하지 않는 경로는 조용히 무시
    return result


# ---------------------------------------------------------------------------
# 리네임 계획 생성 (명세서 §5.3 ~ §5.6)
# ---------------------------------------------------------------------------

def _sort_group(items: list[Path], order: SortOrder, drop_index: dict[Path, int]) -> list[Path]:
    if order == SortOrder.NAME:
        # 확장자를 뺀 stem 으로 정렬해야 'Hum' 이 'Hum (4)' 보다 앞에 온다.
        # (전체 파일명으로 정렬하면 공백(0x20)<점(0x2e) 때문에 '(4)'가 앞서는 문제 발생)
        return sorted(items, key=lambda p: (natural_key(p.stem), p.suffix.lower()))
    if order == SortOrder.MTIME:
        def mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0
        return sorted(items, key=lambda p: (mtime(p), natural_key(p.name)))
    if order == SortOrder.DROPPED:
        return sorted(items, key=lambda p: drop_index.get(p, 0))
    return items


def build_plan(files: Sequence[Path], options: RenameOptions) -> list[RenamePlanItem]:
    """파일 목록과 옵션으로 리네임 계획을 만든다.

    반환 순서는 각 그룹 내 정렬 순서를 따르며, 그룹은 정리된 이름의 자연 정렬 순.
    충돌(대상 이름이 그룹 밖의 기존 파일과 겹침, 혹은 계획 내 중복)은
    `RenameStatus.CONFLICT` 로 표시한다 (덮어쓰기 금지, 명세서 §5.6).
    """
    drop_index = {p: i for i, p in enumerate(files)}

    # 1) 그룹핑 — 키: (정리된 stem[, 부모폴더])
    groups: dict[tuple, list[Path]] = {}
    for p in files:
        stem = p.stem
        if options.clean_windows_suffix:
            stem = clean_stem(stem)
        else:
            # 제거를 끄더라도 재처리 방지용 기존 연번은 그대로 두고 원본 stem 사용
            stem = p.stem
        key: tuple
        if options.per_folder_group:
            key = (str(p.parent), stem)
        else:
            key = (stem,)
        groups.setdefault(key, []).append(p)

    # 2) 그룹 순서: 정리된 이름 자연 정렬 (미리보기 안정성)
    def group_sort_key(kv):
        key = kv[0]
        stem = key[-1]
        return (natural_key(str(key[0])) if len(key) > 1 else [], natural_key(stem))

    plan: list[RenamePlanItem] = []
    # 계획된 대상 경로 → 이를 만든 항목들 (계획 내 충돌 감지)
    planned_targets: dict[Path, list[RenamePlanItem]] = {}

    for key, members in sorted(groups.items(), key=group_sort_key):
        stem = key[-1]
        ordered = _sort_group(members, options.sort_order, drop_index)
        seq = options.start
        for p in ordered:
            num = str(seq).zfill(options.digits)
            new_name = f"{stem}_{num}{p.suffix}"
            item = RenamePlanItem(src=p, new_name=new_name)
            if new_name == p.name:
                item.status = RenameStatus.NO_CHANGE
            plan.append(item)
            planned_targets.setdefault(p.with_name(new_name).resolve(), []).append(item)
            seq += 1

    # 3) 충돌 검사
    src_set = {p.resolve() for p in files}
    for target, items in planned_targets.items():
        # (a) 계획 내에서 여러 항목이 같은 대상으로 가면 충돌
        if len(items) > 1:
            for it in items:
                if it.status != RenameStatus.NO_CHANGE:
                    it.status = RenameStatus.CONFLICT
                    it.message = "대상 이름 중복 (계획 내 충돌)"
            continue
        it = items[0]
        if it.status == RenameStatus.NO_CHANGE:
            continue
        # (b) 대상 이름이 이미 존재하고, 그 파일이 이번 작업 대상이 아니면 충돌
        if target.exists() and target not in src_set:
            it.status = RenameStatus.CONFLICT
            it.message = "대상 이름이 이미 존재함 (덮어쓰기 금지)"

    return plan


# ---------------------------------------------------------------------------
# 실행 (명세서 §5.6)  — 그룹 내 이름 교환을 위해 2단계(임시명) 리네임.
# ---------------------------------------------------------------------------

@dataclass
class RenameResult:
    success: int = 0
    skipped: int = 0
    failed: int = 0
    # undo(되돌리기) 용: (현재경로, 원래경로) 쌍
    undo_pairs: list[tuple[Path, Path]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def execute_plan(plan: Sequence[RenamePlanItem]) -> RenameResult:
    """계획을 실제로 실행한다.

    A→B, B→C 처럼 그룹 내에서 이름이 순환/교환될 수 있으므로,
    먼저 모든 대상을 고유 임시 이름으로 바꾼 뒤 최종 이름으로 바꾸는 2단계 방식.
    이렇게 하면 중간 충돌 없이 안전하게 처리된다.
    """
    result = RenameResult()
    to_rename = [it for it in plan if it.status == RenameStatus.OK]

    # 건너뜀 집계 (충돌 등)
    result.skipped = sum(
        1 for it in plan if it.status in (RenameStatus.CONFLICT, RenameStatus.ERROR)
    )

    # 1단계: 원본 → 임시명
    staged: list[tuple[Path, Path]] = []  # (임시경로, 최종경로, 원본경로)
    staged_full: list[tuple[Path, Path, Path]] = []
    for it in to_rename:
        src = it.src
        tmp = src.with_name(f".seqrenamer_tmp_{uuid.uuid4().hex}{src.suffix}")
        try:
            src.rename(tmp)
            staged_full.append((tmp, it.dst, src))
        except OSError as e:
            it.status = RenameStatus.ERROR
            it.message = str(e)
            result.failed += 1
            result.errors.append(f"{src.name}: {e}")

    # 2단계: 임시명 → 최종명
    for tmp, dst, src in staged_full:
        it = next((x for x in to_rename if x.src == src), None)
        try:
            tmp.rename(dst)
            if it is not None:
                it.done = True
            result.success += 1
            result.undo_pairs.append((dst, src))
        except OSError as e:
            # 실패 시 임시명을 원래대로 되돌려 최대한 원상복구
            try:
                tmp.rename(src)
            except OSError:
                pass
            if it is not None:
                it.status = RenameStatus.ERROR
                it.message = str(e)
            result.failed += 1
            result.errors.append(f"{src.name}: {e}")

    return result


def undo(undo_pairs: Sequence[tuple[Path, Path]]) -> RenameResult:
    """직전 실행을 1회 되돌린다 (명세서 §5.6 / §8, 2차 기능)."""
    result = RenameResult()
    # 역시 2단계로 안전하게
    staged: list[tuple[Path, Path]] = []
    for current, original in undo_pairs:
        tmp = current.with_name(f".seqrenamer_undo_{uuid.uuid4().hex}{current.suffix}")
        try:
            current.rename(tmp)
            staged.append((tmp, original))
        except OSError as e:
            result.failed += 1
            result.errors.append(f"{current.name}: {e}")
    for tmp, original in staged:
        try:
            tmp.rename(original)
            result.success += 1
            result.undo_pairs.append((original, tmp))
        except OSError as e:
            result.failed += 1
            result.errors.append(f"{original.name}: {e}")
    return result
