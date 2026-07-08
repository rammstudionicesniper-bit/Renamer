"""핵심 로직 테스트 — 명세서 §2, §5, §9 엣지 케이스 검증."""

from pathlib import Path

import pytest

from seqrenamer.core import (
    RenameOptions,
    RenameStatus,
    SortOrder,
    build_plan,
    clean_stem,
    collect_files,
    execute_plan,
    natural_key,
    parse_ext_filter,
    undo,
)


# --------------------------------------------------------------------------- #
# clean_stem — 윈도우식 표기 제거 (§5.2)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "given,expected",
    [
        ("Amb_NPC_Child_Elf_Female_A_Hum", "Amb_NPC_Child_Elf_Female_A_Hum"),
        ("Amb_NPC_Child_Elf_Female_A_Hum (4)", "Amb_NPC_Child_Elf_Female_A_Hum"),
        ("Amb_NPC_Child_Elf_Female_A_Hum (5)", "Amb_NPC_Child_Elf_Female_A_Hum"),
        ("Amb_NPC_Child_Elf_Female_A_Hum (7)", "Amb_NPC_Child_Elf_Female_A_Hum"),
        # 공백 없는 괄호
        ("Hum(4)", "Hum"),
        # 모든 숫자 (4만 아님)
        ("File (10)", "File"),
        ("File (123)", "File"),
        # 여러 개 붙은 경우 전부 제거
        ("File (2) (3)", "File"),
        ("File (2) (3) (4)", "File"),
        # 복사본 / Copy
        ("File - 복사본", "File"),
        ("File - 복사본 (2)", "File"),
        ("File - Copy", "File"),
        ("File - Copy (2)", "File"),
        ("File - copy (2)", "File"),  # 대소문자 무시
        # 중간 괄호는 보존 (끝 앵커 $)
        ("Sound (final) mix", "Sound (final) mix"),
        ("Track (A) part (2)", "Track (A) part"),
        # 재처리 방지: _01 제거
        ("Hum_01", "Hum"),
        ("Hum (4)_01", "Hum"),
    ],
)
def test_clean_stem(given, expected):
    assert clean_stem(given) == expected


def test_clean_stem_keep_existing_seq():
    # strip_existing_seq=False 면 _01 을 남기므로 그 뒤에 숨은 (4)는 건드릴 수 없다.
    assert clean_stem("Hum_01", strip_existing_seq=False) == "Hum_01"
    assert clean_stem("Hum (4)_01", strip_existing_seq=False) == "Hum (4)_01"
    # 반대로 (4)가 끝에 있으면 seq 제거 없이도 벗겨진다.
    assert clean_stem("Hum_01 (4)", strip_existing_seq=False) == "Hum_01"


# --------------------------------------------------------------------------- #
# natural_key — 자연 정렬 (§5.4)
# --------------------------------------------------------------------------- #

def test_natural_sort_order():
    names = ["File (10)", "File (2)", "File (1)", "File"]
    ordered = sorted(names, key=natural_key)
    assert ordered == ["File", "File (1)", "File (2)", "File (10)"]


# --------------------------------------------------------------------------- #
# parse_ext_filter (§5.7)
# --------------------------------------------------------------------------- #

def test_parse_ext_filter():
    assert parse_ext_filter("") is None
    assert parse_ext_filter("   ") is None
    assert parse_ext_filter("wav") == {".wav"}
    assert parse_ext_filter("wav, mp3") == {".wav", ".mp3"}
    assert parse_ext_filter(".WAV, MP3") == {".wav", ".mp3"}
    assert parse_ext_filter("wav;mp3 flac") == {".wav", ".mp3", ".flac"}


# --------------------------------------------------------------------------- #
# build_plan — 핵심 시나리오 (§2 실제 사고 케이스)
# --------------------------------------------------------------------------- #

def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.write_bytes(b"x")
    return p


def test_real_accident_case(tmp_path):
    base = "Amb_NPC_Child_Elf_Female_A_Hum"
    for n in [f"{base}.wav", f"{base} (4).wav", f"{base} (5).wav", f"{base} (7).wav"]:
        _touch(tmp_path, n)
    files = collect_files([tmp_path])
    plan = build_plan(files, RenameOptions())
    result = {it.original_name: it.new_name for it in plan}
    assert result[f"{base}.wav"] == f"{base}_01.wav"
    assert result[f"{base} (4).wav"] == f"{base}_02.wav"
    assert result[f"{base} (5).wav"] == f"{base}_03.wav"
    assert result[f"{base} (7).wav"] == f"{base}_04.wav"
    # ❌ (4) 가 이름에 남으면 안 된다
    for it in plan:
        assert "(4)" not in it.new_name
        assert "(" not in it.new_name


def test_single_file_gets_01(tmp_path):
    _touch(tmp_path, "Hum.wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    assert plan[0].new_name == "Hum_01.wav"


def test_independent_groups(tmp_path):
    _touch(tmp_path, "A.wav")
    _touch(tmp_path, "A (2).wav")
    _touch(tmp_path, "B.wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    res = {it.original_name: it.new_name for it in plan}
    assert res["A.wav"] == "A_01.wav"
    assert res["A (2).wav"] == "A_02.wav"
    assert res["B.wav"] == "B_01.wav"  # 다른 이름 → 각자 _01 부터


def test_digits_option(tmp_path):
    _touch(tmp_path, "X.wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions(digits=4))
    assert plan[0].new_name == "X_0001.wav"


def test_start_option(tmp_path):
    _touch(tmp_path, "X.wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions(start=5))
    assert plan[0].new_name == "X_05.wav"


def test_extension_case_preserved(tmp_path):
    _touch(tmp_path, "Loud.WAV")
    _touch(tmp_path, "Loud (2).WAV")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    res = {it.original_name: it.new_name for it in plan}
    assert res["Loud.WAV"] == "Loud_01.WAV"
    assert res["Loud (2).WAV"] == "Loud_02.WAV"


def test_middle_paren_preserved(tmp_path):
    _touch(tmp_path, "Sound (final) mix.wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    assert plan[0].new_name == "Sound (final) mix_01.wav"


def test_korean_and_spaces(tmp_path):
    _touch(tmp_path, "발소리 크게.wav")
    _touch(tmp_path, "발소리 크게 (2).wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    res = {it.original_name: it.new_name for it in plan}
    assert res["발소리 크게.wav"] == "발소리 크게_01.wav"
    assert res["발소리 크게 (2).wav"] == "발소리 크게_02.wav"


def test_reprocess_no_double_seq(tmp_path):
    # 이미 _01 붙은 파일 재처리 시 _01_01 안 되게
    _touch(tmp_path, "Hum_01.wav")
    _touch(tmp_path, "Hum_02.wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    res = {it.original_name: it.new_name for it in plan}
    # 이미 원하는 이름이므로 NO_CHANGE
    assert res["Hum_01.wav"] == "Hum_01.wav"
    assert res["Hum_02.wav"] == "Hum_02.wav"
    for it in plan:
        assert it.status == RenameStatus.NO_CHANGE


def test_conflict_with_existing_file(tmp_path):
    # 대상 이름이 이미 존재하는(작업 대상 아님) 파일과 충돌
    _touch(tmp_path, "S.wav")          # → S_01.wav 로 가려는데
    _touch(tmp_path, "S_01.wav.keep")  # 무관
    existing = _touch(tmp_path, "S_01.wav")  # 이미 존재, 작업 대상 아니게 하려고...
    # existing 을 대상에서 빼려면 확장자 필터로 S.wav 만 넣는다
    files = [tmp_path / "S.wav"]
    plan = build_plan(files, RenameOptions())
    assert plan[0].status == RenameStatus.CONFLICT


# --------------------------------------------------------------------------- #
# subfolder / grouping (§9)
# --------------------------------------------------------------------------- #

def test_subfolder_include(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(tmp_path, "T.wav")
    _touch(sub, "T (2).wav")
    files = collect_files([tmp_path], include_subfolders=True)
    assert len(files) == 2


def test_subfolder_exclude(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(tmp_path, "T.wav")
    _touch(sub, "T (2).wav")
    files = collect_files([tmp_path], include_subfolders=False)
    assert len(files) == 1


def test_unified_numbering_across_folders(tmp_path):
    # 기본: 파일명 기준 통합 연번
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _touch(a, "Step.wav")
    _touch(b, "Step.wav")
    files = collect_files([tmp_path])
    plan = build_plan(files, RenameOptions())
    new_names = sorted(it.new_name for it in plan)
    assert new_names == ["Step_01.wav", "Step_02.wav"]


def test_per_folder_numbering(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _touch(a, "Step.wav")
    _touch(b, "Step.wav")
    files = collect_files([tmp_path])
    plan = build_plan(files, RenameOptions(per_folder_group=True))
    new_names = sorted(it.new_name for it in plan)
    assert new_names == ["Step_01.wav", "Step_01.wav"]  # 각 폴더 독립


def test_ext_filter(tmp_path):
    _touch(tmp_path, "a.wav")
    _touch(tmp_path, "b.mp3")
    _touch(tmp_path, "c.txt")
    files = collect_files([tmp_path], ext_filter=parse_ext_filter("wav, mp3"))
    names = sorted(f.name for f in files)
    assert names == ["a.wav", "b.mp3"]


# --------------------------------------------------------------------------- #
# execute + undo (§5.6, §8)
# --------------------------------------------------------------------------- #

def test_execute_renames(tmp_path):
    base = "Hum"
    for n in [f"{base}.wav", f"{base} (4).wav", f"{base} (5).wav"]:
        _touch(tmp_path, n)
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    result = execute_plan(plan)
    assert result.success == 3
    assert result.failed == 0
    got = sorted(p.name for p in tmp_path.iterdir())
    assert got == ["Hum_01.wav", "Hum_02.wav", "Hum_03.wav"]


def test_execute_name_swap(tmp_path):
    # S_02→S_01, S_01→S_02 식 교환도 임시명 2단계로 안전하게.
    # 드롭 순서를 강제해 실제로 이름이 맞교환되도록 만든다.
    p2 = _touch(tmp_path, "S_02.wav")  # 첫 번째로 드롭 → S_01
    p1 = _touch(tmp_path, "S_01.wav")  # 두 번째로 드롭 → S_02
    plan = build_plan([p2, p1], RenameOptions(sort_order=SortOrder.DROPPED))
    res = {it.original_name: it.new_name for it in plan}
    assert res["S_02.wav"] == "S_01.wav"
    assert res["S_01.wav"] == "S_02.wav"
    result = execute_plan(plan)
    assert result.failed == 0
    assert result.success == 2
    got = sorted(p.name for p in tmp_path.iterdir())
    assert got == ["S_01.wav", "S_02.wav"]


def test_undo_restores(tmp_path):
    _touch(tmp_path, "Hum.wav")
    _touch(tmp_path, "Hum (2).wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    result = execute_plan(plan)
    assert result.success == 2
    undo_result = undo(result.undo_pairs)
    assert undo_result.success == 2
    got = sorted(p.name for p in tmp_path.iterdir())
    assert got == ["Hum (2).wav", "Hum.wav"]


def test_no_change_not_renamed(tmp_path):
    _touch(tmp_path, "Ready_01.wav")
    plan = build_plan(collect_files([tmp_path]), RenameOptions())
    assert plan[0].status == RenameStatus.NO_CHANGE
    result = execute_plan(plan)
    assert result.success == 0  # 변경 대상 없음
