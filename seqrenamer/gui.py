"""연번 리네이머 GUI (PySide6).

명세서 §6 UI 레이아웃을 구현한다. 모든 리네임 로직은 `core.py` 에 있고
이 파일은 화면/입력/이벤트만 담당한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core import (
    RenameOptions,
    RenameStatus,
    SortOrder,
    build_plan,
    collect_files,
    execute_plan,
    parse_ext_filter,
    undo,
)

# 정렬 콤보 인덱스 → SortOrder
_SORT_CHOICES = [
    ("파일명순", SortOrder.NAME),
    ("수정일시순", SortOrder.MTIME),
    ("드롭한 순서", SortOrder.DROPPED),
]

_CONFLICT_BG = QColor(255, 205, 205)   # 빨강 (충돌/오류)
_NOCHANGE_FG = QColor(140, 140, 140)   # 회색 (변경 없음)


class DropTable(QTableWidget):
    """파일/폴더 드래그앤드롭을 받는 미리보기 표."""

    def __init__(self, on_drop, parent=None):
        super().__init__(0, 2, parent)
        self._on_drop = on_drop
        self.setHorizontalHeaderLabels(["원본", "변경 후"])
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionMode(QTableWidget.NoSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTableWidget.DropOnly)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)

    # --- 드래그앤드롭 이벤트 ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
        if paths:
            self._on_drop(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("연번 리네이머 (SeqRenamer)")
        self.resize(760, 560)
        self.setAcceptDrops(True)

        # 상태
        self._dropped_paths: list[str] = []   # 사용자가 드롭/추가한 원본 경로 (파일+폴더)
        self._plan = []
        self._last_undo_pairs: list = []

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 안내
        hint = QLabel("여기(표)로 파일/폴더를 끌어다 놓으세요.  또는 아래 버튼으로 추가.")
        hint.setAlignment(Qt.AlignCenter)
        root.addWidget(hint)

        # 옵션 줄 1: 체크박스
        opt1 = QHBoxLayout()
        self.cb_subfolders = QCheckBox("하위 폴더 포함")
        self.cb_subfolders.setChecked(True)
        self.cb_clean = QCheckBox("(숫자)·복사본 제거")
        self.cb_clean.setChecked(True)
        self.cb_clean.setToolTip(
            "윈도우가 붙인 ' (4)', ' - 복사본', ' - Copy' 및 기존 '_01' 연번을 제거합니다.\n"
            "이 툴의 핵심 기능이라 보통 켜 둡니다."
        )
        opt1.addWidget(self.cb_subfolders)
        opt1.addWidget(self.cb_clean)
        opt1.addStretch(1)
        root.addLayout(opt1)

        # 옵션 줄 2: 자릿수/시작/정렬
        opt2 = QHBoxLayout()
        opt2.addWidget(QLabel("자릿수"))
        self.sp_digits = QSpinBox()
        self.sp_digits.setRange(2, 4)
        self.sp_digits.setValue(2)
        opt2.addWidget(self.sp_digits)

        opt2.addSpacing(12)
        opt2.addWidget(QLabel("시작"))
        self.sp_start = QSpinBox()
        self.sp_start.setRange(0, 999999)
        self.sp_start.setValue(1)
        opt2.addWidget(self.sp_start)

        opt2.addSpacing(12)
        opt2.addWidget(QLabel("정렬"))
        self.cmb_sort = QComboBox()
        for label, _ in _SORT_CHOICES:
            self.cmb_sort.addItem(label)
        opt2.addWidget(self.cmb_sort)

        opt2.addSpacing(12)
        self.cb_per_folder = QCheckBox("폴더별 독립 연번")
        self.cb_per_folder.setToolTip(
            "끄면(기본) 같은 파일명은 폴더가 달라도 통합 연번.\n켜면 폴더마다 _01 부터 다시 시작."
        )
        opt2.addWidget(self.cb_per_folder)
        opt2.addStretch(1)
        root.addLayout(opt2)

        # 옵션 줄 3: 확장자 필터
        opt3 = QHBoxLayout()
        opt3.addWidget(QLabel("확장자 필터"))
        self.le_ext = QLineEdit()
        self.le_ext.setPlaceholderText("예: wav, mp3  (비우면 전체)")
        opt3.addWidget(self.le_ext)
        root.addLayout(opt3)

        # 미리보기 표
        self.table = DropTable(self._on_drop)
        root.addWidget(self.table, 1)

        # 요약
        self.lbl_summary = QLabel("성공 0 / 건너뜀 0 / 실패 0")
        root.addWidget(self.lbl_summary)

        # 버튼 줄
        btns = QHBoxLayout()
        self.btn_add_files = QPushButton("파일 추가")
        self.btn_add_folder = QPushButton("폴더 열기")
        self.btn_run = QPushButton("이름 변경 실행")
        self.btn_undo = QPushButton("되돌리기")
        self.btn_reset = QPushButton("초기화")
        self.btn_undo.setEnabled(False)
        for b in (self.btn_add_files, self.btn_add_folder):
            btns.addWidget(b)
        btns.addStretch(1)
        for b in (self.btn_run, self.btn_undo, self.btn_reset):
            btns.addWidget(b)
        root.addLayout(btns)

        # 시그널
        self.btn_add_files.clicked.connect(self._add_files_dialog)
        self.btn_add_folder.clicked.connect(self._add_folder_dialog)
        self.btn_run.clicked.connect(self._run_rename)
        self.btn_undo.clicked.connect(self._run_undo)
        self.btn_reset.clicked.connect(self._reset)
        for w in (self.cb_subfolders, self.cb_clean, self.cb_per_folder):
            w.stateChanged.connect(self._refresh_preview)
        self.sp_digits.valueChanged.connect(self._refresh_preview)
        self.sp_start.valueChanged.connect(self._refresh_preview)
        self.cmb_sort.currentIndexChanged.connect(self._refresh_preview)
        self.le_ext.textChanged.connect(self._refresh_preview)

    # 창 전체에도 드롭 허용 (표 밖에 떨어뜨려도 받도록)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self._on_drop(paths)

    # ------------------------------------------------------------ actions
    def _on_drop(self, paths: list[str]):
        for p in paths:
            if p not in self._dropped_paths:
                self._dropped_paths.append(p)
        self._refresh_preview()

    def _add_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "파일 선택")
        if files:
            self._on_drop(files)

    def _add_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            self._on_drop([folder])

    def _current_options(self) -> RenameOptions:
        return RenameOptions(
            clean_windows_suffix=self.cb_clean.isChecked(),
            digits=self.sp_digits.value(),
            start=self.sp_start.value(),
            sort_order=_SORT_CHOICES[self.cmb_sort.currentIndex()][1],
            per_folder_group=self.cb_per_folder.isChecked(),
        )

    def _refresh_preview(self):
        ext_filter = parse_ext_filter(self.le_ext.text())
        files = collect_files(
            self._dropped_paths,
            include_subfolders=self.cb_subfolders.isChecked(),
            ext_filter=ext_filter,
        )
        self._plan = build_plan(files, self._current_options())
        self._render_table()

    def _render_table(self):
        self.table.setRowCount(len(self._plan))
        n_ok = n_skip = 0
        for row, it in enumerate(self._plan):
            src_item = QTableWidgetItem(it.original_name)
            dst_text = it.new_name
            if it.message:
                dst_text = f"{it.new_name}   ⚠ {it.message}"
            dst_item = QTableWidgetItem(dst_text)

            if it.status in (RenameStatus.CONFLICT, RenameStatus.ERROR):
                for cell in (src_item, dst_item):
                    cell.setBackground(_CONFLICT_BG)
                n_skip += 1
            elif it.status == RenameStatus.NO_CHANGE:
                for cell in (src_item, dst_item):
                    cell.setForeground(_NOCHANGE_FG)
            else:
                n_ok += 1

            self.table.setItem(row, 0, src_item)
            self.table.setItem(row, 1, dst_item)

        self.lbl_summary.setText(
            f"변경 예정 {n_ok} / 건너뜀 {n_skip} / 전체 {len(self._plan)}"
        )
        self.btn_run.setEnabled(n_ok > 0)

    def _run_rename(self):
        renameable = [it for it in self._plan if it.status == RenameStatus.OK]
        if not renameable:
            return
        reply = QMessageBox.question(
            self,
            "이름 변경 실행",
            f"{len(renameable)}개 파일의 이름을 변경합니다. 진행할까요?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        result = execute_plan(self._plan)
        self._last_undo_pairs = result.undo_pairs
        self.btn_undo.setEnabled(bool(result.undo_pairs))

        self.lbl_summary.setText(
            f"성공 {result.success} / 건너뜀 {result.skipped} / 실패 {result.failed}"
        )
        if result.errors:
            QMessageBox.warning(
                self, "일부 실패", "\n".join(result.errors[:20])
            )

        # 실행 후 실제 파일명이 바뀌었으니 미리보기 갱신
        self._refresh_preview()

    def _run_undo(self):
        if not self._last_undo_pairs:
            return
        result = undo(self._last_undo_pairs)
        self._last_undo_pairs = []
        self.btn_undo.setEnabled(False)
        self.lbl_summary.setText(
            f"되돌림: 성공 {result.success} / 실패 {result.failed}"
        )
        if result.errors:
            QMessageBox.warning(self, "되돌리기 일부 실패", "\n".join(result.errors[:20]))
        self._refresh_preview()

    def _reset(self):
        self._dropped_paths.clear()
        self._plan = []
        self._last_undo_pairs = []
        self.btn_undo.setEnabled(False)
        self.table.setRowCount(0)
        self.lbl_summary.setText("성공 0 / 건너뜀 0 / 실패 0")


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
