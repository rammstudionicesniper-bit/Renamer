"""연번 리네이머 (SeqRenamer) 실행 진입점.

  python main.py

빌드(단일 exe)는 README 의 PyInstaller 명령 참고.
"""

from seqrenamer.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
