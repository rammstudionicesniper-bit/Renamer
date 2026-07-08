@echo off
REM ============================================================
REM  연번 리네이머 (SeqRenamer) - 윈도우 단일 exe 빌드 스크립트
REM  집 PC(윈도우)에서 더블클릭하거나 명령창에서 실행하세요.
REM ============================================================

echo [1/3] 의존성 설치...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo [2/3] 이전 빌드 정리...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/3] PyInstaller 단일 exe 빌드...
pyinstaller --onefile --windowed --name SeqRenamer main.py
if errorlevel 1 goto :error

echo.
echo 완료!  dist\SeqRenamer.exe 를 확인하세요.
goto :eof

:error
echo.
echo *** 빌드 중 오류가 발생했습니다. 위 메시지를 확인하세요. ***
exit /b 1
