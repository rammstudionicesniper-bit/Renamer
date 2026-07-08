@echo off
setlocal
REM ============================================================
REM  연번 리네이머 (SeqRenamer) - 윈도우 단일 exe 빌드 스크립트
REM  집 PC(윈도우)에서 더블클릭하거나 명령창에서 실행하세요.
REM ============================================================
REM  창이 바로 닫히지 않도록 끝에서 pause 로 멈춥니다.

cd /d "%~dp0"

echo ============================================================
echo   연번 리네이머 빌드 시작
echo ============================================================
echo.

REM ---- 파이썬 실행기 찾기 (python 또는 py) --------------------
set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY (
    where py >nul 2>&1 && set "PY=py"
)
if not defined PY (
    echo *** Python 을 찾지 못했습니다. ***
    echo.
    echo   1) https://www.python.org/downloads/ 에서 Python 3.11+ 설치
    echo   2) 설치 화면에서 반드시 "Add Python to PATH" 체크
    echo   3) 설치 후 이 build.bat 을 다시 실행
    echo.
    goto :end
)

echo [사용할 Python] %PY%
%PY% --version
echo.

echo [1/3] 의존성 설치 (PySide6, PyInstaller)...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo.
echo [2/3] 이전 빌드 정리...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo [3/3] PyInstaller 단일 exe 빌드...
%PY% -m PyInstaller --onefile --windowed --name SeqRenamer main.py
if errorlevel 1 goto :error

echo.
echo ============================================================
echo   완료!  dist\SeqRenamer.exe 를 확인하세요.
echo ============================================================
goto :end

:error
echo.
echo *** 빌드 중 오류가 발생했습니다. 위 메시지를 확인하세요. ***

:end
echo.
echo (이 창을 닫으려면 아무 키나 누르세요)
pause >nul
endlocal
