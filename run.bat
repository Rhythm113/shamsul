@echo off
rem run.bat - Run script for shamsul (Windows Command Prompt)
echo ==> Starting shamsul...
set PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%
uv run shamsul
if %ERRORLEVEL% neq 0 (
    echo.
    echo error: Failed to start shamsul. Make sure you ran setup.ps1 first.
    pause
    exit /b %ERRORLEVEL%
)
