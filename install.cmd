@echo off
rem Double-click installer: runs setup.ps1 with a relaxed execution policy for this process only.
rem Extra arguments are passed through, e.g.  install.cmd -LlmUrl http://192.168.1.10:8080/v1 -LlmModel my-model
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set RESULT=%errorlevel%
echo.
if "%RESULT%"=="0" (echo Setup finished. Double-click start.cmd to launch.) else (echo Setup did not finish cleanly ^(exit code %RESULT%^). See setup.log.)
pause
exit /b %RESULT%
