@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Ambiente non trovato: esegui prima avvia_tutto.bat
  pause
  exit /b 1
)

echo Servizio su http://127.0.0.1:8000
echo Documentazione interattiva: http://127.0.0.1:8000/docs
echo Per fermarlo: Ctrl+C
"%PY%" -m uvicorn app:app --host 127.0.0.1 --port 8000 --app-dir src
