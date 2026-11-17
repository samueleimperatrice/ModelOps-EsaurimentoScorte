@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [PREPARAZIONE] Creo l'ambiente virtuale .venv...
  python -m venv .venv || goto :err
  echo [PREPARAZIONE] Installo le dipendenze principali...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt || goto :err
  echo [setup] Installazione opzionale di MLflow; il progetto funziona anche senza questa dipendenza...
  "%PY%" -m pip install -r requirements-mlflow.txt
)

echo.
echo [AVVIO] Eseguo la pipeline completa del progetto...
"%PY%" src\run_all.py
if errorlevel 1 goto :run_err
echo.
echo [COMPLETATO] Pipeline terminata correttamente. Premi un tasto per chiudere.
pause
exit /b 0

:err
echo.
echo ERRORE durante la preparazione dell'ambiente: controlla Python e la connessione.
pause
exit /b 1

:run_err
echo.
echo [ERRORE] La pipeline e' terminata con errori. Controlla il riepilogo sopra.
pause
exit /b 1
