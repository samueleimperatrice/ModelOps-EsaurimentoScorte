"""Esegue l'intera pipeline del Progetto 8 con un solo comando.

Flusso: train (v1, v2) -> controlli in-process -> servizio FastAPI ->
probe locale -> gate -> drill v1 -> v2 -> v1 -> verifica finale -> stop del servizio.

Uso: python src/run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

from common import load_config, read_json, resolve  # noqa: E402

PY = sys.executable


def stage(number: int, title: str) -> None:
    print(f"\n{'=' * 78}\n[{number}/8] {title}\n{'=' * 78}", flush=True)


def run(script: str, *args: str) -> int:
    print(f"\n$ python src/{script} {' '.join(args)}".rstrip(), flush=True)
    return subprocess.call([PY, "-u", str(SRC / script), *args], cwd=str(SRC))


def wait_health(base_url: str, timeout: float = 40.0) -> dict:
    import httpx
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return httpx.get(f"{base_url}/health", timeout=2.0).json()
        except Exception as exc:  # il servizio non e' ancora pronto
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"il servizio non risponde su {base_url}: {last}")


def main() -> int:
    cfg = load_config()
    artifacts_dir = resolve(cfg["artifacts_dir"])
    logs_dir = resolve(cfg["logs_dir"])
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    for name in ("serving.log", "uvicorn.out"):
        p = os.path.join(logs_dir, name)
        if os.path.exists(p):
            os.remove(p)

    print("=" * 78, flush=True)
    print(f" {cfg['project']}", flush=True)
    print("=" * 78, flush=True)

    stage(1, "ADDESTRAMENTO DEI MODELLI")
    if run("train.py") != 0:
        print("Training fallito: interrompo.", flush=True)
        return 1

    stage(2, "CONTROLLI AUTOMATICI")
    checks_rc = run("checks.py")

    base_url = f"http://{cfg['service']['host']}:{cfg['service']['port']}"
    out_path = os.path.join(logs_dir, "uvicorn.out")
    stage(3, "AVVIO E CONTROLLO DEL SERVIZIO API")
    print(f"$ Avvio servizio su {base_url}", flush=True)
    with open(out_path, "w", encoding="utf-8") as out:
        server = subprocess.Popen(
            [PY, "-m", "uvicorn", "app:app", "--host", cfg["service"]["host"],
             "--port", str(cfg["service"]["port"])],
            cwd=str(SRC), stdout=out, stderr=subprocess.STDOUT)
    try:
        health = wait_health(base_url)
        print(f"Servizio pronto: {health}", flush=True)

        stage(4, "PROBE: MISURA DI LATENZA ED ERRORI")
        run("probe.py")

        stage(5, "GATE: DECISIONE SULLA PROMOZIONE")
        run("gate.py")

        stage(6, "DRILL: CAMBIO VERSIONE E RIPRISTINO")
        drill_rc = run("monitor.py")

        import httpx
        stage(7, "VERIFICA FINALE E RIEPILOGO")
        print("/metrics finale:", httpx.get(f"{base_url}/metrics", timeout=5.0).json(),
              flush=True)
        print("/health finale :", httpx.get(f"{base_url}/health", timeout=5.0).json(),
              flush=True)

        print(f"Controlli in-process : {'OK' if checks_rc == 0 else 'CON ERRORI'}",
              flush=True)
        gate = read_json(os.path.join(artifacts_dir, "gate_report.json"))
        print(f"Gate candidata v2    : {gate['verdict']}", flush=True)
        drill = read_json(os.path.join(artifacts_dir, "drill_results.json"))
        print(f"Drill error rate     : {drill['error_rate']} ({drill['tipo']})",
              flush=True)
        rollback = read_json(os.path.join(artifacts_dir, "rollback_log.json"))
        print(f"Stato finale         : {rollback['stato_finale']}", flush=True)
        print(f"Log servizio         : {os.path.join(logs_dir, 'serving.log')}",
              flush=True)
        print("Evidenze: artifacts/registry.json, v1/, v2/, probe_results.json,",
              flush=True)
        print("          gate_report.json, drill_results.json, rollback_log.json",
              flush=True)
        return 0 if (checks_rc == 0 and drill_rc == 0) else 1
    finally:
        stage(8, "ARRESTO DEL SERVIZIO")
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
