"""S3 - Servizio di serving (FastAPI).

- identifica la versione attiva dal registry e carica solo l'artefatto servito;
- valida il contratto di input (30 feature finite, ordine dai metadati) -> 422;
- espone /predict/{version}, /health, /metrics e /admin/activate/{version};
- registra timestamp, request ID, versione, esito HTTP e durata (anche per i 422)
  senza mai scrivere i payload grezzi nei log.

Avvio: python -m uvicorn app:app --host 127.0.0.1 --port 8000   (cwd = src/)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from common import (ArtifactMismatch, EXPECTED_FEATURES, load_artifact, load_config,
                    read_json, resolve, write_json)

CFG = load_config()
ARTIFACTS_DIR = resolve(CFG["artifacts_dir"])
REGISTRY_PATH = os.path.join(ARTIFACTS_DIR, "registry.json")
LOG_PATH = resolve(CFG["logs_dir"], "serving.log")
WINDOW_SECONDS = int(CFG["service"]["metrics_window_seconds"])
PREDICT_PATH_PREFIX = "/predict"

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s serving %(message)s")
log = logging.getLogger("serving")

app = FastAPI(title="Progetto 8 - Rischio esaurimento scorte",
              description="Classificazione: esaurimento scorte entro 7 giorni (target 1).")

# Stato in memoria: versione attiva, artefatti caricati, eventi per le metriche.
REGISTRY = read_json(REGISTRY_PATH)
LOADED: dict[str, dict] = {}
EVENTS: deque = deque(maxlen=10000)          # (epoch, version, status, duration_ms)

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class PredictRequest(BaseModel):
    features: list[FiniteFloat] = Field(..., description="30 valori numerici finiti, "
                                                         "ordine = dizionario del README")


def _log_line(**fields) -> None:
    line = json.dumps(fields, ensure_ascii=False)
    log.info(line)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _record_event(version: str, status: int, duration_ms: float) -> None:
    EVENTS.append((time.time(), version, status, duration_ms))


def _save_registry() -> None:
    write_json(REGISTRY_PATH, REGISTRY)


def _ensure_loaded(version: str) -> dict:
    """Carica e verifica l'artefatto della versione richiesta (lazy)."""
    if version in LOADED:
        return LOADED[version]
    bundle, meta = load_artifact(version, ARTIFACTS_DIR)
    if len(bundle["features"]) != len(EXPECTED_FEATURES):
        raise ArtifactMismatch(f"schema incoerente per {version}")
    LOADED[version] = {"bundle": bundle, "meta": meta}
    _log_line(ts=dt.datetime.now().isoformat(timespec="seconds"), event="artifact_loaded",
              version=version, run_id=meta["run_id"],
              artifact_sha256=meta["artifact_sha256"])
    return LOADED[version]


def _predict(version: str, payload: PredictRequest, request_id: str) -> dict:
    t0 = time.perf_counter()
    if version not in REGISTRY["models"]:
        _record_event(version, 404, (time.perf_counter() - t0) * 1000)
        raise HTTPException(404, f"versione sconosciuta: {version}")

    # 3.2 - quantita' e ordine coerenti con i metadati: 422 PRIMA di chiamare il modello.
    if len(payload.features) != len(EXPECTED_FEATURES):
        ms = (time.perf_counter() - t0) * 1000
        _record_event(version, 422, ms)
        _log_line(ts=dt.datetime.now().isoformat(timespec="seconds"), request_id=request_id,
                  version=version, status=422, duration_ms=round(ms, 2),
                  note=f"attese {len(EXPECTED_FEATURES)} feature, ricevute {len(payload.features)}")
        raise HTTPException(422, f"Attese {len(EXPECTED_FEATURES)} feature, "
                                 f"ricevute {len(payload.features)}")

    entry = _ensure_loaded(version)
    probability = float(entry["bundle"]["model"].predict_proba([payload.features])[0][1])
    ms = (time.perf_counter() - t0) * 1000
    out = {"request_id": request_id, "model_version": version,
           "run_id": entry["meta"]["run_id"], "prediction": int(probability >= 0.5),
           "probability": round(probability, 4), "threshold": 0.5}
    _record_event(version, 200, ms)
    _log_line(ts=dt.datetime.now().isoformat(timespec="seconds"), request_id=request_id,
              version=version, status=200, prediction=out["prediction"],
              probability=out["probability"], duration_ms=round(ms, 2))
    return out


@app.middleware("http")
async def _attach_ids(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    request.state.t0 = time.perf_counter()
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    """Input non valido (es. valori non finiti o tipi errati) -> 422 registrato nei log."""
    rid = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    ms = (time.perf_counter() - getattr(request.state, "t0", time.perf_counter())) * 1000
    version = request.url.path.rsplit("/", 1)[-1] if request.url.path.startswith(PREDICT_PATH_PREFIX) else "n/a"
    _record_event(version, 422, ms)
    _log_line(ts=dt.datetime.now().isoformat(timespec="seconds"), request_id=rid,
              version=version, status=422, duration_ms=round(ms, 2), note="validazione input")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "request_id": rid})


@app.get("/", include_in_schema=False)
def root():
    """Comodita': chi apre la radice viene portato alla documentazione interattiva."""
    return RedirectResponse(url="/docs")


@app.post("/predict/{version}")
def predict_version(version: str, payload: PredictRequest, request: Request):
    return _predict(version, payload, request.state.request_id)


@app.post("/predict")
def predict_active(payload: PredictRequest, request: Request):
    return _predict(REGISTRY["active"], payload, request.state.request_id)


@app.get("/health")
def health():
    active = REGISTRY["active"]
    meta = REGISTRY["models"][active]
    return {"status": "ok", "active_version": active, "active_run_id": meta["run_id"],
            "active_artifact_sha256": meta["artifact_sha256"],
            "versions": list(REGISTRY["models"]), "expected_feature_count": len(EXPECTED_FEATURES),
            "dataset_sha256": REGISTRY["models"][active].get("dataset_sha256"),
            "previous_version": REGISTRY.get("previous")}


@app.get("/metrics")
def metrics():
    """Finestra esplicita sulle sole predizioni (health/metrics non contano)."""
    now = time.time()
    window = [(ts, v, st, ms) for (ts, v, st, ms) in EVENTS if now - ts <= WINDOW_SECONDS]
    durations = [ms for *_, ms in window]
    errors = [e for e in window if e[2] >= 400]
    by_version: dict[str, int] = {}
    for _, v, _, _ in window:
        by_version[v] = by_version.get(v, 0) + 1
    return {
        "window_seconds": WINDOW_SECONDS,
        "requests": len(window),
        "errors_status_ge_400": len(errors),
        "error_rate": round(len(errors) / len(window), 4) if window else 0.0,
        "p95_ms": round(float(np.percentile(durations, 95)), 2) if durations else None,
        "by_version": by_version,
        "active_version": REGISTRY["active"],
    }


@app.post("/admin/activate/{version}")
def activate(version: str):
    """Transizione di versione: carica (verifica) l'artefatto e registra lo stato prima/dopo."""
    if version not in REGISTRY["models"]:
        raise HTTPException(404, f"versione sconosciuta: {version}")
    try:
        entry = _ensure_loaded(version)
    except ArtifactMismatch as exc:
        raise HTTPException(409, f"attivazione rifiutata: {exc}")

    before = REGISTRY["active"]
    REGISTRY["previous"] = before
    REGISTRY["active"] = version
    for other, info in REGISTRY["models"].items():
        info["stage"] = "Production" if other == version else "Archived"
    transition = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "azione": f"activate {version}", "stato_prima": before, "stato_dopo": version,
        "run_id": entry["meta"]["run_id"],
        "artifact_sha256": entry["meta"]["artifact_sha256"],
    }
    REGISTRY.setdefault("transitions", []).append(transition)
    _save_registry()
    _log_line(ts=transition["ts"], event="transition", **{k: transition[k] for k in
              ("azione", "stato_prima", "stato_dopo", "run_id", "artifact_sha256")})
    return transition
