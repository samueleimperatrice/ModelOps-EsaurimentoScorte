"""S4.4 - Probe locale sul servizio (istanza di prova).

Per ogni versione: 1 payload valido assegnato, 3 richieste di warm-up e 30 misure
locali in sequenza. Annota p95, error rate, ambiente e fingerprint del modello
realmente misurato. Richiede il servizio attivo su host:porta della configurazione.

Uso: python src/probe.py
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx
import numpy as np
import sklearn

from common import load_config, load_requests, read_json, resolve, sha256_file

WARMUP = None
MEASURES = None


def probe_version(client: httpx.Client, base_url: str, version: str, features: list[float]):
    for _ in range(WARMUP):
        client.post(f"{base_url}/predict/{version}", json={"features": features})
    durations, statuses = [], []
    for _ in range(MEASURES):
        t0 = time.perf_counter()
        resp = client.post(f"{base_url}/predict/{version}", json={"features": features})
        durations.append((time.perf_counter() - t0) * 1000)
        statuses.append(resp.status_code)
    statuses = np.array(statuses)
    return {
        "version": version,
        "requests": int(len(statuses)),
        "errors_status_ge_400": int((statuses >= 400).sum()),
        "error_rate": round(float((statuses >= 400).mean()), 4),
        "p50_ms": round(float(np.percentile(durations, 50)), 2),
        "p95_ms": round(float(np.percentile(durations, 95)), 2),
        "max_ms": round(float(np.max(durations)), 2),
    }


def main() -> int:
    global WARMUP, MEASURES
    cfg = load_config()
    WARMUP = int(cfg["probe"]["warmup"])
    MEASURES = int(cfg["probe"]["measures"])
    artifacts_dir = resolve(cfg["artifacts_dir"])
    registry = read_json(os.path.join(artifacts_dir, "registry.json"))
    requests_csv = load_requests(resolve(cfg["requests_path"]))
    payload_id = cfg["probe"]["payload_request_id"]
    payload = next(r for r in requests_csv if r["request_id"] == payload_id)
    assert len(payload["features"]) == 30, "la probe richiede un payload valido a 30 feature"

    base_url = f"http://{cfg['service']['host']}:{cfg['service']['port']}"
    print(f"probe su {base_url} - payload={payload_id}, "
          f"warm-up={WARMUP}, misure={MEASURES}")

    results = []
    with httpx.Client(timeout=10.0) as client:
        for version in ("v1", "v2"):
            res = probe_version(client, base_url, version, payload["features"])
            meta_dir = os.path.join(artifacts_dir, version)
            res["artifact_sha256"] = sha256_file(os.path.join(meta_dir, "model.pkl"))
            res["expected_artifact_sha256"] = registry["models"][version]["artifact_sha256"]
            res["artifact_verified"] = res["artifact_sha256"] == res["expected_artifact_sha256"]
            res["run_id"] = registry["models"][version]["run_id"]
            results.append(res)
            print(f"  {version}: p95={res['p95_ms']} ms  p50={res['p50_ms']} ms  "
                  f"error rate={res['error_rate']}  artefatto verificato={res['artifact_verified']}")

    report = {
        "payload_request_id": payload_id,
        "warmup": WARMUP,
        "measures": MEASURES,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "http_client": f"httpx {httpx.__version__}",
            "cpu_count": os.cpu_count(),
        },
        "results": results,
        "note": "Misure locali in-process/HTTP sullo stesso host: non rappresentano il carico reale.",
    }
    out = os.path.join(artifacts_dir, "probe_results.json")
    from common import write_json
    write_json(out, report)
    print(f"probe scritta: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
