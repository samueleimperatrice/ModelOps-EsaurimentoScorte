"""S5 - Drill di attivazione e ripristino con le versioni realmente servite.

Segue lo stesso ordine del CSV assegnato (v1 -> v2 -> v1) eseguendo le transizioni
reali sul servizio: prima della richiesta diretta a una versione diversa da quella
attiva, chiama /admin/activate/{version}. Poi serve la richiesta incompleta (422).

Produce:
  artifacts/drill_results.json  - tabella delle 4 richieste + error rate
  artifacts/rollback_log.json   - registro delle transizioni e stato finale

Uso: python src/monitor.py      (richiede il servizio attivo su host:porta)
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx

from common import load_config, load_requests, read_json, resolve, sha256_file, write_json


def main() -> int:
    cfg = load_config()
    artifacts_dir = resolve(cfg["artifacts_dir"])
    base_url = f"http://{cfg['service']['host']}:{cfg['service']['port']}"
    requests_csv = load_requests(resolve(cfg["requests_path"]))

    gate_path = os.path.join(artifacts_dir, "gate_report.json")
    gate = read_json(gate_path) if os.path.exists(gate_path) else None
    gate_verdict = gate["verdict"] if gate else "non valutato"

    rows, transitions = [], []
    with httpx.Client(timeout=10.0) as client:
        current_active = client.get(f"{base_url}/health").json()["active_version"]
        print(f"versione attiva all'inizio del drill: {current_active}")

        for req in requests_csv:
            target = req["model_version"]
            if target != current_active:
                tr = client.post(f"{base_url}/admin/activate/{target}").json()
                transitions.append(tr)
                print(f"  transizione: {tr['stato_prima']} -> {tr['stato_dopo']} "
                      f"(run={tr['run_id']}, fingerprint={tr['artifact_sha256'][:12]}...)")
                current_active = tr["stato_dopo"]

            t0 = time.perf_counter()
            resp = client.post(f"{base_url}/predict/{target}", json={"features": req["features"]})
            ms = (time.perf_counter() - t0) * 1000
            try:
                body = resp.json()
            except Exception:
                body = {}
            rows.append({
                "n": len(rows) + 1,
                "request_id": req["request_id"],
                "versione_chiesta": target,
                "versione_servita": body.get("model_version"),
                "run_id_servita": body.get("run_id"),
                "n_feature": len(req["features"]),
                "status": resp.status_code,
                "expected_status": req["expected_status"],
                "coerente": resp.status_code == req["expected_status"],
                "prediction": body.get("prediction"),
                "probability": body.get("probability"),
                "ms": round(ms, 2),
            })

    error_rate = sum(1 for r in rows if r["status"] >= 400) / len(rows)
    final_health = httpx.get(f"{base_url}/health", timeout=10.0).json()
    final_artifact = sha256_file(os.path.join(artifacts_dir, final_health["active_version"], "model.pkl"))
    registry = read_json(os.path.join(artifacts_dir, "registry.json"))
    baseline_sha = registry["models"]["v1"]["artifact_sha256"]

    drill = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "gate_verdict": gate_verdict,
        "tipo": "attivazione documentata" if gate_verdict == "PASS" else "drill locale dichiarato",
        "richieste": rows,
        "error_rate": round(error_rate, 4),
        "error_rate_atteso": 0.25,
        "nota": "L'error rate del drill e' tenuto separato dalle misure della probe.",
    }
    write_json(os.path.join(artifacts_dir, "drill_results.json"), drill)

    rollback = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "motivo": "drill di ripristino alla baseline",
        "transizioni": transitions,
        "stato_finale": {
            "active_version": final_health["active_version"],
            "active_run_id": final_health["active_run_id"],
            "artifact_sha256": final_artifact,
            "coincide_con_baseline_v1": final_artifact == baseline_sha,
        },
    }
    write_json(os.path.join(artifacts_dir, "rollback_log.json"), rollback)

    print(f"\n{'#':<3}{'richiesta':<10}{'chiesta':<9}{'servita':<9}{'n feat':<8}"
          f"{'status':<8}{'atteso':<8}{'pred':<6}{'ms':<8}")
    for r in rows:
        print(f"{r['n']:<3}{r['request_id']:<10}{r['versione_chiesta']:<9}"
              f"{str(r['versione_servita']):<9}{r['n_feature']:<8}{r['status']:<8}"
              f"{r['expected_status']:<8}{str(r['prediction']):<6}{r['ms']:<8}")
    print(f"\nerror rate drill = {sum(1 for r in rows if r['status'] >= 400)}/{len(rows)} = {error_rate:.2f}")
    print(f"stato finale: attiva={final_health['active_version']} "
          f"run={final_health['active_run_id']} "
          f"artefatto==baseline_v1: {final_artifact == baseline_sha}")
    return 0 if (final_health["active_version"] == "v1" and final_artifact == baseline_sha) else 1


if __name__ == "__main__":
    raise SystemExit(main())
