"""S4.5 - Gate di promozione della candidata (regola fissata PRIMA dei risultati).

Requisiti (uno per riga: misura, soglia, esito, motivazione):
  R1 accuracy(v2) >= accuracy(v1)            [protocollo comune del laboratorio]
  R2 recall classe 1 (esaurimenti) v2 >= v1  [metrica del soggetto: falso negativo = vendite perse]
  R3 p95 della candidata <= 120 ms           [vincolo operativo del laboratorio]
  R4 error rate <= 0,01 su richieste valide  [vincolo operativo del laboratorio]

Il gate NON cambia la versione attiva: la decisione e' solo una raccomandazione.
Uso: python src/gate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import load_config, read_json, resolve, write_json


def main() -> int:
    cfg = load_config()
    g = cfg["gate"]
    artifacts_dir = resolve(cfg["artifacts_dir"])
    registry = read_json(os.path.join(artifacts_dir, "registry.json"))
    probe_path = os.path.join(artifacts_dir, "probe_results.json")
    probe = read_json(probe_path) if os.path.exists(probe_path) else None

    m1 = registry["models"]["v1"]["metrics"]
    m2 = registry["models"]["v2"]["metrics"]

    rows = []

    def add(req, misura, soglia, esito, motivazione, fonte):
        rows.append({"requisito": req, "misura": misura, "soglia": soglia,
                     "esito": "PASS" if esito else "FAIL", "motivazione": motivazione,
                     "fonte": fonte})

    add("R1", f"accuracy v2={m2['accuracy']} - v1={m1['accuracy']}",
        ">= 0", m2["accuracy"] >= m1["accuracy"],
        "protocollo comune: la candidata non deve peggiorare l'accuratezza",
        f"artefatto {registry['models']['v2']['artifact_sha256'][:12]}...")

    add("R2", f"recall classe 1 v2={m2['recall_class_1']} - v1={m1['recall_class_1']}",
        ">= 0", m2["recall_class_1"] >= m1["recall_class_1"],
        "metrica del soggetto (classe 1 = esaurimento): un falso negativo e' una vendita persa",
        f"validation {cfg['data_path']}")

    if probe is None:
        add("R3", "p95 candidata non misurato", f"<= {g['max_p95_ms']} ms", False,
            "probe assente: nessuna attivazione autorizzata", "artifacts/probe_results.json mancante")
        add("R4", "error rate non misurato", f"<= {g['max_error_rate_valid']}", False,
            "probe assente: nessuna attivazione autorizzata", "artifacts/probe_results.json mancante")
        probe_ok = False
    else:
        p2 = next(r for r in probe["results"] if r["version"] == "v2")
        probe_ok = bool(p2["artifact_verified"])
        add("R3", f"p95 v2={p2['p95_ms']} ms", f"<= {g['max_p95_ms']} ms",
            p2["p95_ms"] <= g["max_p95_ms"] and probe_ok,
            "vincolo operativo del laboratorio (p95 locale <= 120 ms)",
            f"probe su {p2['version']}, fingerprint {p2['artifact_sha256'][:12]}...")
        add("R4", f"error rate v2={p2['error_rate']} su {p2['requests']} richieste valide",
            f"<= {g['max_error_rate_valid']}", p2["error_rate"] <= g["max_error_rate_valid"] and probe_ok,
            "vincolo operativo del laboratorio: le richieste valide non devono fallire",
            f"probe su {p2['version']}")

    verdict = "PASS" if all(r["esito"] == "PASS" for r in rows) else "FAIL"
    report = {
        "verdict": verdict,
        "active_version_prima_del_gate": registry["active"],
        "candidata": "v2",
        "baseline": "v1",
        "metrica_soggetto": g["subject_metric"],
        "regola_dichiarata_prima_dei_risultati": True,
        "requisiti": rows,
        "confronto_metriche": {"v1": m1, "v2": m2},
        "nota": "Un miglioramento di accuracy non compensa una violazione della latenza.",
    }
    out = os.path.join(artifacts_dir, "gate_report.json")
    write_json(out, report)

    print(f"{'REQ':<4}{'ESITO':<7}{'MISURA':<52}{'SOGLIA':<16}MOTIVAZIONE")
    for r in rows:
        print(f"{r['requisito']:<4}{r['esito']:<7}{r['misura'][:50]:<52}{r['soglia']:<16}{r['motivazione'][:60]}")
    print(f"\nesito del gate: {verdict}  (versione attiva invariata: {registry['active']})")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
