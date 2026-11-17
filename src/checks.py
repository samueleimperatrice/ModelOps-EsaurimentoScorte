"""Controlli del contratto (S1/S2/S3) eseguiti come client in-process.

Verifica:
- il lettore del CSV rifiuta file non conformi con motivo esplicito;
- il caricatore dell'artefatto rifiuta una copia che non corrisponde ai metadati;
- il servizio risponde 200 su una richiesta valida e 422 su un input incompleto;
- i payload delle richieste non appartengono al training.

Uso: python src/checks.py     (richiede artifacts/ da un train precedente)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import common
from common import (ArtifactMismatch, DatasetContractError, EXPECTED_FEATURES,
                    load_artifact, load_config, load_dataset, load_requests, resolve)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, detail))
            print(f"  OK   {name} {detail}")
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
            print(f"  KO   {name} -> {exc}")
        except Exception as exc:  # errore inatteso
            RESULTS.append((name, False, f"errore inatteso: {exc!r}"))
            print(f"  KO   {name} -> errore inatteso: {exc!r}")
        return fn
    return deco


def _expect_dataset_error(text: str) -> str:
    try:
        load_dataset(text, expected_features=EXPECTED_FEATURES, strict_splits=False)
    except DatasetContractError as exc:
        return str(exc)
    raise AssertionError("il file non conforme e' stato accettato")


def main() -> int:
    cfg = load_config()
    data_path = resolve(cfg["data_path"])
    requests_path = resolve(cfg["requests_path"])
    artifacts_dir = resolve(cfg["artifacts_dir"])
    print("== controlli contratto dati ==")

    @check("dataset assegnato conforme (1000 righe, 30 feature, split 800/200)")
    def _():
        ds = load_dataset(data_path, expected_features=EXPECTED_FEATURES)
        assert ds.split_counts == {"train": 800, "validation": 200}, ds.split_counts
        assert len(ds.features) == 30
        return f"sha256={ds.sha256[:12]}..."

    with tempfile.TemporaryDirectory() as tmp:
        @check("file vuoto rifiutato")
        def _():
            p = os.path.join(tmp, "vuoto.csv")
            open(p, "w").close()
            return _expect_dataset_error(p)

        @check("colonne duplicate rifiutate")
        def _():
            p = os.path.join(tmp, "dup.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("record_id,split,target,record_id\n1,train,0,2\n")
            return _expect_dataset_error(p)

        @check("record_id ripetuti rifiutati")
        def _():
            import pandas as pd
            df = pd.read_csv(data_path)
            df.loc[1, "record_id"] = df.loc[0, "record_id"]
            p = os.path.join(tmp, "id_dup.csv")
            df.to_csv(p, index=False)
            return _expect_dataset_error(p)

        @check("valori non finiti rifiutati")
        def _():
            import pandas as pd
            df = pd.read_csv(data_path)
            df.loc[0, EXPECTED_FEATURES[0]] = float("nan")
            p = os.path.join(tmp, "nan.csv")
            df.to_csv(p, index=False)
            return _expect_dataset_error(p)

        @check("split inatteso rifiutato")
        def _():
            import pandas as pd
            df = pd.read_csv(data_path)
            df.loc[0, "split"] = "test"
            p = os.path.join(tmp, "split.csv")
            df.to_csv(p, index=False)
            return _expect_dataset_error(p)

    print("== controlli caricatore artefatti (S2) ==")

    @check("artefatto v1 originale caricato e verificato")
    def _():
        bundle, meta = load_artifact("v1", artifacts_dir)
        return f"run={meta['run_id']} fingerprint={meta['artifact_sha256'][:12]}..."

    @check("copia dell'artefatto non corrispondente ai metadati rifiutata")
    def _():
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "v1")
            shutil.copytree(os.path.join(artifacts_dir, "v1"), dst)
            with open(os.path.join(dst, "model.pkl"), "ab") as fh:
                fh.write(b"corruzione")
            try:
                load_artifact("v1", tmp)
            except ArtifactMismatch as exc:
                return f"rifiutata: {exc}"
            raise AssertionError("la copia corrotta e' stata accettata")

    print("== controllo client in-process (S3) ==")
    from fastapi.testclient import TestClient
    import app as app_module
    client = TestClient(app_module.app)
    requests_csv = load_requests(requests_path)
    valid = next(r for r in requests_csv if r["expected_status"] == 200)
    incomplete = next(r for r in requests_csv if r["expected_status"] == 422)

    @check("richiesta valida -> 200 con versione, run ID e request ID")
    def _():
        resp = client.post("/predict/v1", json={"features": valid["features"]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model_version"] == "v1" and body["run_id"] and body["request_id"]
        assert body["prediction"] in (0, 1)
        return f"pred={body['prediction']} prob={body['probability']}"

    @check("richiesta incompleta (29 valori) -> 422")
    def _():
        resp = client.post("/predict/v1", json={"features": incomplete["features"]})
        assert resp.status_code == 422, f"status={resp.status_code}"
        return f"{len(incomplete['features'])} valori -> 422"

    @check("valore non numerico -> 422")
    def _():
        bad = list(valid["features"])
        bad[0] = "non-numerico"
        resp = client.post("/predict/v1", json={"features": bad})
        assert resp.status_code == 422, f"status={resp.status_code}"
        return "tipo errato rifiutato"

    @check("health espone versione attiva e run ID")
    def _():
        body = client.get("/health").json()
        assert body["active_version"] and body["active_run_id"]
        return f"active={body['active_version']} run={body['active_run_id']}"

    @check("metrics conta le predizioni (escluse health/metrics)")
    def _():
        body = client.get("/metrics").json()
        assert body["requests"] >= 3, body
        assert body["errors_status_ge_400"] >= 2, body
        return f"req={body['requests']} errori={body['errors_status_ge_400']} p95={body['p95_ms']}ms"

    @check("i payload assegnati NON appartengono al train (non entrano nel training)")
    def _():
        ds = load_dataset(data_path, expected_features=EXPECTED_FEATURES)
        train = ds.df[ds.df["split"] == "train"]
        valid_matrix = ds.df[ds.df["split"] == "validation"][ds.features].to_numpy(dtype=float)
        train_matrix = train[ds.features].to_numpy(dtype=float)
        in_validation = 0
        for req in requests_csv:
            if len(req["features"]) != 30:
                continue
            vec = np.array(req["features"], dtype=float)
            assert not np.isclose(train_matrix, vec).all(axis=1).any(), \
                f"{req['request_id']} coincide con una riga del TRAIN"
            if np.isclose(valid_matrix, vec).all(axis=1).any():
                in_validation += 1
        return (f"{len(requests_csv)} richieste: 0 nel train, "
                f"{in_validation} coincidono con righe di validation")

    ok = sum(1 for _, passed, _ in RESULTS if passed)
    print(f"\nrisultato controlli: {ok}/{len(RESULTS)} OK")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
