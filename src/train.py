"""S1/S2/S4 - Training ripetibile di v1 (baseline) e v2 (candidata).

Regole fisse:
- fit SOLO sullo split train; nessuna trasformazione adattata sul validation;
- stessa pipeline di preparazione dentro l'artefatto servito;
- v1 non viene ricreata quando si valuta v2: viene caricata da disco e verificata.

Uso:
    python src/train.py                 # allena v1 e v2, scrive registry (active=v1)
    python src/train.py --versions v1
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import sklearn
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (BASE_DIR, EXPECTED_FEATURES, get_split, load_artifact,
                    load_config, load_dataset, new_run_id, resolve, save_artifact, write_json)

# Configurazione decisa PRIMA di guardare i risultati del gate e non modificata dopo.
CONFIG = {
    "v1": {
        "descrizione": "Baseline: StandardScaler + LogisticRegression",
        "params": {"max_iter": 1000, "random_state": 1308, "solver": "lbfgs"},
    },
    "v2": {
        "descrizione": "Candidata: HistGradientBoostingClassifier con pesi di classe bilanciati",
        "ipotesi": "Il limite della baseline sono i falsi negativi (recall classe 1 basso): "
                   "un modello non lineare con peso maggiore alla classe 1 dovrebbe segnalare "
                   "piu' esaurimenti, accettando piu' falsi positivi.",
        "params": {"max_iter": 400, "learning_rate": 0.05, "max_depth": 3,
                   "min_samples_leaf": 20, "l2_regularization": 1.0, "random_state": 1308},
        "class_weight_balanced": True,
    },
}


def build(version: str, seed: int):
    params = dict(CONFIG[version]["params"])
    params["random_state"] = seed
    if version == "v1":
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    return HistGradientBoostingClassifier(**params)


def fit_model(version: str, X, y, seed: int):
    model = build(version, seed)
    if CONFIG[version].get("class_weight_balanced"):
        # Peso inversamente proporzionale alla frequenza: obiettivo recall della classe 1.
        weight = (y == 0).sum() / max((y == 1).sum(), 1)
        model.fit(X, y, sample_weight=np.where(y == 1, weight, 1.0))
    else:
        model.fit(X, y)
    return model


def evaluate(model, X, y_true) -> dict:
    t0 = time.perf_counter()
    pred = model.predict(X)
    latency_ms = (time.perf_counter() - t0) / max(len(y_true), 1) * 1000
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision_class_1": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall_class_1": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1_class_1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "latency_ms_per_row_batch": round(latency_ms, 4),
    }


def log_mlflow(version, run_id, metadata, artifacts_dir):
    """Collega la run a configurazione, dati e metriche. Non blocca se MLflow manca."""
    try:
        import mlflow
        db_path = os.path.join(BASE_DIR, "mlflow.db").replace("\\", "/")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("progetto8_esaurimento_scorte")
        with mlflow.start_run(run_name=f"{version}_{run_id}"):
            mlflow.log_params({"version": version, "seed": metadata["seed"],
                               "model": metadata["model"], **{f"param_{k}": v for k, v in metadata["params"].items()}})
            mlflow.log_metrics({k: v for k, v in metadata["metrics"].items() if isinstance(v, (int, float))})
            mlflow.log_artifact(os.path.join(artifacts_dir, version, "model_metadata.json"))
            mlflow.set_tags({"dataset_sha256": metadata["dataset_sha256"],
                             "features_order_hash": metadata["features_order_hash"]})
        return True
    except Exception as exc:  # pragma: no cover - dipendenza opzionale
        print(f"  [MLflow] non disponibile, run non registrata: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--versions", nargs="+", default=["v1", "v2"])
    args = ap.parse_args()

    cfg = load_config()
    data_path = args.data or resolve(cfg["data_path"])
    artifacts_dir = resolve(cfg["artifacts_dir"])
    os.makedirs(artifacts_dir, exist_ok=True)

    ds = load_dataset(data_path, expected_features=EXPECTED_FEATURES)
    print(f"dataset: {ds.path}")
    print(f"  sha256         : {ds.sha256}")
    print(f"  feature        : {len(ds.features)} (ordine derivato dall'intestazione)")
    print(f"  righe per split: {ds.split_counts}  target=1: {ds.positive_counts}")

    Xtr, ytr, idtr, Xva, yva, idva = get_split(ds)
    # Matrici numeriche: stesso formato usato dal servizio (nessun nome di colonna).
    Xtr, Xva = np.asarray(Xtr, dtype=float), np.asarray(Xva, dtype=float)
    ytr, yva = np.asarray(ytr, dtype=int), np.asarray(yva, dtype=int)
    registry_path = os.path.join(artifacts_dir, "registry.json")
    registry = {"project": cfg["project"], "active": "v1", "previous": None,
                "transitions": [], "models": {}}

    for version in args.versions:
        print(f"\n== {version}: {CONFIG[version]['descrizione']} ==")
        if version == "v1":
            model = fit_model("v1", Xtr, ytr, cfg["seed"])
            run_id = new_run_id()
        else:
            # S4.2: la vera v1 salvata viene solo caricata e valutata, non ricreata.
            _bundle, v1_meta = load_artifact("v1", artifacts_dir)
            print(f"  v1 caricata da disco: run={v1_meta['run_id']} "
                  f"fingerprint={v1_meta['artifact_sha256'][:12]}...")
            metrics_v1 = evaluate(_bundle["model"], Xva, yva)
            assert v1_meta["validation_record_ids"] == list(idva), \
                "gli ID di validation non coincidono con quelli della baseline"
            model = fit_model("v2", Xtr, ytr, cfg["seed"])
            run_id = new_run_id()
            print(f"  controllo di identita' dati: OK (stessi ID di validation di v1)")
            print(f"  v1 su validation (da artefatto): accuracy={metrics_v1['accuracy']}")

        metrics = evaluate(model, Xva, yva)
        metadata = {
            "version": version,
            "run_id": run_id,
            "model": CONFIG[version]["descrizione"],
            "ipotesi": CONFIG[version].get("ipotesi"),
            "class_weight_balanced": bool(CONFIG[version].get("class_weight_balanced")),
            "params": {k: v for k, v in CONFIG[version]["params"].items()},
            "features": list(ds.features),
            "features_order_hash": str(hash(tuple(ds.features))),
            "dataset_path": os.path.relpath(data_path, BASE_DIR).replace("\\", "/"),
            "dataset_sha256": ds.sha256,
            "split_counts": ds.split_counts,
            "positive_counts": ds.positive_counts,
            "validation_record_ids": list(idva),
            "seed": cfg["seed"],
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "metrics": metrics,
            "stage": "Production" if version == "v1" else "Staging",
        }
        model_path = save_artifact(version, model, metadata, artifacts_dir)
        meta = dict(metadata, artifact_file="model.pkl")
        from common import sha256_file
        meta["artifact_sha256"] = sha256_file(model_path)
        print(f"  metriche: {metrics}")
        print(f"  salvato : {model_path}")
        mlflow_ok = log_mlflow(version, run_id, {**meta, "metrics": metrics}, artifacts_dir)
        meta["mlflow_logged"] = mlflow_ok

        registry["models"][version] = {
            "version": version,
            "run_id": run_id,
            "dir": f"{cfg['artifacts_dir']}/{version}",
            "artifact_sha256": meta["artifact_sha256"],
            "dataset_sha256": ds.sha256,
            "model": CONFIG[version]["descrizione"],
            "params": CONFIG[version]["params"],
            "stage": meta["stage"],
            "mlflow_logged": mlflow_ok,
            "metrics": metrics,
        }

    registry["active"] = "v1"          # la baseline resta attiva finche' il gate non decide
    registry["previous"] = None
    write_json(registry_path, registry)
    print(f"\nregistry scritto: {registry_path} (active=v1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
