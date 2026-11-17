"""Contratto condiviso del Progetto 8 - Esaurimento delle scorte di magazzino.

Questo modulo centralizza tutto cio' che deve restare coerente in ogni fase
(contratto dati, training, serving, monitoraggio):

1. lettura e validazione dei CSV assegnati (schema, split, classi, ID, finitenza);
2. derivazione dell'ordine delle feature dall'intestazione del dataset;
3. fingerprint sha256 dei file e degli artefatti;
4. salvataggio e caricamento degli artefatti con verifica di identita'/schema;
5. lettura del CSV delle richieste di monitoraggio.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

# --- Percorsi -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# --- Colonne tecniche (NON sono feature) --------------------------------------
ID_COL = "record_id"
SPLIT_COL = "split"
TARGET_COL = "target"
TECH_COLS = (ID_COL, SPLIT_COL, TARGET_COL)

# Ordine delle feature dell'API: identico all'intestazione del dataset assegnato.
EXPECTED_FEATURES = [
    "scorta_disponibile_unita", "scorta_riservata_unita", "copertura_stimata_giorni",
    "quota_scorta_utilizzabile_pct", "rotture_scorta_storiche_90g",
    "intervallo_riordino_pianificato_giorni", "vendite_medie_giornaliere_30g",
    "crescita_domanda_7g_pct", "variabilita_domanda_pct", "ordini_clienti_aperti_unita",
    "quota_vendite_promozionali_storica_pct", "previsione_domanda_7g_unita",
    "consegna_fornitore_media_giorni", "variabilita_consegna_fornitore_giorni",
    "ritardi_fornitore_90g_pct", "quota_consegne_complete_90g_pct",
    "fornitori_alternativi_disponibili", "affidabilita_fornitore_indice",
    "quantita_in_arrivo_unita", "giorni_al_prossimo_arrivo_previsto",
    "quota_ordini_acquisto_confermati_pct", "margine_scorta_sicurezza_unita",
    "giorni_da_ultimo_riordino", "capacita_ricevimento_disponibile_pct",
    "giorni_a_prossima_promozione", "incremento_domanda_promozione_stimato_pct",
    "indice_stagionalita_domanda", "articoli_sostitutivi_disponibili",
    "quota_richieste_urgenti_30g_pct", "lotto_minimo_riordino_unita",
]
assert len(EXPECTED_FEATURES) == 30

# Split assegnati dal docente (scheda del soggetto): non vengono rigenerati.
EXPECTED_SPLIT_TOTAL = {"train": 800, "validation": 200}
EXPECTED_SPLIT_POSITIVE = {"train": 310, "validation": 78}


class DatasetContractError(ValueError):
    """Violazione del contratto dati: il file assegnato non e' utilizzabile."""


class ArtifactMismatch(RuntimeError):
    """L'artefatto su disco non corrisponde ai metadati della sua versione."""


# --- Configurazione -----------------------------------------------------------
def load_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve(*parts: str) -> str:
    """Percorso assoluto a partire dalla radice del progetto."""
    return os.path.join(BASE_DIR, *parts)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def sha256_file(path: str) -> str:
    """Fingerprint calcolato sui byte del file (nome e numero di righe non bastano)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_run_id() -> str:
    import datetime as dt
    import uuid
    return f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


# --- Lettura del dataset ------------------------------------------------------
@dataclass
class Dataset:
    """Risultato del lettore: dati validati + informazioni di contratto."""
    df: pd.DataFrame
    features: list[str]
    path: str
    sha256: str
    split_counts: dict = field(default_factory=dict)
    positive_counts: dict = field(default_factory=dict)


def _read_header(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        try:
            return next(csv.reader(fh))
        except StopIteration:
            raise DatasetContractError(f"file vuoto: {path}")


def load_dataset(path: str, expected_features: list[str] | None = None,
                 strict_splits: bool = True) -> Dataset:
    """Legge il CSV assegnato e ne controlla il contratto (senza modificarlo).

    Controlli: file vuoto, colonne duplicate, colonne tecniche mancanti,
    ID mancanti o ripetuti, valori non numerici o non finiti, split inattesi,
    classi non binarie, numerosita' degli split e ordine delle feature.
    """
    if not os.path.exists(path):
        raise DatasetContractError(f"file non trovato: {path}")

    columns = _read_header(path)
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise DatasetContractError(f"colonne duplicate: {duplicates}")

    missing_tech = [c for c in TECH_COLS if c not in columns]
    if missing_tech:
        raise DatasetContractError(f"colonne tecniche mancanti: {missing_tech}")

    # Ordine delle feature ricavato dall'intestazione, non hard-coded.
    features = [c for c in columns if c not in TECH_COLS]
    if expected_features is not None and features != expected_features:
        raise DatasetContractError(
            "ordine/nomi delle feature diversi dal contratto:\n"
            f"  attesi : {expected_features}\n  trovati: {features}"
        )

    df = pd.read_csv(path)
    if df.empty:
        raise DatasetContractError("dataset senza righe")

    if df[ID_COL].isna().any():
        raise DatasetContractError(f"{ID_COL} contiene valori mancanti")
    if df[ID_COL].duplicated().any():
        dup = df.loc[df[ID_COL].duplicated(), ID_COL].head(5).tolist()
        raise DatasetContractError(f"{ID_COL} ripetuti (esempi): {dup}")

    try:
        matrix = df[features].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(f"feature non numeriche: {exc}") from exc
    if not np.isfinite(matrix).all():
        raise DatasetContractError("le feature contengono valori non finiti (NaN/inf)")

    splits = set(df[SPLIT_COL].unique())
    if splits != set(EXPECTED_SPLIT_TOTAL):
        raise DatasetContractError(f"split inattesi: {sorted(splits)}")

    targets = set(df[TARGET_COL].unique())
    if not targets <= {0, 1}:
        raise DatasetContractError(f"classi attese in {{0,1}}, trovate: {sorted(targets)}")

    split_counts, positive_counts = {}, {}
    for split in EXPECTED_SPLIT_TOTAL:
        part = df[df[SPLIT_COL] == split]
        split_counts[split] = int(len(part))
        positive_counts[split] = int(part[TARGET_COL].sum())

    if strict_splits:
        for split, total in EXPECTED_SPLIT_TOTAL.items():
            if split_counts[split] != total:
                raise DatasetContractError(
                    f"split {split}: attese {total} righe, trovate {split_counts[split]}"
                )
            if positive_counts[split] != EXPECTED_SPLIT_POSITIVE[split]:
                raise DatasetContractError(
                    f"split {split}: attesi {EXPECTED_SPLIT_POSITIVE[split]} target=1, "
                    f"trovati {positive_counts[split]}"
                )

    return Dataset(df=df, features=features, path=path, sha256=sha256_file(path),
                   split_counts=split_counts, positive_counts=positive_counts)


def get_split(ds: Dataset):
    """Restituisce train/validation come (X, y, record_id) senza trasformazioni."""
    tr, va = ds.df[ds.df[SPLIT_COL] == "train"], ds.df[ds.df[SPLIT_COL] == "validation"]
    return (
        tr[ds.features], tr[TARGET_COL], tr[ID_COL].tolist(),
        va[ds.features], va[TARGET_COL], va[ID_COL].tolist(),
    )


# --- Lettura delle richieste di monitoraggio ---------------------------------
def load_requests(path: str) -> list[dict]:
    """Legge richieste_monitoraggio.csv: request_id, model_version, payload_json, expected_status."""
    if not os.path.exists(path):
        raise DatasetContractError(f"file non trovato: {path}")
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            payload = json.loads(row["payload_json"])
            out.append({
                "request_id": row["request_id"],
                "model_version": row["model_version"],
                "features": [float(v) for v in payload["features"]],
                "expected_status": int(row["expected_status"]),
            })
    return out


# --- Artefatti: salvataggio e caricamento verificato -------------------------
def save_artifact(version: str, model, metadata: dict, artifacts_dir: str) -> str:
    """Salva model.pkl + model_metadata.json nella cartella della versione."""
    version_dir = os.path.join(artifacts_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    model_path = os.path.join(version_dir, "model.pkl")
    joblib.dump({"model": model, "features": metadata["features"], "version": version},
                model_path)
    metadata = dict(metadata, artifact_file="model.pkl",
                    artifact_sha256=sha256_file(model_path))
    write_json(os.path.join(version_dir, "model_metadata.json"), metadata)
    return model_path


def load_artifact(version: str, artifacts_dir: str) -> tuple[dict, dict]:
    """Carica un artefatto verificando identita', schema e fingerprint.

    Solleva ArtifactMismatch con motivo esplicito se la copia su disco non
    corrisponde ai metadati della versione.
    """
    version_dir = os.path.join(artifacts_dir, version)
    meta_path = os.path.join(version_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        raise ArtifactMismatch(f"metadati mancanti per la versione {version}: {meta_path}")
    meta = read_json(meta_path)
    model_path = os.path.join(version_dir, meta.get("artifact_file", "model.pkl"))
    if not os.path.exists(model_path):
        raise ArtifactMismatch(f"artefatto mancante per la versione {version}: {model_path}")

    actual = sha256_file(model_path)
    if meta.get("artifact_sha256") != actual:
        raise ArtifactMismatch(
            f"fingerprint dell'artefatto {version} non coerente: "
            f"atteso {meta.get('artifact_sha256')}, calcolato {actual}"
        )
    if meta.get("features") != EXPECTED_FEATURES:
        raise ArtifactMismatch(f"schema delle feature incoerente per la versione {version}")

    bundle = joblib.load(model_path)
    if bundle.get("features") != meta["features"]:
        raise ArtifactMismatch(f"lo schema salvato nel modello {version} non coincide con i metadati")
    if bundle.get("version") != version:
        raise ArtifactMismatch(f"il modello salvato dichiara versione {bundle.get('version')}, attesa {version}")
    return bundle, meta
