# Progetto 8 — Esaurimento delle scorte di magazzino

Project work di gruppo (Lab 13): pipeline di classificazione, serving FastAPI,
monitoraggio delle richieste, gate di promozione e drill di ripristino.

## 1. Soggetto, scopo e classi

- **Domanda:** quali coppie articolo–magazzino rischiano di esaurire le scorte nei prossimi 7 giorni?
- **target = 1:** scorte esaurite almeno una volta nei 7 giorni successivi.
- **target = 0:** nessun esaurimento nei 7 giorni successivi.
- **Momento della previsione:** inizio della giornata di pianificazione; tutte le 30 feature sono
  disponibili in quel momento (storico vendite/fornitori + stime di domanda e arrivi).
- **Uso della previsione:** generare una lista di segnalazioni di riordino da far verificare al
  planner. Il modello non ordina da solo: prioritizza l'attenzione umana.

**Costo dei due errori**

| Errore | Significato operativo | Costo |
| --- | --- | --- |
| Falso negativo (target 1 previsto 0) | esaurimento non segnalato | vendite perse, cliente insoddisfatto |
| Falso positivo (target 0 previsto 1) | riordino non necessario | scorte immobilizzate, costo d'ordine, rischio obsolescenza |

Per questo la **metrica del soggetto è il recall della classe 1** (quanti esaurimenti reali
riusciamo a intercettare) e la **precision della classe 1** misura il rumore delle segnalazioni
di riordino. Le due metriche vanno lette insieme: la matrice di confusione è riportata nel registry.

## 2. Proposta iniziale (Lab 13, sezione 3)

1. **Evento previsto / uso:** esaurimento entro 7 giorni → lista di segnalazioni per il planner.
2. **Classi:** 0 = nessun esaurimento, 1 = almeno un esaurimento nei 7 giorni.
3. **Momento della previsione:** inizio giornata di pianificazione; nessuna feature contiene l'esito futuro.
4. **Conseguenze:** falso negativo = vendite perse; falso positivo = scorte/costi inutili.
5. **Metriche e limiti:** recall e precision classe 1, accuracy, matrice di confusione;
   limiti operativi: p95 ≤ 120 ms, error rate ≤ 0,01 sulle richieste valide, accuracy della
   candidata ≥ baseline (**regola scritta prima di vedere i risultati**).
6. **Attività:** vedi §11 (divisione tra dati/training, servizio/telemetria, aggiornamento/documentazione).

**Percorso:** dati → training → artefatto e metadati → servizio → osservazioni → decisione di
aggiornamento → ripristino. Ogni passaggio è verificato da un file di evidenza in `artifacts/`
(§8) e la versione precedente resta sempre su disco e ricaricabile con una sola chiamata.

## 3. Dati e contratto

File assegnati (nella cartella del progetto, non rigenerati):

| File | Ruolo |
| --- | --- |
| `dataset.csv` | 1000 righe, 30 feature numeriche, `record_id`, `split`, `target` |
| `richieste_monitoraggio.csv` | `request_id`, `model_version`, `payload_json`, `expected_status` |

- Split assegnato e stratificato: train 800 (490 negativi / 310 positivi), validation 200 (122 / 78).
- `record_id`, `split` e `target` hanno ruolo tecnico e **non sono feature**.
- **Ordine delle feature**: derivato dall'intestazione del CSV (`src/common.py`), verificato
  contro il dizionario del README e salvato nei metadati di ogni artefatto.
- Fingerprint: sha256 sui byte di `dataset.csv` = `1c4079eb2069…` (il comando lo ristampa a ogni run).
- I payload delle richieste **non entrano nel training**: 0 coincidono con righe di train,
  3 coincidono con righe di validation (controllo automatico in `src/checks.py`).
- Nessuna trasformazione è adattata sul validation: `StandardScaler` è dentro la pipeline e
  viene fittato solo sul train.

Controlli del lettore (`src/checks.py`, 14/14 OK): file vuoto, colonne duplicate, `record_id`
mancanti/ripetuti, valori non finiti, split inattesi, classi non binarie, numerosità degli split.

## 4. Modelli

| Versione | Modello | Configurazione |
| --- | --- | --- |
| v1 (baseline) | `StandardScaler` + `LogisticRegression` | `max_iter=1000`, `solver=lbfgs`, `random_state=1308` |
| v2 (candidata) | `HistGradientBoostingClassifier` con pesi di classe bilanciati | `max_iter=400`, `lr=0.05`, `max_depth=3`, `min_samples_leaf=20`, `l2=1.0` |

**Ipotesi di v2** (dichiarata prima del gate): il limite della baseline sono i falsi negativi
(recall classe 1 ≈ 0,51). Un modello non lineare con peso maggiore alla classe 1 dovrebbe
segnalare più esaurimenti, accettando più falsi positivi.

## 5. Risultati (ultima esecuzione — valori completi in `artifacts/*.json`)

| Metrica | v1 baseline | v2 candidata |
| --- | ---: | ---: |
| accuracy | **0,710** | 0,655 |
| precision classe 1 | 0,667 | 0,554 |
| recall classe 1 | 0,513 | **0,590** |
| f1 classe 1 | 0,580 | 0,571 |
| matrice di confusione (tn/fp/fn/tp) | 102 / 20 / 38 / 40 | 85 / 37 / 32 / 46 |

Interpretazione: v2 intercetta 6 esaurimenti reali in più (tp 46 vs 40, fn 38 → 32) ma produce
17 falsi positivi in più (37 vs 20). L'ipotesi è confermata **solo in parte**: il recall migliora,
l'accuracy peggiora.

### Gate di promozione (`artifacts/gate_report.json`)

| Requisito | Misura | Soglia | Esito |
| --- | --- | --- | --- |
| R1 accuracy v2 ≥ v1 | 0,655 vs 0,710 | ≥ 0 | **FAIL** |
| R2 recall classe 1 v2 ≥ v1 (metrica del soggetto) | 0,590 vs 0,513 | ≥ 0 | PASS |
| R3 p95 candidata | 19,6 ms | ≤ 120 ms | PASS |
| R4 error rate su richieste valide | 0,00 (30 richieste) | ≤ 0,01 | PASS |

**Esito: FAIL → v2 non viene promossa.** La regola non è stata modificata dopo aver visto i
risultati: un recall migliore non compensa il peggioramento di accuracy richiesto dal protocollo
comune. La versione attiva resta v1.

### Probe locale (istanza di prova, `artifacts/probe_results.json`)

Payload valido `req_01`, 3 warm-up + 30 misure, verifica del fingerprint dell'artefatto misurato:
v1 p95 = 12,25 ms (p50 8,55), v2 p95 = 19,6 ms (p50 14,82), error rate 0 su entrambe.
Ambiente: Python 3.14.3, scikit-learn 1.9.1, httpx 0.28.1, 8 CPU, Windows 11.

### Drill di attivazione e ripristino (`artifacts/drill_results.json`)

Gate FAIL → il cambio versione è un **drill locale dichiarato**, ma eseguito realmente sul
servizio (la versione attiva cambia davvero e l'artefatto viene ricaricato da disco).

| # | richiesta | versione chiesta | versione servita | feature | status | atteso | predizione |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | req_01 | v1 | v1 | 30 | 200 | 200 | 0 |
| 2 | req_02 | v2 | v2 | 30 | 200 | 200 | 0 |
| 3 | req_03 | v1 (dopo il ripristino) | v1 | 30 | 200 | 200 | 1 |
| 4 | req_04 | v1 | — (respinta) | 29 | 422 | 422 | — |

**error rate del drill = 1/4 = 0,25**, coerente con il valore atteso della consegna. Le misure
della probe sono tenute separate da quelle del drill.

## 6. Stato finale

- versione attiva: **v1** (baseline ripristinata), run `run_…`, fingerprint
  `a7c2627d2f5a…` = fingerprint dell'artefatto della baseline (`coincide_con_baseline_v1: true`).
- registro delle transizioni in `artifacts/rollback_log.json`: `v1 → v2`, `v2 → v1`.
- il servizio è fermato al termine della pipeline (`cleanup`).

## 7. File principali e a cosa servono

| File | Ruolo |
| --- | --- |
| `config.json` | percorsi, seed 1308, soglie del gate, parametri della probe |
| `dataset.csv`, `richieste_monitoraggio.csv` | input assegnati dal docente (non modificati) |
| `README.md` (questo file) | scheda del soggetto distribuita + relazione del progetto |
| `requirements.txt` / `requirements-mlflow.txt` | dipendenze obbligatorie / opzionale MLflow |
| `src/common.py` | contratto condiviso: lettura e validazione CSV, ordine feature, sha256, carica/salva artefatti |
| `src/train.py` | S2/S4: addestra v1 e v2, valuta sul validation, salva artefatti + metadati + registry, registra le run MLflow |
| `src/app.py` | S3: servizio FastAPI (`/predict/{version}`, `/health`, `/metrics`, `/admin/activate/{version}`) con log JSON |
| `src/checks.py` | S1–S3: controlli del contratto, rifiuto degli artefatti non coerenti, client in-process |
| `src/probe.py` | S4: probe di latenza/error rate per versione |
| `src/gate.py` | S4: tabella del gate (misura, soglia, esito, motivazione); non cambia la versione attiva |
| `src/monitor.py` | S5: drill v1 → v2 → v1 con le 4 richieste assegnate |
| `src/run_all.py` | orchestrazione completa end-to-end |
| `avvia_tutto.bat` | crea l'ambiente (se serve) ed esegue tutta la pipeline |
| `avvia_api.bat` | avvia solo il servizio per la demo interattiva |
| `artifacts/`, `logs/`, `mlflow.db` | evidenze generate (modelli, metadati, report, log, run MLflow) |

## 8. Come si avvia

**Modo semplice (consigliato)** — doppio clic su `avvia_tutto.bat`: prepara `.venv` al primo
avvio, installa le dipendenze, esegue tutta la pipeline e ferma il servizio alla fine.

**Riga di comando**

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\run_all.py
```

**Solo servizio (per la demo)**

```bat
avvia_api.bat
```

poi, da un altro terminale:

```bat
curl -X POST http://127.0.0.1:8000/predict/v1 -H "Content-Type: application/json" ^
     -d "{\"features\":[139,40,6,84.99,6,14,9.761,-26.545,22.189,55,18.114,123,6,3,26.096,90.963,3,68.705,157,3,61.269,33,7,79.544,10,36.729,46.74,1,16.286,98]}"
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

Documentazione interattiva dell'API: <http://127.0.0.1:8000/docs>

## 9. Che cosa viene eseguito all'avvio e come sono collegati i componenti

`run_all.py` esegue in sequenza, come un unico processo orchestrato:

1. **`train.py`** legge `dataset.csv` con il contratto di `common.py`, fitta v1 sul train, salva
   `artifacts/v1/model.pkl` + `model_metadata.json`, registra la run in MLflow; poi **carica la v1
   già salvata** (verificando fingerprint e ID di validation), fitta v2 sul medesimo train e salva
   `artifacts/v2/`; infine scrive `artifacts/registry.json` con `active = v1`.
2. **`checks.py`** apre l'app FastAPI con un client in-process e verifica contratto, rifiuto degli
   artefatti corrotti, 200 sulla richiesta valida e 422 sull'input incompleto.
3. **`app.py`** viene avviato come servizio uvicorn. All'avvio legge `registry.json` e carica
   l'artefatto della versione attiva verificandone il fingerprint; ogni richiesta produce una riga
   JSON in `logs/serving.log`.
4. **`probe.py`** misura p95 ed error rate di v1 e v2 sullo stesso payload valido.
5. **`gate.py`** incrocia le metriche del validation con le misure della probe e produce la
   decisione (non tocca la versione attiva).
6. **`monitor.py`** esegue il drill: chiama `/admin/activate/v2`, serve `req_02` con v2, richiama
   `/admin/activate/v1` (ripristino della baseline), serve `req_03`, poi invia `req_04` incompleta
   (422). Confronta lo stato finale con la baseline e scrive `rollback_log.json`.
7. Alla fine il servizio viene terminato e viene stampato il riepilogo.

**Come si recupera la versione precedente:** `POST /admin/activate/{version}` ricarica l'artefatto
da disco verificando il fingerprint, aggiorna `active`/`previous` nel registry e registra la
transizione; la risposta e `rollback_log.json` contengono stato prima/dopo, run ID e fingerprint.

## 10. Limiti e cleanup

- La validation del laboratorio **non è un test temporale futuro**: serve al confronto didattico,
  non a stimare le prestazioni in produzione.
- La probe gira sullo stesso host del servizio: non rappresenta il carico reale.
- I dati sono sintetici (seed 1308): i numeri non misurano un sistema reale.
- Container Docker: **non eseguito** in questa consegna (solo in-process + avvio HTTP locale).
- La candidata **non è stata promossa**: non sono garantite accuracy superiori né attivazioni.
- Cleanup: `run_all.py` ferma il servizio in un blocco `finally`. Se si usa `avvia_api.bat`,
  fermare con `Ctrl+C`. Eventuali `logs/` e `artifacts/` sono rieseguibili: cancellarli e
  rilanciare `run_all.py` rigenera tutto da `dataset.csv`.

## 11. Contributi e revisioni

Divisione dei ruoli prevista dal lab (da completare con i nomi del gruppo):

| Ruolo | Attività | Revisione incrociata |
| --- | --- | --- |
| Dati e training | contratto, v1, v2, metriche, MLflow | rilettura del contratto e delle metriche |
| Servizio e telemetria | API, validazione, log, health/metrics | prova del caso valido e del caso 422 |
| Aggiornamento e documentazione | gate, transizioni, drill, README | verifica del ripristino e dello stato finale |

Tutti i componenti devono saper ricostruire una predizione fino a modello, run e dati usando
`request_id` → `logs/serving.log` → `run_id` → `artifacts/<versione>/model_metadata.json`.
