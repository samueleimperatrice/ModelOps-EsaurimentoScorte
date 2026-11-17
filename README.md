# Progetto 8 - Esaurimento delle scorte di magazzino

## Domanda e momento della previsione

**Domanda:** Quali coppie articolo–magazzino potrebbero esaurire le scorte nei prossimi 7 giorni?

**Momento di osservazione:** Inizio della giornata di pianificazione. Vendite e affidabilità descrivono lo storico; domanda e arrivi futuri sono stime disponibili in quel momento, non esiti già osservati.

- **target = 1:** Scorte esaurite almeno una volta nei 7 giorni successivi.
- **target = 0:** Nessun esaurimento delle scorte nei 7 giorni successivi.

I dati sono **interamente sintetici**, creati per il corso. Non rappresentano ticket, utenti, aziende, articoli o magazzini reali. Gli intervalli sono scelte della simulazione didattica; i risultati non misurano prestazioni su un sistema reale.

## File da usare insieme

| File | Contenuto |
| --- | --- |
| [dataset.csv](dataset.csv) | 1000 esempi, 30 feature numeriche, record_id, split e target 0/1 |
| [richieste_monitoraggio.csv](richieste_monitoraggio.csv) | Tre richieste valide v1 → v2 → v1 e una richiesta incompleta con 29 valori → 422 |

Segui il [Lab 13](../../13_project_work_pipeline_serving_monitoring_e_aggiornamento.md) usando i CSV di questa cartella. Il gruppo costruisce il proprio codice e conserva schema, ordine feature e metadati coerenti in tutte le fasi.

## Split e classi

| Split | target 0 | target 1 | Totale |
| --- | ---: | ---: | ---: |
| train | 490 | 310 | 800 |
| validation | 122 | 78 | 200 |

Lo split stratificato è già assegnato: usalo per v1 e v2. record_id, split e target non sono feature. Non adattare trasformazioni sul validation e non aggiungere i payload al training.

## Dizionario delle 30 feature

Tutte le feature sono disponibili al momento della previsione descritto sopra. Le percentuali di quota sono tra 0 e 100; le variazioni possono essere negative quando previsto nella tabella. I punteggi sono indici sintetici. Le quantità discrete sono intere; le altre misure mantengono decimali.

**L'ordine della tabella è l'ordine della lista features dell'API.** I payload distribuiti seguono lo stesso ordine senza includere il target.

| Posizione API | Colonna | Unità / scala | Intervallo della simulazione |
| --- | --- | --- | --- |
| 1 | `scorta_disponibile_unita` | unità | 5–600 |
| 2 | `scorta_riservata_unita` | unità | 0–200 |
| 3 | `copertura_stimata_giorni` | giorni | 1–35 |
| 4 | `quota_scorta_utilizzabile_pct` | % | 30–100 |
| 5 | `rotture_scorta_storiche_90g` | conteggio | 0–15 |
| 6 | `intervallo_riordino_pianificato_giorni` | giorni | 1–30 |
| 7 | `vendite_medie_giornaliere_30g` | unità/giorno | 1–80 |
| 8 | `crescita_domanda_7g_pct` | % | -40–120 |
| 9 | `variabilita_domanda_pct` | % | 0–100 |
| 10 | `ordini_clienti_aperti_unita` | unità | 0–300 |
| 11 | `quota_vendite_promozionali_storica_pct` | % | 0–100 |
| 12 | `previsione_domanda_7g_unita` | unità | 10–650 |
| 13 | `consegna_fornitore_media_giorni` | giorni | 1–30 |
| 14 | `variabilita_consegna_fornitore_giorni` | giorni | 0–15 |
| 15 | `ritardi_fornitore_90g_pct` | % | 0–80 |
| 16 | `quota_consegne_complete_90g_pct` | % | 20–100 |
| 17 | `fornitori_alternativi_disponibili` | conteggio | 0–10 |
| 18 | `affidabilita_fornitore_indice` | punteggio 0–100 | 0–100 |
| 19 | `quantita_in_arrivo_unita` | unità | 0–500 |
| 20 | `giorni_al_prossimo_arrivo_previsto` | giorni | 0–30 |
| 21 | `quota_ordini_acquisto_confermati_pct` | % | 0–100 |
| 22 | `margine_scorta_sicurezza_unita` | unità | -100–300 |
| 23 | `giorni_da_ultimo_riordino` | giorni | 0–35 |
| 24 | `capacita_ricevimento_disponibile_pct` | % | 0–100 |
| 25 | `giorni_a_prossima_promozione` | giorni | 0–60 |
| 26 | `incremento_domanda_promozione_stimato_pct` | % | 0–100 |
| 27 | `indice_stagionalita_domanda` | punteggio 0–100 | 0–100 |
| 28 | `articoli_sostitutivi_disponibili` | conteggio | 0–12 |
| 29 | `quota_richieste_urgenti_30g_pct` | % | 0–100 |
| 30 | `lotto_minimo_riordino_unita` | unità | 1–300 |

## Lavoro del gruppo

1. Definire l'uso della previsione e il costo dei due tipi di errore. Confrontare recall degli esaurimenti e precision delle segnalazioni di riordino. Discutere il compromesso tra vendite perse e costi di scorte o ordini non necessari.
2. Seguire S1–S6 del Lab 13: contratto, baseline tracciata, API, osservazione, candidata, gate, drill e presentazione.
3. Conservare il gate comune (accuracy v2 almeno pari a v1 e vincoli operativi definiti nel lab); aggiungere precision, recall e matrice di confusione. Non cambiare la regola dopo aver visto i risultati.
4. Dimostrare il ripristino con le versioni realmente servite. Sulle quattro richieste assegnate, una 422 su quattro corrisponde a error rate 0,25.
5. Consegnare misure effettive, versioni, configurazioni, limiti e contributi individuali; nessuna accuracy o promozione di v2 è garantita.

## Provenienza e riproducibilità

Generazione sintetica del docente, seed 1308. Le feature hanno correlazioni per gruppi e il target dipende da un processo probabilistico con rumore ed effetti non lineari. Nessuna feature contiene l'esito futuro osservato. I CSV distribuiti sono gli input assegnati: il codice del gruppo deve leggerli, senza rigenerare dati o split.

La validation del laboratorio non simula un test temporale futuro. Per il calendario e la scadenza di presentazione segui la guida Lab 13 e il catalogo dei progetti.

---

## Avviare il progetto da Windows

### Requisiti

- Windows 10 o successivo.
- Python installato e disponibile dal terminale con il comando `python`.
- Connessione Internet al primo avvio, per installare le dipendenze.

Per controllare che Python sia disponibile, apri PowerShell e digita:

```powershell
python --version
```

### Sequenza consigliata: eseguire tutto dal terminale

Apri **Windows PowerShell**. Digita ciascun comando nell'ordine indicato e premi `Invio` dopo ognuno.

**1. Entra nella cartella del progetto**

```powershell
cd "C:\Users\matteo.guagliardo\Desktop\progetto modelops"
```

**2. Controlla che Python sia disponibile**

```powershell
python --version
```

Il terminale deve mostrare una versione di Python. Se dice che `python` non e' riconosciuto, installa Python e abilita l'opzione per aggiungerlo al `PATH`, poi riapri PowerShell.

**3. Avvia l'intero progetto**

```powershell
.\avvia_tutto.bat
```

Al primo avvio il file prepara automaticamente l'ambiente `.venv` e installa i pacchetti necessari; per questo serve Internet. Poi avvia training, controlli, API, probe, gate e drill. Attendi che compaia `[COMPLETATO] Pipeline terminata correttamente`; se il terminale chiede di premere un tasto, premilo per chiudere la finestra. Ai successivi avvii basta ripetere i passaggi 1 e 3. Non occorre avviare `avvia_api.bat` durante questa procedura.

### Alternativa: eseguire i comandi di preparazione manualmente

Questa procedura e' un'alternativa a `avvia_tutto.bat`. Usala se vuoi installare le dipendenze e lanciare direttamente lo script Python. Esegui i comandi in ordine, uno alla volta:

```powershell
cd "C:\Users\matteo.guagliardo\Desktop\progetto modelops"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-mlflow.txt
.\.venv\Scripts\python.exe src\run_all.py
```

L'installazione di MLflow e' facoltativa: se vuoi saltarla, ometti il comando relativo a `requirements-mlflow.txt`. Se l'ambiente `.venv` e' gia' stato creato e i pacchetti sono installati, per eseguire di nuovo l'intero esercizio usa invece solo:

```powershell
.\.venv\Scripts\python.exe src\run_all.py
```

Scegli una sola procedura: quella automatica con `avvia_tutto.bat` oppure quella manuale.

## Fasi eseguite

Durante l'avvio completo, il terminale mostra intestazioni numerate. In breve:

1. **Addestramento dei modelli** (`train.py`): controlla il dataset, addestra la baseline v1 e la candidata v2, valuta le prestazioni e salva modelli e metadati in `artifacts/`.
2. **Controlli automatici** (`checks.py`): verifica il contratto dei dati e degli artefatti, e prova richieste API valide e non valide.
3. **Avvio e controllo del servizio API** (`app.py`): avvia temporaneamente FastAPI e controlla che risponda all'indirizzo locale.
4. **Probe** (`probe.py`): invia richieste di prova e misura tempi di risposta ed errori per v1 e v2.
5. **Gate** (`gate.py`): confronta v2 con v1 secondo le soglie definite in `config.json` e decide se la candidata puo' essere promossa. Il gate non cambia da solo la versione attiva.
6. **Drill di monitoraggio** (`monitor.py`): esegue le richieste assegnate, simula il passaggio v1 -> v2 -> v1 e verifica il ripristino.
7. **Verifica finale** (`run_all.py`): mostra stato, metriche e risultati salvati.
8. **Arresto del servizio**: chiude il server avviato per la pipeline.

`run_all.py` coordina gli script nell'ordine corretto. Il servizio carica i modelli preparati dal training; probe, gate e monitor comunicano con il servizio tramite HTTP.

## Risultati da interpretare

- **14/14 controlli OK** indica che i controlli automatici sono stati superati.
- **Gate FAIL** e' il risultato previsto con i dati attuali: v2 migliora il recall della classe di esaurimento, ma peggiora l'accuracy rispetto a v1. Per questo v1 resta attiva.
- **Drill error rate 0.25** e' previsto: una delle quattro richieste ha 29 feature invece delle 30 richieste e viene correttamente respinta con HTTP 422.
- `StarletteDeprecationWarning` e' un avviso di compatibilita' futura, non un errore se i controlli terminano con esito OK.

## Avviare solo l'API

Per avviare l'API senza rieseguire training, probe e drill, assicurati prima di aver eseguito almeno una volta `avvia_tutto.bat`. Questo prepara l'ambiente Python e crea gli artefatti dei modelli.

### Metodo semplice da PowerShell

Apri PowerShell e inserisci i comandi seguenti, uno alla volta:

```powershell
cd "C:\Users\matteo.guagliardo\Desktop\progetto modelops"
.\avvia_api.bat
```

Lascia aperto il terminale: il server rimane in esecuzione finché non premi `Ctrl+C`.

### Avvio diretto con Uvicorn

In alternativa a `avvia_api.bat`, puoi avviare la stessa API direttamente con Python:

```powershell
cd "C:\Users\matteo.guagliardo\Desktop\progetto modelops"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --app-dir src
```

Usa il metodo semplice oppure quello diretto, non entrambi. Con il server attivo, apri nel browser uno di questi indirizzi:

- [http://localhost:8000](http://localhost:8000) oppure [http://127.0.0.1:8000](http://127.0.0.1:8000): indirizzo locale dell'API; la pagina iniziale reindirizza alla documentazione;
- [http://localhost:8000/docs](http://localhost:8000/docs): documentazione interattiva per provare gli endpoint;
- [http://localhost:8000/health](http://localhost:8000/health): stato del servizio e versione attiva.

Se il browser dice che il servizio non e' disponibile, controlla che il terminale con Uvicorn sia ancora aperto e che mostri `Uvicorn running on http://127.0.0.1:8000`. Se la porta `8000` risulta gia' occupata, potrebbe esserci gia' un'istanza attiva: usa quella senza avviarne un'altra. L'API ascolta solo sul computer locale.

## File e risultati

- `avvia_tutto.bat`: prepara l'ambiente, se necessario, ed esegue l'intera pipeline.
- `avvia_api.bat`: avvia solo il servizio API.
- `config.json`: contiene percorsi, parametri del servizio e soglie del gate.
- `src/`: contiene training, controlli, API, probe, gate, monitor e orchestratore.
- `artifacts/`: contiene modelli, registry e report prodotti dalla pipeline.
- `logs/`: contiene i log del servizio.
- `mlflow.db`: database locale usato per tracciare gli esperimenti quando MLflow e' installato.
