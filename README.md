# PCG Heart Sound Pipelines

Pipeline end-to-end per classificazione binaria di fonocardiogrammi (PCG):
`healthy` vs `unhealthy`.

Il progetto copre tutto il flusso:
1. preprocessing audio configurabile da JSON,
2. conversione in log-mel spectrogram 128x128 (grayscale),
3. training/evaluation di una CNN (ResNet18),
4. report metrici e visuali.

## A cosa serve la repo
Questa repo serve per confrontare diverse pipeline di preprocessing sullo stesso dataset,
misurarne l'impatto sulle metriche di classificazione e produrre artefatti riproducibili
(checkpoint, confusion matrix, grafici metriche, sanity plot step-by-step).

## Struttura directory
| Percorso | Scopo |
|---|---|
| `src/` | Logica core: operazioni audio, trasformate tempo-frequenza, loader dataset, builder pipeline. |
| `configs/` | Configurazioni pipeline (`pipelines.json`). |
| `scripts/` | Script operativi per download dati, sanity check e preprocessing in batch. |
| `cnn/` | Script per training, valutazione su test e inferenza singola immagine. |
| `data/` | Dataset audio `.wav` in split `train/`, `val/`, opzionale `test/`. |
| `data_preprocessed/` | Dataset trasformato in immagini (`.png`) pronte per la CNN. |
| `runs/` | Output esperimenti: sanity plot, checkpoint modello, metriche e grafici. |
| `relazione_progetto_pcg.txt` | Relazione tecnica in formato testo. |
| `relazione_progetto_pcg.tex` | Relazione tecnica in LaTeX. |
| `requirements.txt` | Dipendenze Python del progetto. |

## Prerequisiti
- Python 3.10+ (consigliato)
- `pip`
- Opzionale: Kaggle CLI (`kaggle`) se vuoi scaricare il dataset via script

## Installazione ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Formato dataset atteso
Struttura consigliata:
```text
data/
  train/
    healthy/*.wav
    unhealthy/*.wav
  val/
    healthy/*.wav
    unhealthy/*.wav
  test/                # opzionale ma consigliato per evaluate_testset.py
    healthy/*.wav
    unhealthy/*.wav
```

## Pipeline disponibili
Definite in `configs/pipelines.json`:
- `trim`
- `bandpass`
- `bandpass_norm_rms`

Nota: alcuni script hanno default legacy (`bandpass_mel`), quindi è consigliato passare
sempre `--pipeline` esplicitamente.

## Workflow consigliato (quick start)

### 1) (Opzionale) Download dataset da Kaggle
```bash
python3 scripts/download_data.py --data_dir data --force
```
Richiede autenticazione Kaggle configurata nel sistema.

### 2) Sanity check rapido di una pipeline
```bash
python3 scripts/sanity_check.py \
  --data_dir data \
  --cfg configs/pipelines.json \
  --pipeline bandpass_norm_rms \
  --out_png runs/sanity.png
```

Per debug dettagliato step-by-step (waveform/FFT/spectrogrammi):
```bash
python3 scripts/sanity_check_vis.py \
  --data_dir data \
  --cfg configs/pipelines.json \
  --pipeline bandpass_norm_rms \
  --out_dir runs/sanity
```

### 3) Preprocessing batch da WAV a PNG
```bash
python3 scripts/data_preprocessing.py \
  --data_dir data \
  --cfg configs/pipelines.json \
  --pipeline bandpass_norm_rms \
  --out_dir data_preprocessed
```

Lo script crea una cartella timestampata, ad esempio:
`data_preprocessed/pipeline_bandpass_norm_rms_YYYYMMDD_HHMM/`.

Recupera l'ultima cartella generata:
```bash
PREP_DIR=$(ls -dt data_preprocessed/pipeline_bandpass_norm_rms_* | head -n1)
echo "$PREP_DIR"
```

### 4) Training CNN
```bash
python3 cnn/train.py \
  --data_dir "$PREP_DIR" \
  --out_dir runs/cnn_checkpoint \
  --epochs 10 \
  --batch_size 32 \
  --pretrained
```

### 5) Valutazione su test set
```bash
RUN_DIR=$(ls -dt runs/cnn_checkpoint/* | head -n1)
python3 cnn/evaluate_testset.py \
  --model_path "$RUN_DIR/model_best.pt" \
  --test_dir "$PREP_DIR/test" \
  --out_dir runs/metrics
```

### 6) Inferenza su una singola immagine
```bash
python3 cnn/infer.py \
  --model_path "$RUN_DIR/model_best.pt" \
  --input_path "$PREP_DIR/test/healthy/EXAMPLE.png"
```

## Script principali (cosa fanno)

### `scripts/`
- `download_data.py`: scarica dataset Kaggle e prepara `data/train`, `data/val`.
- `sanity_check.py`: verifica veloce output pipeline su 1 healthy + 1 unhealthy.
- `sanity_check_vis.py`: visualizzazione completa dei passaggi intermedi della pipeline.
- `data_preprocessing.py`: applica pipeline a tutto il dataset e salva PNG.
- `create_train_val_test_split.py`: utility per creare split da directory sorgente.

### `cnn/`
- `train.py`: training ResNet18 su immagini preprocessate (`train/` e `val/`).
- `evaluate_testset.py`: metriche complete su test (accuracy, precision/recall/F1,
  confusion matrix, plot globali e per-classe).
- `infer.py`: predizione su singola immagine con probabilità per classe.

## Output principali
- `runs/sanity/`: plot diagnostici step-by-step e summary pipeline.
- `runs/cnn_checkpoint/`: checkpoint training (`model_best.pt`, `model_last.pt`, epoche).
- `runs/metrics/`: report valutazione (`metrics.json`, `stats.txt`, confusion matrix, plot).
- `data_preprocessed/`: dataset PNG prodotto dal preprocessing.

## Note operative
- In presenza di dataset sbilanciato, valuta il modello con metriche per-classe e F1 weighted,
  non solo accuracy.
- Assicurati che `test/` sia presente nel dataset preprocessato se vuoi usare
  `cnn/evaluate_testset.py`.
- Se cambi pipeline, rigenera i PNG prima di rilanciare il training.
