# PCG Heart Sound Pipelines Benchmark (Healthy vs Unhealthy)

Obiettivo: classificare registrazioni di suoni cardiaci (**PCG**) in due classi:
- **healthy** (0)
- **unhealthy** (1)

Focus del progetto: **confrontare più pipeline di preprocessing** (prima della CNN) e motivare
le differenze osservate tramite analisi del segnale (FFT/STFT/Mel) e metriche di classificazione.

---

## Dataset

Dataset Kaggle: **Heart Sound Database** (swapnilpanda/heart-sound-database).  
Struttura attesa dopo il download:
data/
    train/
        healthy/.wav
        unhealthy/.wav
    val/
        healthy/.wav
        unhealthy/.wav

> Nota: il train risulta sbilanciato (molti più healthy che unhealthy).  
> Questo va tenuto in considerazione nel training (metriche come F1 e uso di class weights).

---

## Pipeline implementate finora

Le pipeline sono definite in `configs/pipelines.json` come sequenza di step.

### 1) `bandpass_mel` (pipeline “base”)
- Load / Resample / Mono
- **Band-pass filter** (es. 20–800 Hz)
- Normalize RMS (volume)
- Chunk/Pad (durata fissa)
- **Mel-spectrogram + log**

Motivazione: ridurre rumore fuori banda e fornire alla CNN una rappresentazione stabile e standard.

### 2) `fft_denoise_mel` (denoising in frequenza)
- Denoising basato su STFT/FFT (spectral gating)
- Normalize RMS
- Chunk/Pad
- Mel-spectrogram + log

Motivazione: denoising adattivo (dipende dal file), utile in presenza di rumore variabile.

### 3) `wavelet_denoise_mel` (wavelets)
- Wavelet denoise (thresholding)
- Normalize RMS
- Chunk/Pad
- Mel-spectrogram + log

Motivazione: denoising multi-scala per segnali non stazionari (utile per transienti).

---

## Cosa è stato fatto fino ad ora (stato progetto)

### 1) Download dataset + setup struttura cartelle
Script: `scripts/download_data.py`  
Scarica il dataset da Kaggle e lo organizza in `data/train` e `data/val`.

### 2) Sanity check: pipeline produce spettrogrammi corretti
Script: `scripts/sanity_check.py`  
- Legge train/val
- Applica la pipeline scelta a un sample healthy e uno unhealthy
- Salva un PNG (`runs/sanity.png`) con i due Mel-spectrogrammi

### 3) Audit del preprocessing (controllo numerico + visivo)
Script: `scripts/audit_preprocessing.py`  
Per ciascuna pipeline e per un piccolo numero di file per classe:
- controlla durata finale (`chunk_len`)
- controlla assenza di NaN/Inf
- calcola statistiche (RMS e peak)
- salva figure: waveform raw, FFT raw, waveform/process, FFT/process, Mel finale

Output:
- `runs/audit/<pipeline>/sample_*.png`
- `runs/audit/<pipeline>/stats.json`
- `runs/audit/summary.json`

### Risultato importante emerso dall’audit
- La pipeline `bandpass_mel` può aumentare i picchi (peak) a causa di effetti noti del filtro (filtfilt/overshoot).
- Le pipeline `fft_denoise_mel` e `wavelet_denoise_mel` risultano più “stabili” sui peak.

Questo porta a un prossimo intervento: **inserire un limiter/soft-clip dopo il band-pass** per rendere la pipeline più robusta.

### 4) Data preprocessing (generazione dataset preprocessato)
Script: `scripts/data_preprocessing.py`  

Input:
- cartella dataset con struttura `data/train|val/healthy|unhealthy/*.wav`
- pipeline definita in `configs/pipelines.json`

Azioni:
- applica la pipeline scelta a **tutti** i file audio (train e val)
- produce un Mel-spectrogramma per ogni file
- salva i risultati in formato PNG
- scrive un file `stats.txt` con data/ora del run e conteggi per split/classe

Output:
- cartella `data_preprocessed/pipeline_<pipeline_name>/train|val/healthy|unhealthy/*.png`
- file `data_preprocessed/pipeline_<pipeline_name>/stats.txt`

---

## Installazione (ambiente Python)
usare virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
