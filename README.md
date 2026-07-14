# Hybrid LSTM–CNN Model for the SIMON Cipher

A deep-learning **neural cryptanalysis** project that attacks the
[SIMON](https://en.wikipedia.org/wiki/Simon_(cipher)) lightweight block cipher
(the NSA Feistel family) using a **hybrid CNN + LSTM** neural network. The model
is trained to recover the **plaintext directly from the ciphertext** of
round-reduced SIMON, and the experiment is repeated round-by-round from 1 up to
the full round count to measure how the attack degrades as cipher strength
increases.

The code reproduces the methodology of a research paper that evaluates a neural
distinguisher/decryptor against three families of input text:

| Abbreviation | Meaning | Datasets | How the plaintext is produced |
|--------------|---------|----------|-------------------------------|
| **CTT**  | Correlated Text Test      | `Dataset-1` (incremental), `Dataset-2…4` (decremental) | 5 fixed bytes + 3 incrementing/decrementing bytes (highly correlated blocks) |
| **NCTT** | Non-Correlated Text Test  | `Dataset-2…4` (used as unseen test sets) | Model trained on `Dataset-1`, then evaluated on the decremental sets |
| **RWTT** | Real-World Text Test      | `Dataset-5…7`                          | 8-byte blocks extracted from real PDF documents (natural, non-uniform data) |

> This is the SIMON counterpart of the companion
> [Hybrid-LSTM-CNN-Model-Present-Cipher](https://github.com/hudailameenal/Hybrid-LSTM-CNN-Model-Present-Cipher)
> project; the pipeline is identical, only the target cipher changes.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [The SIMON Cipher Implementation](#the-simon-cipher-implementation)
- [The Hybrid Model](#the-hybrid-model)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage / Commands](#usage--commands)
- [Metrics](#metrics)
- [Results](#results)
- [Notes & Caveats](#notes--caveats)

---

## How It Works

The pipeline has four conceptual stages:

1. **Cipher** – A pure-Python implementation of SIMON (`cipher.py`) provides the
   encryption primitive used to build labelled data.
2. **Data generation** – `generate.py` builds seven CSV datasets of
   `plaintext,ciphertext` pairs (correlated, decremental, and PDF-derived) using
   the **SIMON 64/128** configuration.
3. **Round-by-round training** – The training scripts (`train_ctt.py`,
   `train_nctt.py`, `train_rwtt.py`) encrypt the data through an increasing
   number of SIMON rounds and train a hybrid CNN–LSTM to invert the mapping
   `ciphertext → plaintext` for **each** round (1 → 42). The ciphertext of round
   *n* is chained (fed forward) as the input to round *n+1*, matching the
   iterative Feistel structure of the cipher.
4. **Evaluation** – The `evaluate_*.py` scripts load the saved per-round metrics
   (and optionally the saved Keras models) and print them as tables.

The core research question: **how many rounds of SIMON can a neural network
"see through"?** As expected, restoration accuracy is highest at 1 round and
decays as rounds increase.

---

## Project Structure

```
Hybrid-LSTM-CNN-Model-SIMON-Cipher/
├── cipher.py             # SIMON block cipher (all block/key sizes + 6 modes)
├── generate.py           # Generates Dataset-1…7 CSVs (CTT / NCTT / RWTT)
├── preprocessing.py      # Loads CSVs → normalized byte matrices, train/test split
├── model.py              # Baseline LSTM model (reference architecture)
│
├── train_ctt.py          # Train hybrid model on Correlated Text (Dataset-1)
├── train_nctt.py         # Train / test on Non-Correlated text (Dataset-2…4)
├── train_rwtt.py         # Train on Real-World PDF-derived text (Dataset-5…7)
│
├── evaluate_ctt.py       # Pretty-print saved CTT results / model summary
├── evaluate_nctt.py      # Pretty-print saved NCTT results / model summary
├── evaluate_rwtt.py      # Pretty-print saved RWTT results / model summary
│
├── datasets/             # Generated Dataset-1…7.csv (plaintext,ciphertext)
├── src/data/             # Copy of the same CSVs
├── pdfs/                 # 1.pdf, 2.pdf, 3.pdf — source material for NCTT/RWTT
├── models/               # Saved .h5 models (simon_ctt_hybrid, nctt_*, rwtt_*)
└── results/              # Per-round metric CSVs
```

> `preprocessing.load_all_data()` reads directly from **`datasets/`** (dataset
> names `Dataset-1` … `Dataset-7`, keyed internally as `dataset1` … `dataset7`).
> A duplicate copy also lives under `src/data/`.

---

## The SIMON Cipher Implementation

`cipher.py` is a full-featured, standards-compliant SIMON implementation
(original author: *inmcm*). Highlights:

- **All standard configurations** – block sizes 32/48/64/96/128 bits with their
  valid key sizes (e.g. SIMON 32/64, 64/128, 128/256), selected via the
  `__valid_setups` table, which also fixes the round count and *z*-sequence.
- **Feistel round function** – circular shifts (`<<1`, `<<8`, `<<2`), an
  AND/XOR chain, and the round key, with matching `encrypt`/`decrypt`.
- **Key schedule** – precomputed round keys using the *z* constant sequences.
- **Six block-cipher modes** – ECB, CTR, CBC, PCBC, CFB, OFB.
- **This project uses SIMON 64/128** (64-bit block, 128-bit key) in ECB mode for
  data generation.

```python
from cipher import SimonCipher

# Official SIMON 32/64 test vector
c = SimonCipher(0x1918111009080100, key_size=64, block_size=32)
hex(c.encrypt(0x65656877))     # -> 0xc69be9bb

# Configuration used by generate.py
key = 0x1b1a1918131211100b0a090803020100
c = SimonCipher(key, key_size=128, block_size=64)
```

---

## The Hybrid Model

Each training script builds a **CNN → stacked-LSTM → Dense** regressor that maps
an 8-byte ciphertext block (reshaped to `(8, 1)`) to 8 normalized plaintext
bytes:

```
Input (8, 1)
 └─ Conv1D(32, k=3, relu, same) → MaxPooling1D(2) → Conv1D(16, k=2, relu, same)   # local features
     └─ LSTM(h, tanh, return_sequences) → Dropout
         └─ LSTM(h/2, sigmoid) → Dropout
             └─ Dense(8, linear)                                                  # 8 predicted plaintext bytes
```

- The **CTT** model (`train_ctt.py`) is the lightest variant (2 Conv + 2 LSTM).
- The **NCTT** model (`train_nctt.py`) has the same shape with dropout tuning.
- The **RWTT** model (`train_rwtt.py`) is the deepest (3 LSTM + 3 Dense blocks),
  since real-world PDF text is the hardest to attack.

Training uses **MAE loss** with **Adam/RMSprop** optimizers. Hyperparameters for
key rounds (1, 2, 3, 4, 20, …) come from the paper's tables; any other round
falls back to an **Optuna** search over hidden units, optimizer, learning rate,
epochs, and batch size. `model.py` holds a simpler **LSTM model**
baseline for reference.

---

## Datasets

Generated by `generate.py` as `plaintext,ciphertext` CSV pairs (16 hex chars
each = 8 bytes):

| Dataset | Type | Size | Source |
|---------|------|------|--------|
| `Dataset-1` | CTT – incremental | 2¹⁵ blocks | 5 random fixed bytes + 3 incrementing bytes |
| `Dataset-2…4` | CTT – decremental | 2¹¹ blocks each | 5 random fixed bytes + 3 decrementing bytes |
| `Dataset-5` | RWTT / NCTT | ~2¹⁶·³ blocks | 8-byte chunks of `pdfs/1.pdf` |
| `Dataset-6` | RWTT / NCTT | ~2¹⁴·⁶ blocks | 8-byte chunks of `pdfs/2.pdf` |
| `Dataset-7` | RWTT / NCTT | ~2¹⁵·³ blocks | 8-byte chunks of `pdfs/3.pdf` |

- CTT datasets use the **fixed SIMON test-vector key**
  (`0x1b1a1918131211100b0a090803020100`).
- PDF-based datasets use a **fresh random 128-bit key** each, and duplicate
  blocks are removed.

**Preprocessing** (`preprocessing.py`): hex → bytes → `uint8` matrix → normalize
to `[0,1]` → reshape ciphertext to `(samples, 8, 1)`. Split is 90/10 (sequential
for `Dataset-1`, random shuffled for the rest).

---

## Installation

Requires **Python 3.8+**. There is no `requirements.txt`; install the
dependencies directly:

```bash
pip install tensorflow numpy scikit-learn optuna
```

Then clone the repo:

```bash
git clone https://github.com/hudailameenal/Hybrid-LSTM-CNN-Model-SIMON-Cipher.git
cd Hybrid-LSTM-CNN-Model-SIMON-Cipher
```

The repository already ships the generated datasets and pre-trained `.h5`
models, so you can jump straight to evaluation.

---

## Usage / Commands

### 1. Verify the cipher
```bash
python cipher.py            # runs the SIMON 32/64 test vector -> 0xc69be9bb
```

### 2. (Optional) Regenerate the datasets
```bash
python generate.py          # writes datasets/Dataset-1.csv … Dataset-7.csv
```

### 3. Sanity-check preprocessing
```bash
python preprocessing.py     # prints train/test shapes for Dataset-1…7
```

### 4. Inspect the baseline model
```bash
python model.py             # prints the LSTM model summary
```

### 5. Train

Each script trains one model per round (1 → 42), saves the final `.h5` model(s)
to `models/`, and writes per-round metrics to `results/`.

```bash
python train_ctt.py         # Correlated Text  -> models/simon_ctt_hybrid.h5
python train_nctt.py        # Non-Correlated   -> models/nctt_dataset-*_hybrid_final.h5
python train_rwtt.py        # Real-World PDF   -> models/rwtt_dataset-*_hybrid_final.h5
```

> The training scripts wrap the pipeline in a `try/except`; the initial guarded
> call (`import model`/`utils`/`lib` helpers) is expected to fail, and the real
> round-by-round loop runs in the `except` branch. Expect long runtimes on CPU —
> training ~42 models per text type is compute-heavy.

### 6. Evaluate / view results
```bash
python evaluate_ctt.py      # tabulates results/ctt_hybrid_round_results.csv
python evaluate_nctt.py     # tabulates results/nctt_hybrid_round_results.csv
python evaluate_rwtt.py     # tabulates results/rwtt_hybrid_round_results.csv
```

---

## Metrics

All three pipelines report the same core metrics (bytes are de-normalized back
to `0–255` before scoring):

- **Restoration / Test Accuracy (%)** – fraction of the 8 output bytes predicted
  exactly (RWTT reports the closely-related **bitwise** accuracy).
- **Mean Byte Error (MBE)** – average absolute per-byte error, normalized to
  `[0,1]`.
- **Hamming Accuracy (%)** – `1 − (differing bits / total bits)`; a bit-level
  score of how close the prediction is.

---

## Results

Per-round metrics are stored in `results/`. Accuracy is highest at low round
counts and steadily decays as rounds increase:

**CTT (`Dataset-1`)** — restoration accuracy `82.1%` (round 1) → `74.3%`
(round 42).

**NCTT (tested on `Dataset-2/3/4`)** — round-1 accuracy `~69–72%`, decaying to
`~62–64%` by round 42.

**RWTT (`Dataset-5/6/7`)** — bitwise test accuracy `~67%` (round 1) → `~60%` in
later rounds, the hardest case since PDF text is closest to real-world data.

See the CSVs for the complete round-by-round tables:
`results/ctt_hybrid_round_results.csv`,
`results/nctt_hybrid_round_results.csv`,
`results/rwtt_hybrid_round_results.csv`.

---

## Notes & Caveats

- **Cipher config vs. round loop** — `generate.py` encrypts with SIMON 64/128
  (44 full rounds), while the training loops iterate rounds `1 → 42` and some
  in-code comments reference SIMON 64/96 (42 rounds). The attack is on a
  *round-reduced* cipher regardless; treat the exact round ceiling as an
  experimental setting rather than the cipher's spec.
- **`import model` / `utils` / `lib` in the train scripts** — the top of each
  training script tries to import an external helper and call a function like
  `model.train_for_all_rounds()`. That call is expected to fail, and the real
  training loop runs in the `except` block. This is by design in the current
  code, though a little unconventional.
- **Educational / research use only** — this is a study of neural cryptanalysis
  against *round-reduced* SIMON. It does not break full SIMON and is not a
  practical attack on deployed systems.
- **No pinned dependencies** — TensorFlow 2.x, NumPy, scikit-learn, and Optuna
  are required; exact versions are not pinned.
- **Cipher attribution** — `cipher.py` is based on a public SIMON/SPECK
  implementation by *inmcm*.


  The *LSTM-only* model this hybrid is compared against:
[LSTM-Model-SIMON-Cipher](https://github.com/hudailameenal/LSTM-Model-SIMON-Cipher.git).
It runs the same SIMON plaintext-recovery task without the CNN layers, so the accuracy gap shows what the CNN feature-extraction adds.