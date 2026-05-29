# Language Modeling on Project Gutenberg

Word-level neural language models trained on 18 classic public-domain books. Four architectures compared — Dense NN, LSTM, GRU, and Bidirectional LSTM — evaluated using perplexity on a chronological train/val/test split.

> **Course:** CSC Machine Learning — Semester Project  
> **Author:** Adil Hussain  
> **Dataset:** Project Gutenberg via NLTK (auto-downloads, no manual setup needed)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Project Structure](#project-structure)

---

## Overview

A language model predicts the next word given a sequence of previous words. This project trains and compares four neural network architectures on 2.6 million tokens of 19th-century literary text, then evaluates them using **perplexity** — the standard metric for language models.

**Key findings:**
- GRU achieves the best validation perplexity: **503.9**
- All neural models beat the unigram baseline (852.6) — 1.69x improvement for GRU
- All models beat the random baseline (10,000) — 19.8x improvement for GRU
- Bidirectional context (BiLSTM) does not help for next-word prediction, confirming the task is inherently left-to-right
- Dense NN overfits fastest and performs worst, confirming sequential inductive bias matters for language

---

## Dataset

**Source:** NLTK Gutenberg corpus — 18 public domain books, auto-downloaded at runtime.

| Stat | Value |
|---|---|
| Total books | 18 |
| Raw tokens | 2,621,613 |
| Unique raw tokens | 51,156 |
| Clean tokens (after preprocessing) | 2,135,400 |
| Vocabulary size used | 10,000 |
| Corpus coverage at 10K vocab | 96.21% |
| OOV rate | 3.79% |

**Books included:**

| Genre | Books |
|---|---|
| Novel | Moby Dick, Emma, Sense and Sensibility, Persuasion |
| Shakespeare | Hamlet, Macbeth, Julius Caesar |
| Bible | King James Bible (1,010,654 tokens — 38% of corpus) |
| Poetry | Walt Whitman, William Blake |
| Other prose | G.K. Chesterton, Lewis Carroll, Maria Edgeworth |

Word frequency follows **Zipf's Law** — confirmed by a log-log straight-line frequency plot. The top 10 words are all function words: *the, and, of, to, a, in, i, that, he, it*.

---

## Preprocessing

Three steps applied sequentially:

### 1. Lowercase normalization
All tokens converted to lowercase so `The` and `the` map to the same vocabulary entry.

### 2. Punctuation removal
Only tokens passing `word.isalpha()` are kept. This removed **486,213 tokens (18.5%)**, leaving **2,135,400 clean tokens**.

### 3. Vocabulary cutoff
Top 10,000 words kept; all others replaced with `<UNK>`. Coverage curve confirms diminishing returns beyond 10,000.

### Sequence construction
- **Context window:** 20 words
- **Stride:** 3 (sliding window moves 3 positions each step)
- **Total sequences:** 711,794
- **Input shape:** (711794, 20), each value is a word ID in [0, 9999]

### Train / Validation / Test Split

| Split | Sequences | % |
|---|---|---|
| Train | 569,435 | 80% |
| Validation | 71,179 | 10% |
| Test | 71,180 | 10% |

Split is **chronological** (not random) to prevent leakage of sequential context across splits.

---

## Models

All models: `Input(20) → Embedding(10000, 64) → [architecture] → Dropout(0.3) → Dense(10000, softmax)`

Loss: Sparse categorical cross-entropy | Optimizer: Adam

### Model 1 — Dense Neural Network
```
Embedding(10000, 64) → Flatten → Dense(256, ReLU) → Dropout(0.3)
→ Dense(128, ReLU) → Dropout(0.3) → Dense(10000, softmax)
```
No sequential processing. Treats all 20 context words as a flat feature vector.  
**Parameters: 2,290,832**

### Model 2 — LSTM
```
Embedding(10000, 64) → LSTM(128) → Dropout(0.3) → Dense(10000, softmax)
```
Three gates (forget, input, output) allow modeling long-range dependencies.  
**Parameters: 2,028,816**

### Model 3 — GRU ⭐ Best
```
Embedding(10000, 64) → GRU(128) → Dropout(0.3) → Dense(10000, softmax)
```
Simplified LSTM with two gates (reset, update). ~25% fewer recurrent params than LSTM.  
**Parameters: 2,004,496**

### Model 4 — Bidirectional LSTM
```
Embedding(10000, 64) → Bidirectional(LSTM(64)) → Dropout(0.3) → Dense(10000, softmax)
```
Forward + backward LSTM concatenated to 128-dim. Despite more complex design, has fewest parameters because each direction only needs 64 units.  
**Parameters: 1,996,048**

---

## Results

### Training setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Max epochs | 10 |
| Early stopping patience | 2 epochs |
| Restore best weights | Yes |
| Batch size (Dense NN) | 512 |
| Batch size (LSTM / GRU / BiLSTM) | 256 |
| Dropout rate | 0.3 |

### Validation Results

| Model | Parameters | Val Perplexity | Val Accuracy |
|---|---|---|---|
| Dense NN | 2,290,832 | 592.5 | 11.41% |
| LSTM | 2,028,816 | 553.4 | 12.05% |
| **GRU** | **2,004,496** | **503.9** | **12.30%** |
| BiLSTM | 1,996,048 | 534.0 | 12.30% |

### Baseline Comparison

| Baseline / Model | Perplexity | Factor vs GRU |
|---|---|---|
| Random (uniform over 10K words) | 10,000 | 19.8x worse |
| Unigram (word frequency only, no context) | 852.6 | 1.69x worse |
| Dense NN | 592.5 | 1.18x worse |
| LSTM | 553.4 | 1.10x worse |
| BiLSTM | 534.0 | 1.06x worse |
| **GRU (best)** | **503.9** | — |

### Final Test Evaluation — GRU

Test set was evaluated **once only** on the winning model:

| Split | Perplexity | Accuracy |
|---|---|---|
| Validation | 503.9 | 12.30% |
| **Test** | **751.4** | **12.01%** |

The test perplexity is higher than validation because the split is chronological — the test set is the final 10% of the corpus, furthest from the training distribution. This is expected behavior with chronological splits, not overfitting.

---

## Visualizations

The notebook includes 12+ visualizations:

| Part | Visualization |
|---|---|
| Part 1 | Word count bar chart + pie chart across 18 books |
| Part 1 | Zipf's Law log-log frequency plot |
| Part 2 | Vocabulary coverage curve |
| Part 2 | Train/Val/Test split bar chart |
| Part 3 | Model size comparison (4 models, parameter counts) |
| Part 4 | Training curves — accuracy and loss (all 4 models) |
| Part 4 | Perplexity learning curves — val + train/val gap |
| Part 5 | Validation perplexity and accuracy bar charts |
| Part 5 | Perplexity vs random baseline |
| Part 5 | Perplexity vs unigram baseline (log scale) |
| Part 6 | Final test evaluation chart (GRU val vs test) |
| Part 6 | Top-10 predicted next words for 3 seed phrases |
| Part 6 | Word embedding PCA — 5 semantic groups across 10,000-word space |
| Part 7 | Text generation at temperatures 0.5, 0.8, 1.2 |

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Language_Modeling_Gutenberg.ipynb`
3. Click **Runtime → Run All**
4. All data downloads automatically — no manual setup needed

GPU is not required but speeds up LSTM/GRU/BiLSTM training. In Colab: `Runtime → Change runtime type → T4 GPU`.

Total runtime: ~15–20 minutes on GPU, ~40–60 minutes on CPU.

### Option 2 — Local (Jupyter)

```bash
# Install dependencies
pip install tensorflow nltk numpy matplotlib scikit-learn

# Download NLTK data (done automatically in notebook, but you can pre-download)
python -c "import nltk; nltk.download('gutenberg')"

# Launch notebook
jupyter notebook Language_Modeling_Gutenberg.ipynb
```

---

## Requirements

```
tensorflow >= 2.12
nltk >= 3.8
numpy >= 1.23
matplotlib >= 3.6
scikit-learn >= 1.2
```

No GPU required. Tested on Google Colab (TF 2.16, Python 3.10).

---

## Project Structure

```
language-modeling-gutenberg/
├── Language_Modeling_Gutenberg.ipynb   # Main notebook — all code, outputs, and report
└── README.md                           # This file
```

The notebook is self-contained:
- Downloads data automatically via NLTK
- All preprocessing, training, evaluation, and visualization in one file
- Written report in Part 8 with full analysis and results tables

---

## Key Takeaways

1. **GRU > LSTM > BiLSTM > Dense NN** for this task and corpus size
2. **Sequential inductive bias matters** — Dense NN performs worst despite most parameters
3. **Bidirectional context does not help** for next-word prediction (left-to-right task)
4. **Honest baseline comparison**: GRU is only 1.69x better than unigram — word frequency dominates at this scale
5. **Temperature controls diversity vs coherence** in text generation; temperature=0.5 collapses into `<UNK>` loops (a known artifact of word-level models with explicit UNK tokens)
6. **Word embeddings learn semantics** — PCA shows maritime, religious, and conflict words cluster separately in the 64-dimensional embedding space

---

## License

Dataset: [Project Gutenberg](https://www.gutenberg.org/) — public domain.  
Code: MIT License.
