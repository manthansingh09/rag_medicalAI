# UAR-Med: Uncertainty-Guided Adaptive Retrieval for Reliable Medical Question Answering

A medical RAG prototype where the system **decides for itself whether it has found
enough evidence**, and retrieves more passages only when it has not.

- No paid API, no GPU: everything runs on a CPU with Python and scikit-learn.
- Dataset: [PubMedQA](https://pubmedqa.github.io) (1,000 expert-labelled questions, MIT licence), bundled in `data/`.
- Comes with a Streamlit web app you can deploy for free.

> ⚠️ Research and learning prototype. It is **not** a source of medical advice.

---

## 1. The idea in one picture

```
question ──► retrieve top-3 ──► compute R, A, C ──► U = uncertainty
                  ▲                                   │
                  │      U > τ and budget left        │  U ≤ τ (or budget used up)
                  └──────── retrieve 2 more ◄─────────┤
                                                      ▼
                                          answer  (or abstain if U still high)
```

| Signal | Meaning | How it is computed |
|---|---|---|
| **R** Relevance | Do the passages match the question? | mean retrieval score ÷ a typical high score from training data |
| **A** Agreement | Do the passages talk about the same thing? | mean pairwise cosine similarity between passages |
| **C** Conflict | Do the passages point to different answers? | share of passages whose yes/no/maybe vote differs from the majority |

```
U = ( w1·(1−R) + w2·(1−A) + w3·C ) / (w1 + w2 + w3)
```

`w1, w2, w3` and the stopping threshold `τ` are found by a grid search on the
**validation** set. The **test** set is used only once, for the final numbers.

## 2. What is compared

| System | How it decides how many passages to read |
|---|---|
| No retrieval | none, the question alone |
| Fixed top-3 / top-5 / top-10 | always the same number |
| Threshold (R only) | adds passages until relevance R is high enough |
| **UAR-Med (ours)** | adds passages until the combined uncertainty U is low enough |

All of them use the same questions, corpus, retriever and answer model, so only
the retrieval rule changes. Three retrievers are available: **BM25**, **TF-IDF**
and **Dense (LSA)** (a light, CPU-only stand-in for sentence embeddings).

**Metrics:** accuracy, Recall@k, MRR, evidence found (sufficiency), average
passages and tokens read (cost), latency, answered rate, and *unsafe answers*
(answering without having retrieved any gold evidence). The app also plots
accuracy and recall against cost, which shows accuracy at a fixed retrieval budget.

## 3. Project structure

```
app.py                 Streamlit web app (start here to *use* the project)
run_experiments.py     same experiment from the command line / Colab
uarmed/
  data.py              load PubMedQA, build the passage corpus, split train/val/test
  retrievers.py        TF-IDF, BM25 (written from scratch), Dense (LSA)
  answerer.py          yes/no/maybe classifier (TF-IDF + logistic regression)
  uncertainty.py       the R, A, C signals and the U formula
  policies.py          no retrieval, fixed top-k, threshold, UAR-Med loop
  evaluation.py        metrics + fast grid search for the weights
  pipeline.py          builds everything and runs the full comparison
data/ori_pqal.json     PubMedQA labelled set (MIT licence, see data/PUBMEDQA_LICENSE)
```

**Suggested reading order:** `data.py` → `retrievers.py` → `answerer.py` →
`uncertainty.py` → `policies.py` → `evaluation.py` → `pipeline.py` → `app.py`.
Every file has comments explaining each step.

## 4. Run it

### On your computer

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
python run_experiments.py --all    # prints the tables, saves results/*.csv
```

### In Google Colab

```python
!git clone https://github.com/manthansingh09/rag_medicalAI.git
%cd rag_medicalAI
!pip install -q -r requirements.txt
!python run_experiments.py --all
```

### Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub (already done if you are reading it there).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose this repository and branch → main file path **`app.py`**.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically. The first
   load takes a few seconds while the index is built; after that it is cached.

## 5. First results (BM25, 300 test questions, default settings)

| Policy | Accuracy | Recall@k | Evidence found | Avg passages |
|---|---|---|---|---|
| No retrieval | 53.3% | 0.0% | 0.0% | 0.00 |
| Fixed top-3 | 54.0% | 62.3% | 98.3% | 3.00 |
| Fixed top-5 | 54.0% | 69.0% | 98.7% | 5.00 |
| Fixed top-10 | 53.3% | 75.2% | 99.0% | 10.00 |
| Threshold (R only) | 54.0% | 62.5% | 98.7% | 3.12 |
| UAR-Med (ours) | 54.3% | 62.6% | 98.3% | 3.15 |

**How to read this:**

- The adaptive methods match fixed top-5 accuracy while reading ~37% fewer
  passages, and match top-10 accuracy with ~69% fewer. The cost is lower recall
  (62.6% vs 69.0% for top-5), because most questions stop after 3 passages.
- The accuracy differences (≈1 point) are **smaller than the noise** for 300
  questions (±5.7 points). They are not evidence that one policy answers better.
- The bottleneck is the answer model. A bag-of-words classifier stays close to
  the "always say yes" rate (53.7% on this split), so extra evidence barely changes
  its answers. The retrieval metrics (recall, evidence found, cost) are the
  meaningful part of these results.
- The conflict signal C is weak for the same reason: the classifier votes "yes"
  for about 93% of single passages.

## 6. Next steps (in order)

1. **Better answerer.** Replace `AnswerModel` with a small open-source LLM or a
   biomedical NLI model (for example in Colab with a free GPU). Keep the same
   interface (`predict`, `predict_proba`, `passage_stances`) and the rest of the code works unchanged.
   This also makes C (conflict) meaningful.
2. **Harder retrieval.** Add MedQA or the PubMedQA unlabelled set to the corpus so
   the retriever has more distracting passages to get past.
3. **More signals.** Score margin between the top two passages, score spread, or
   answer-probability entropy are easy additions to `uncertainty.py`.
4. **Statistics.** Use bootstrap confidence intervals or a paired test (McNemar)
   before claiming one policy beats another.

## Citation for the dataset

Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). *PubMedQA: A Dataset for
Biomedical Research Question Answering.* EMNLP-IJCNLP.
