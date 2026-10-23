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

**Significance:** every policy is compared with fixed top-5 on the same questions
using a 95% paired bootstrap confidence interval and McNemar's test, so you can
tell a real improvement from luck.

**Answer models:** the default is a small TF-IDF + logistic-regression classifier
with extra "finding phrase" features such as *no significant difference* or
*significantly reduced* (fast, deploys anywhere). An optional zero-shot **Flan-T5** answerer
(`uarmed/llm_answerer.py`) can be switched on in Colab; see section 4.

## What the app answers

In the **Ask a question** tab you get:

- **A written answer.** The best-matching sentences from the evidence, joined into
  a short paragraph, each with a citation like **[2]**. Nothing is invented,
  so every sentence can be traced to a real paper.
- **A short yes/no/maybe verdict**, only for yes/no questions ("Does…", "Is…").
- **A weak-evidence warning** when the passages match the question poorly.
- **Two places to search:** the offline PubMedQA library (1,000 abstracts), or
  **Live PubMed**, which uses NCBI's free E-utilities service (no key, no cost)
  and covers millions of abstracts. The same UAR-Med loop runs on the fetched
  papers. If PubMed cannot be reached, the app falls back to the library.

## 3. Project structure

```
app.py                 Streamlit web app (start here to *use* the project)
run_experiments.py     same experiment from the command line / Colab
uarmed/
  data.py              load PubMedQA, build the passage corpus, split train/val/test
  retrievers.py        TF-IDF, BM25 (written from scratch), Dense (LSA)
  answerer.py          yes/no/maybe classifier (TF-IDF + finding cues + logistic regression)
  explainer.py         writes the answer paragraph from evidence sentences, with citations
  pubmed.py            OPTIONAL live PubMed search (NCBI's free service, no key)
  llm_answerer.py      OPTIONAL zero-shot Flan-T5 answerer (needs torch + transformers)
  uncertainty.py       the R, A, C signals and the U formula
  policies.py          no retrieval, fixed top-k, threshold, UAR-Med loop
  evaluation.py        metrics, significance tests, fast grid search for the weights
  pipeline.py          builds everything and runs the full comparison
requirements.txt       packages for the app (what Streamlit Cloud installs)
requirements-llm.txt   extra packages for the optional LLM answerer
data/ori_pqal.json     PubMedQA labelled set (MIT licence, see data/PUBMEDQA_LICENSE)
```

**Suggested reading order:** `data.py` → `retrievers.py` → `answerer.py` →
`uncertainty.py` → `policies.py` → `evaluation.py` → `pipeline.py` → `app.py`
(`llm_answerer.py` last, once the rest makes sense).
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

**With the Flan-T5 answerer** (Runtime → Change runtime type → T4 GPU first):

```python
!pip install -q -r requirements-llm.txt
!python run_experiments.py --answerer "Flan-T5 (LLM)"
!python run_experiments.py --answerer "Flan-T5 (LLM)" --llm google/flan-t5-large   # bigger model
```

The model downloads from Hugging Face on first use. On a CPU this is slow
(tens of minutes for the full experiment); on a Colab GPU it takes a few minutes.
You can also run `streamlit run app.py` locally after installing
`requirements-llm.txt`: an **Answer model** drop-down appears in the sidebar.
Caveat: Flan-T5's instruction-tuning mix may include PubMedQA, so treat its
absolute accuracy with care; the *comparison between policies* stays fair
because every policy uses the same model.

### Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub (already done if you are reading it there).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose this repository and branch → main file path **`app.py`**.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically. The first
   load takes a few seconds while the index is built; after that it is cached.
   The deployed app uses the lightweight classifier; the LLM option is hidden
   because PyTorch is too large for the free tier.

## 5. First results (BM25, 300 test questions, default settings)

| Policy | Accuracy | Macro F1 | Recall@k | Avg passages | Δ acc vs top-5 [95% CI] | McNemar p |
|---|---|---|---|---|---|---|
| No retrieval | 54.0% | 0.32 | 0.0% | 0.00 | +5.3 [−0.7, +11.3] | 0.13 |
| Fixed top-3 | 54.7% | 0.41 | 62.3% | 3.00 | +6.0 [+1.3, +10.7] | 0.02 |
| Fixed top-5 | 48.7% | 0.39 | 69.0% | 5.00 | baseline | – |
| Fixed top-10 | 48.0% | 0.39 | 75.2% | 10.00 | −0.7 [−6.3, +5.0] | 0.91 |
| Threshold (R only) | 55.0% | 0.42 | 63.2% | 3.35 | +6.3 [+1.3, +11.0] | 0.01 |
| UAR-Med (ours) | 55.0% | 0.42 | 63.2% | 3.35 | +6.3 [+1.3, +11.0] | 0.01 |

**How to read this:**

- **More passages is not always better.** With this answer model, accuracy
  drops after about 3-4 passages: extra passages come from *other* papers and
  add misleading phrases. Adaptive retrieval beats fixed top-5 by 6.3 points
  (significant, p = 0.01) while reading 33% fewer passages.
- **But it does not beat a well-chosen fixed k.** UAR-Med is about equal to
  fixed top-3. On BM25 the tuning chose weights (0.5, 0, 0), i.e. relevance
  only, which is why it matches the threshold baseline exactly. With TF-IDF it
  used conflict only and did slightly worse than top-3. So the extra signals
  A and C have not proven useful yet.
- **The answer model is still the bottleneck.** Its accuracy with the paper's
  own abstract is only about 59%. A plain classifier said "yes" for 296 of
  300 questions; finding-phrase features and balanced class weights fixed that
  (now 196 yes / 82 no / 22 maybe), which raised macro F1 from 0.25 to 0.41.
- Watch out for a lucky baseline: always compare against the best fixed k,
  not only top-5.

## 6. Next steps (in order)

1. **Better answerer.** Run the included Flan-T5 answerer in Colab (section 4)
   and check whether more evidence now changes accuracy. It also makes C
   (conflict) meaningful, because each passage is judged *against the question*.
   Any other model works too if it has the same methods
   (`fit`, `predict`, `predict_proba`, `passage_stances`).
2. **Harder retrieval.** Add MedQA or the PubMedQA unlabelled set to the corpus so
   the retriever has more distracting passages to get past.
3. **More signals.** Score margin between the top two passages, score spread, or
   answer-probability entropy are easy additions to `uncertainty.py`.
4. **More test questions.** 300 questions can only detect large differences.
   Cross-validation over all 1,000 questions gives tighter confidence intervals.

## Citation for the dataset

Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). *PubMedQA: A Dataset for
Biomedical Research Question Answering.* EMNLP-IJCNLP.
