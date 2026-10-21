"""UAR-Med Streamlit app.

Run locally with:      streamlit run app.py
Deploy for free on:    https://share.streamlit.io (point it at this file)

The app has three tabs:
    1. Ask a question  -> watch the adaptive retrieval loop step by step
    2. Experiments     -> compare UAR-Med with the baselines on the test set
    3. How it works    -> the idea and the formula in plain words
"""

import altair as alt  # charts (installed together with Streamlit)
import pandas as pd  # tables
import streamlit as st  # the web app framework

from uarmed import policies
from uarmed.pipeline import ANSWERERS, build_system, run_comparison, tune_on_validation
from uarmed.retrievers import RETRIEVERS

# Must be the first Streamlit command: sets the browser tab title and layout.
st.set_page_config(page_title="UAR-Med", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers
# Streamlit re-runs this whole file every time you click something. Caching
# means slow work (building the index, tuning) happens only once per setting.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building the search index and answer model...")
def get_system(retriever_name, answerer_name):
    # cache_resource: keep ONE shared object in memory (not copied each time).
    return build_system(retriever_name, answerer_name)


@st.cache_data(show_spinner="Tuning UAR-Med on the validation set...")
def get_tuned(retriever_name, answerer_name, start_k, step, budget, cost_weight):
    # cache_data: remember the returned values for these exact arguments.
    system = get_system(retriever_name, answerer_name)
    return tune_on_validation(system, start_k, step, budget, cost_weight)


@st.cache_data(show_spinner="Running every policy on the test set...")
def get_comparison(retriever_name, answerer_name, weights, tau, threshold_tau,
                   start_k, step, budget, abstain):
    system = get_system(retriever_name, answerer_name)
    summaries, curve = run_comparison(
        system, weights, tau, threshold_tau, start_k, step, budget, abstain
    )
    return pd.DataFrame(summaries), pd.DataFrame(curve)


# ---------------------------------------------------------------------------
# Sidebar: all the settings
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

retriever_name = st.sidebar.selectbox("Retriever", list(RETRIEVERS.keys()))
# The LLM option appears only when torch + transformers are installed
# (see requirements-llm.txt); free Streamlit hosting uses the classifier.
answerer_name = st.sidebar.selectbox("Answer model", list(ANSWERERS.keys()))

st.sidebar.subheader("Retrieval budget")
start_k = st.sidebar.slider("Start with k passages", 1, 5, 3)
step = st.sidebar.slider("Add this many per round", 1, 5, 2)
budget = st.sidebar.slider("Maximum passages (budget)", start_k, 15, 10)

st.sidebar.subheader("Weights and threshold")
cost_weight = st.sidebar.slider(
    "Cost penalty used while tuning",
    0.0, 0.3, 0.05, 0.01,
    help="Higher = tuning prefers settings that read fewer passages.",
)
best_uar, best_threshold = get_tuned(
    retriever_name, answerer_name, start_k, step, budget, cost_weight
)

use_tuned = st.sidebar.checkbox("Use weights tuned on validation data", value=True)
if use_tuned:
    weights = best_uar["weights"]
    tau = best_uar["tau"]
    st.sidebar.caption(
        f"Tuned: w1={weights[0]}, w2={weights[1]}, w3={weights[2]}, τ={tau}"
    )
else:
    w1 = st.sidebar.slider("w1 (low relevance)", 0.0, 1.0, 1.0, 0.1)
    w2 = st.sidebar.slider("w2 (low agreement)", 0.0, 1.0, 0.5, 0.1)
    w3 = st.sidebar.slider("w3 (conflict)", 0.0, 1.0, 0.5, 0.1)
    tau = st.sidebar.slider("τ: stop when U ≤ τ", 0.0, 1.0, 0.5, 0.05)
    weights = (w1, w2, w3)

st.sidebar.subheader("Safety")
use_abstain = st.sidebar.checkbox("Abstain when still uncertain", value=False)
abstain = (
    st.sidebar.slider("Abstain if final U is above", 0.0, 1.0, 0.6, 0.05)
    if use_abstain
    else None
)

system = get_system(retriever_name, answerer_name)

# ---------------------------------------------------------------------------
# Page header and tabs
# ---------------------------------------------------------------------------
st.title("🩺 UAR-Med")
st.markdown(
    "**Uncertainty-Guided Adaptive Retrieval for Reliable Medical Question Answering.** "
    "The system searches PubMed abstracts, checks whether the evidence is "
    "sufficient, and searches again only when it is not."
)
st.warning(
    "Research prototype for learning purposes. It is **not** medical advice.",
    icon="⚠️",
)

tab_ask, tab_experiments, tab_about = st.tabs(
    ["🔎 Ask a question", "📊 Experiments", "📖 How it works"]
)

# ---------------------------------------------------------------------------
# Tab 1: ask a single question and watch the loop
# ---------------------------------------------------------------------------
with tab_ask:
    mode = st.radio(
        "Question source", ["Pick from the test set", "Type my own"], horizontal=True
    )
    if mode == "Pick from the test set":
        # Show the first 100 test questions in a drop-down list.
        options = system.test[:100]
        chosen = st.selectbox(
            "Test question", options, format_func=lambda q: q["question"]
        )
        question_text = chosen["question"]
    else:
        chosen = None  # a typed question has no known correct answer
        question_text = st.text_input(
            "Your question", "Does metformin reduce the risk of cardiovascular events?"
        )

    if question_text.strip():
        result = policies.uar_med(question_text, system, weights, tau, start_k, step, budget)
        texts = [system.passages[i]["text"] for i in result.passage_ids]
        probabilities = system.answer_model.predict_proba(question_text, texts)
        answer = max(probabilities, key=probabilities.get)
        abstained = abstain is not None and result.uncertainty > abstain

        left, right = st.columns([3, 2])
        with left:
            st.subheader("1. The adaptive retrieval loop")
            steps = pd.DataFrame(result.steps)
            st.dataframe(
                steps.style.format({"R": "{:.2f}", "A": "{:.2f}", "C": "{:.2f}", "U": "{:.2f}"}),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"Each row is one round. The loop stops as soon as U ≤ τ = {tau:.2f} "
                f"or the budget of {budget} passages is reached."
            )
        with right:
            st.subheader("2. Answer")
            if abstained:
                st.error(f"Insufficient evidence (final U = {result.uncertainty:.2f}). No answer given.")
            else:
                st.success(f"Predicted answer: **{answer.upper()}**")
            if chosen is not None:
                st.info(f"Expert answer from PubMedQA: **{chosen['answer'].upper()}**")
            prob_df = pd.DataFrame({"answer": list(probabilities), "probability": list(probabilities.values())})
            st.bar_chart(prob_df, x="answer", y="probability", height=200)

        st.subheader(f"3. Evidence used ({len(result.passage_ids)} passages)")
        gold = system.gold_ids[chosen["pmid"]] if chosen is not None else set()
        # Which answer each passage points to on its own (used for conflict C).
        votes = system.answer_model.passage_stances(question_text, texts)
        for rank, (passage_id, vote) in enumerate(zip(result.passage_ids, votes), start=1):
            passage = system.passages[passage_id]
            badge = " ✅ gold evidence" if passage_id in gold else ""
            with st.expander(f"#{rank} · PMID {passage['pmid']} · vote: {vote}{badge}"):
                st.write(passage["text"])
                st.markdown(f"[Open on PubMed](https://pubmed.ncbi.nlm.nih.gov/{passage['pmid']}/)")

# ---------------------------------------------------------------------------
# Tab 2: the full comparison on the test set
# ---------------------------------------------------------------------------
with tab_experiments:
    st.markdown(
        f"Every policy uses the **same {len(system.test)} test questions**, the same "
        f"corpus of **{len(system.passages):,} passages**, the same retriever "
        f"(**{retriever_name}**) and the same answer model "
        f"(**{system.answer_model.name}**). Only the rule for "
        "*how many passages to read* changes."
    )
    col1, col2 = st.columns(2)
    col1.metric(
        "UAR-Med tuned on validation",
        f"acc {best_uar['val accuracy']:.3f}",
        f"{best_uar['val avg passages']:.2f} passages",
        delta_color="off",
    )
    col2.metric(
        "Threshold baseline tuned on validation",
        f"acc {best_threshold['val accuracy']:.3f}",
        f"{best_threshold['val avg passages']:.2f} passages",
        delta_color="off",
    )

    if st.button("▶️ Run the comparison", type="primary"):
        st.session_state["run_experiments"] = True  # remember the click

    if st.session_state.get("run_experiments"):
        table, curve = get_comparison(
            retriever_name, answerer_name, tuple(weights), tau, best_threshold["tau"],
            start_k, step, budget, abstain,
        )

        st.subheader("Results")
        percent = ["accuracy", "recall@k", "MRR", "evidence found", "answered",
                   "accuracy answered", "unsafe answers"]
        quality = table.drop(columns=["retriever", "Δ accuracy", "CI low", "CI high", "p-value"])
        st.dataframe(
            quality.style.format(
                {**{c: "{:.1%}" for c in percent},
                 "avg passages": "{:.2f}", "avg tokens": "{:.0f}", "latency (ms)": "{:.2f}"}
            ),
            hide_index=True,
            width="stretch",
        )

        st.subheader("Is the difference real? (each policy vs fixed top-5)")
        significance = table[["policy", "accuracy", "Δ accuracy", "CI low", "CI high", "p-value"]]
        st.dataframe(
            significance.style.format(
                {"accuracy": "{:.1%}", "Δ accuracy": "{:+.1%}", "CI low": "{:+.1%}",
                 "CI high": "{:+.1%}", "p-value": "{:.3f}"},
                na_rep="baseline",
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "CI = 95% paired bootstrap confidence interval of the accuracy difference. "
            "p-value from McNemar's test. If the CI contains 0 (or p ≥ 0.05), the "
            "difference could be just luck."
        )

        st.subheader("Accuracy and recall versus retrieval cost")
        st.markdown(
            "The line is **fixed top-k** for k = 1 … budget. A good adaptive "
            "method sits *above or to the left* of the line: same quality, fewer passages."
        )
        adaptive = table[table["policy"].isin(["Threshold (R only)", "UAR-Med (ours)"])]
        for metric in ["accuracy", "recall@k"]:
            line = alt.Chart(curve).mark_line(point=True, color="gray").encode(
                x=alt.X("avg passages", title="Average passages read"),
                y=alt.Y(metric, scale=alt.Scale(zero=False), title=metric),
                tooltip=["policy", metric, "avg passages"],
            )
            dots = alt.Chart(adaptive).mark_point(size=200, filled=True).encode(
                x="avg passages", y=metric, color="policy", tooltip=["policy", metric, "avg passages"]
            )
            st.altair_chart((line + dots).properties(height=300), width="stretch")

        st.download_button(
            "Download results (CSV)", table.to_csv(index=False), "uar_med_results.csv", "text/csv"
        )

# ---------------------------------------------------------------------------
# Tab 3: explanation
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        r"""
### The problem
Normal RAG always retrieves a fixed number of passages (say 5). That wastes effort
on easy questions and may be too little for hard ones.

### The idea
Retrieve a few passages, **measure how uncertain the evidence is**, and only
retrieve more when needed.

### The three signals
| Signal | Question it answers | How we compute it |
|---|---|---|
| **R** Relevance | Do the passages match the question? | mean retrieval score ÷ a typical high score |
| **A** Agreement | Do the passages discuss the same thing? | mean pairwise cosine similarity between passages |
| **C** Conflict | Do the passages point to different answers? | share of passages whose yes/no/maybe vote differs from the majority |

### The uncertainty score
$$
U = \frac{w_1(1-R) + w_2(1-A) + w_3 C}{w_1 + w_2 + w_3}
$$

### The loop
1. Retrieve the top *k* passages.
2. Compute R, A, C and U.
3. If U ≤ τ → stop and answer. Otherwise retrieve more.
4. Never exceed the budget. Optionally abstain if U is still high.

The weights *w₁, w₂, w₃* and the threshold *τ* are chosen by a grid search on the
**validation** set; the **test** set is only used for the final numbers.

### Dataset
[PubMedQA](https://pubmedqa.github.io) – 1,000 expert-labelled questions (MIT licence).
"""
    )
