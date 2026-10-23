"""Build the whole system once, and run the full baseline comparison.

Both the Streamlit app (app.py) and the command-line script
(run_experiments.py) use these two functions, so they always agree.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import policies
from .answerer import AnswerModel
from .data import build_corpus, gold_passage_ids, load_pubmedqa, split_questions
from .evaluation import compare_to_baseline, evaluate_policy, precompute_rounds, tune
from .llm_answerer import LLMAnswerModel, llm_available
from .retrievers import RETRIEVERS
from .uncertainty import raw_agreement, raw_relevance


@dataclass
class MedRAGSystem:
    """Everything the policies need, kept together in one object."""

    train: list  # training questions
    val: list  # validation questions (for tuning)
    test: list  # test questions (for the final report)
    passages: list  # the corpus: every abstract part of every question
    gold_ids: dict  # pmid -> set of gold passage numbers
    retriever: object  # BM25 / TF-IDF / Dense (LSA)
    answer_model: object  # AnswerModel or LLMAnswerModel
    passage_vectors: object  # TF-IDF vectors used for the agreement signal
    relevance_reference: float  # "typical high" relevance (for normalising R)
    agreement_reference: float  # "typical high" agreement (for normalising A)


# Answer models the user can pick. The LLM one is offered only when the
# optional packages (torch + transformers) are installed.
ANSWERERS = {"Classifier": AnswerModel}
if llm_available():
    ANSWERERS["Flan-T5 (LLM)"] = LLMAnswerModel


def build_system(retriever_name="BM25", answerer_name="Classifier", seed=42, llm_name=None):
    """Load data, build the index, train the answerer, calibrate the scales.

    llm_name: optional Hugging Face model name or local folder for the LLM
    answerer (default google/flan-t5-base).
    """
    questions = load_pubmedqa()
    train, val, test = split_questions(questions, seed=seed)
    passages = build_corpus(questions)
    texts = [p["text"] for p in passages]

    retriever = RETRIEVERS[retriever_name]().fit(texts)  # build the search index
    if answerer_name == "Flan-T5 (LLM)":
        answer_model = LLMAnswerModel(llm_name) if llm_name else LLMAnswerModel()
    else:
        answer_model = ANSWERERS[answerer_name]()
    answer_model.fit(train)  # learn yes/no/maybe (the LLM needs no training)

    # Separate TF-IDF vectors (length 1) used to compare passages with each
    # other for the agreement signal, whatever retriever is selected.
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    passage_vectors = vectorizer.fit_transform(texts)

    # Calibration: look at the top-3 results for each TRAINING question and
    # record the raw relevance and agreement. The 95th percentile becomes the
    # "1.0" of each scale.
    relevances, agreements = [], []
    for q in train:
        ids, scores = retriever.search(q["question"], 3)
        relevances.append(raw_relevance(scores))
        agreements.append(raw_agreement(ids, passage_vectors))

    return MedRAGSystem(
        train=train,
        val=val,
        test=test,
        passages=passages,
        gold_ids=gold_passage_ids(passages),
        retriever=retriever,
        answer_model=answer_model,
        passage_vectors=passage_vectors,
        relevance_reference=float(np.percentile(relevances, 95)),
        agreement_reference=float(np.percentile(agreements, 95)),
    )


def build_live_system(base, articles, question):
    """Make a small system over live PubMed articles, reusing the base system's
    answer model and settings. The same UAR-Med loop can then run on it.

    question: the user's question. The relevance scale of the big library does
    not fit a 20-paper corpus, so here R measures how close each passage
    scores to the BEST passage for this question.
    """
    passages = [
        {"pmid": a["pmid"], "position": i, "text": text, "title": a["title"]}
        for a in articles
        for i, text in enumerate(a["contexts"])
    ]
    texts = [p["text"] for p in passages]
    retriever = type(base.retriever)().fit(texts)  # same kind of retriever, new index
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    best_score = float(retriever.score(question).max())

    return MedRAGSystem(
        train=base.train,
        val=base.val,
        test=[],
        passages=passages,
        gold_ids=gold_passage_ids(passages),
        retriever=retriever,
        answer_model=base.answer_model,
        passage_vectors=vectorizer.fit_transform(texts),
        relevance_reference=best_score if best_score > 0 else 1.0,
        agreement_reference=base.agreement_reference,
    )


def tune_on_validation(system, start_k=3, step=2, budget=10, cost_weight=0.05):
    """Tune UAR-Med AND the threshold baseline on the validation set.

    Both adaptive methods get the same tuning effort, which keeps the
    comparison fair. Returns (best UAR-Med setting, best threshold setting).
    """
    table = precompute_rounds(system.val, system, start_k, step, budget)
    best_uar = tune(table, budget, cost_weight)
    # Threshold baseline = weights fixed at (1, 0, 0); only tau is searched.
    best_threshold = tune(table, budget, cost_weight, weight_grid=[(1.0, 0.0, 0.0)])
    return best_uar, best_threshold


def run_comparison(system, weights, tau, threshold_tau, start_k=3, step=2, budget=10,
                   abstain_threshold=None, questions=None):
    """Evaluate every policy on the same questions and return two lists.

    summaries : one dict per policy (the main results table), including the
                significance test against fixed top-5
    curve     : fixed top-k for k = 1..budget (accuracy vs cost plot)
    """
    questions = system.test if questions is None else questions

    # Each entry: (name, function that maps a question string -> PolicyResult).
    # "lambda question: ..." is a tiny unnamed function.
    candidates = [
        ("No retrieval", lambda question: policies.no_retrieval(question, system)),
        ("Fixed top-3", lambda question: policies.fixed_top_k(question, system, 3)),
        ("Fixed top-5", lambda question: policies.fixed_top_k(question, system, 5)),
        ("Fixed top-10", lambda question: policies.fixed_top_k(question, system, 10)),
        (
            "Threshold (R only)",
            lambda question: policies.uar_med(
                question, system, (1.0, 0.0, 0.0), threshold_tau, start_k, step, budget
            ),
        ),
        (
            "UAR-Med (ours)",
            lambda question: policies.uar_med(
                question, system, weights, tau, start_k, step, budget
            ),
        ),
    ]

    summaries, all_rows = [], {}
    for name, policy_fn in candidates:
        # Only UAR-Med is allowed to abstain; the others always answer.
        abstain = abstain_threshold if name == "UAR-Med (ours)" else None
        summary, rows = evaluate_policy(name, policy_fn, questions, system, abstain)
        summary["retriever"] = system.retriever.name
        summaries.append(summary)
        all_rows[name] = rows  # keep per-question results for the tests below

    # Significance: compare every policy with fixed top-5 on the same questions.
    for summary in summaries:
        if summary["policy"] == "Fixed top-5":
            continue  # comparing the baseline with itself means nothing
        summary.update(compare_to_baseline(all_rows[summary["policy"]], all_rows["Fixed top-5"]))

    curve = []
    for k in range(1, budget + 1):
        # "k=k" freezes the current value of k inside the lambda.
        summary, _ = evaluate_policy(
            f"Fixed top-{k}",
            lambda question, k=k: policies.fixed_top_k(question, system, k),
            questions,
            system,
        )
        curve.append(summary)

    return summaries, curve
