"""Retrieval policies: rules that decide HOW MANY passages to give the answerer.

    no_retrieval      -> Baseline 1: answer from the question alone
    fixed_top_k       -> Baseline 2: always take the top k passages
    threshold_policy  -> Baseline 3: keep adding passages until relevance R is
                         high enough (a single signal)
    uar_med           -> Our method: keep adding passages until the combined
                         uncertainty U (R + A + C) is low enough

Every policy returns a PolicyResult so the evaluation code treats them equally.
"""

from dataclasses import dataclass, field

import numpy as np

from .uncertainty import compute_signals


@dataclass
class PolicyResult:
    passage_ids: np.ndarray  # numbers of the passages given to the answerer
    steps: list = field(default_factory=list)  # one dict per retrieval round
    uncertainty: float = None  # final U (only adaptive policies set this)


def no_retrieval(question, system):
    """Baseline 1: no passages at all."""
    return PolicyResult(passage_ids=np.array([], dtype=int))


def fixed_top_k(question, system, k):
    """Baseline 2: always retrieve exactly k passages."""
    passage_ids, _ = system.retriever.search(question, k)
    return PolicyResult(passage_ids=passage_ids)


def uar_med(question, system, weights, tau, start_k=3, step=2, budget=10):
    """UAR-Med: Uncertainty-Aware Adaptive Retrieval.

    1. Retrieve the top `start_k` passages.
    2. Compute R, A, C and the uncertainty U.
    3. If U <= tau  -> evidence looks sufficient -> STOP.
       If U >  tau  -> retrieve `step` more passages and go back to step 2.
    4. Never go beyond `budget` passages.
    """
    # We rank the corpus ONCE, up to the budget. "Retrieving more" then just
    # means reading further down this ranked list. The real cost we care about
    # is how many passages are read by the answerer (or an LLM later on).
    ranked_ids, ranked_scores = system.retriever.search(question, budget)

    k = min(start_k, len(ranked_ids))  # how many passages we currently use
    steps = []  # a log of every round, shown in the app
    while True:
        signals = compute_signals(ranked_ids[:k], ranked_scores[:k], system, weights)
        stop = signals["U"] <= tau or k >= len(ranked_ids)
        steps.append({"k": k, **signals, "decision": "stop" if stop else "retrieve more"})
        if stop:
            break
        k = min(k + step, len(ranked_ids))  # add `step` passages, respect budget

    return PolicyResult(passage_ids=ranked_ids[:k], steps=steps, uncertainty=signals["U"])


def threshold_policy(question, system, min_relevance, start_k=3, step=2, budget=10):
    """Baseline 3: adapt using ONLY the relevance signal R.

    This is the same loop as UAR-Med with weights (1, 0, 0). Then U = 1 - R, so
    "stop when R >= min_relevance" is the same as "stop when U <= 1 - min_relevance".
    Comparing this with UAR-Med shows whether the extra signals (A and C) help.
    """
    return uar_med(
        question,
        system,
        weights=(1.0, 0.0, 0.0),
        tau=1.0 - min_relevance,
        start_k=start_k,
        step=step,
        budget=budget,
    )
