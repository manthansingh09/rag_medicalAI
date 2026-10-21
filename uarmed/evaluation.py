"""Measuring how good a policy is, and tuning the UAR-Med weights.

Metrics (averaged over all test questions):
    accuracy          : how often the predicted answer equals the true answer
    recall@k          : share of the gold passages that were retrieved
    MRR               : 1 / (rank of the first gold passage); 0 if none found
    evidence found    : share of questions where at least one gold passage
                        was retrieved ("evidence sufficiency")
    avg passages      : how many passages the answerer had to read (cost)
    avg tokens        : how many words those passages contain (cost)
    latency (ms)      : time per question, retrieval + answering
    answered          : share of questions answered (UAR-Med may abstain)
    accuracy answered : accuracy on only the answered questions
    unsafe answers    : share of questions answered WITHOUT any gold evidence
"""

import itertools  # builds every combination of weights for the grid search
import time  # measures latency

import numpy as np

from .uncertainty import compute_signals


def evaluate_policy(name, policy_fn, questions, system, abstain_threshold=None):
    """Run one policy on every question and return (summary, per_question rows).

    policy_fn is a function that takes a question string and returns a
    PolicyResult. abstain_threshold (optional): if the policy's final U is
    above it, the system says "insufficient evidence" instead of answering.
    """
    rows = []
    for q in questions:
        start = time.perf_counter()  # start the stopwatch
        result = policy_fn(q["question"])  # 1) decide which passages to use
        texts = [system.passages[i]["text"] for i in result.passage_ids]
        prediction = system.answer_model.predict(q["question"], texts)  # 2) answer
        latency_ms = (time.perf_counter() - start) * 1000  # stop the stopwatch

        gold = system.gold_ids[q["pmid"]]  # the passages of this question's paper
        hits = [i in gold for i in result.passage_ids]  # True where retrieved is gold
        # MRR: rank of the FIRST gold passage (ranks start at 1, not 0).
        first_hit = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
        evidence_found = any(hits)

        # The system abstains only if a threshold is set AND the policy gives U.
        abstained = (
            abstain_threshold is not None
            and result.uncertainty is not None
            and result.uncertainty > abstain_threshold
        )

        rows.append(
            {
                "pmid": q["pmid"],
                "question": q["question"],
                "true answer": q["answer"],
                "prediction": prediction,
                "correct": prediction == q["answer"],
                "passages": len(result.passage_ids),
                "tokens": sum(len(t.split()) for t in texts),
                "recall": sum(hits) / len(gold),
                "rr": 1.0 / first_hit if first_hit else 0.0,
                "evidence found": evidence_found,
                "answered": not abstained,
                "unsafe": (not abstained) and (not evidence_found),
                "latency_ms": latency_ms,
            }
        )

    summary = summarise(name, rows)
    return summary, rows


def summarise(name, rows):
    """Average the per-question rows into one line of the results table."""
    answered = [r for r in rows if r["answered"]]
    return {
        "policy": name,
        "accuracy": np.mean([r["correct"] for r in rows]),
        "recall@k": np.mean([r["recall"] for r in rows]),
        "MRR": np.mean([r["rr"] for r in rows]),
        "evidence found": np.mean([r["evidence found"] for r in rows]),
        "avg passages": np.mean([r["passages"] for r in rows]),
        "avg tokens": np.mean([r["tokens"] for r in rows]),
        "latency (ms)": np.mean([r["latency_ms"] for r in rows]),
        "answered": len(answered) / len(rows),
        "accuracy answered": np.mean([r["correct"] for r in answered]) if answered else 0.0,
        "unsafe answers": np.mean([r["unsafe"] for r in rows]),
    }


def k_schedule(start_k, step, budget):
    """The k values the adaptive loop can stop at, e.g. (3, 2, 10) -> [3, 5, 7, 9, 10]."""
    schedule = list(range(start_k, budget, step))  # 3, 5, 7, 9
    schedule.append(budget)  # the loop always ends at the budget
    return sorted(set(schedule))  # remove a duplicate if budget was already there


def precompute_rounds(questions, system, start_k, step, budget):
    """For every question and every possible k, store R, A, C and correctness.

    KEY TRICK: R, A and C depend only on WHICH passages were retrieved, not on
    the weights. So we compute them once, and the grid search below can try
    hundreds of (weights, tau) settings with simple arithmetic.
    """
    schedule = k_schedule(start_k, step, budget)
    table = []  # one entry per question: a list of rounds
    for q in questions:
        ranked_ids, ranked_scores = system.retriever.search(q["question"], budget)
        rounds = []
        for k in schedule:
            # Weights (1, 1, 1) here are a placeholder; we only keep R, A, C.
            s = compute_signals(ranked_ids[:k], ranked_scores[:k], system, (1, 1, 1))
            texts = [system.passages[i]["text"] for i in ranked_ids[:k]]
            correct = system.answer_model.predict(q["question"], texts) == q["answer"]
            rounds.append({"k": k, "R": s["R"], "A": s["A"], "C": s["C"], "correct": correct})
        table.append(rounds)
    return table


def simulate(table, weights, tau):
    """Replay the UAR-Med stopping rule on the precomputed table.

    Returns (accuracy, average number of passages).
    """
    w1, w2, w3 = weights
    total = w1 + w2 + w3
    correct, used = [], []
    for rounds in table:
        chosen = rounds[-1]  # if U never drops below tau, we end at the budget
        for r in rounds:
            u = (w1 * (1 - r["R"]) + w2 * (1 - r["A"]) + w3 * r["C"]) / total
            if u <= tau:  # first round where evidence is sufficient
                chosen = r
                break
        correct.append(chosen["correct"])
        used.append(chosen["k"])
    return float(np.mean(correct)), float(np.mean(used))


def tune(table, budget, cost_weight=0.05, weight_grid=None, tau_grid=None):
    """Grid search for the best (weights, tau) on the validation table.

    We maximise:  accuracy - cost_weight * (avg passages / budget)
    so a setting that is equally accurate but reads fewer passages wins.
    """
    if weight_grid is None:
        values = [0.0, 0.5, 1.0]
        # Every (w1, w2, w3) combination except all zeros: 3^3 - 1 = 26 options.
        weight_grid = [w for w in itertools.product(values, repeat=3) if sum(w) > 0]
    if tau_grid is None:
        tau_grid = np.round(np.arange(0.05, 1.0, 0.05), 2)  # 0.05, 0.10, ... 0.95

    best = None
    for weights in weight_grid:
        for tau in tau_grid:
            accuracy, avg_k = simulate(table, weights, tau)
            objective = accuracy - cost_weight * avg_k / budget
            if best is None or objective > best["objective"]:
                best = {
                    "weights": tuple(float(w) for w in weights),
                    "tau": float(tau),
                    "val accuracy": accuracy,
                    "val avg passages": avg_k,
                    "objective": objective,
                }
    return best
