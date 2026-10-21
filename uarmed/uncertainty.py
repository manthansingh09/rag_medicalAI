"""The three evidence signals and the combined uncertainty score U.

For a set of retrieved passages we compute:

    R  (Relevance) : how well do the passages match the question?     0..1
    A  (Agreement) : do the passages talk about the same thing?        0..1
    C  (Conflict)  : do the passages point to different answers?       0..1

and combine them into one uncertainty score:

    U = ( w1 * (1 - R)  +  w2 * (1 - A)  +  w3 * C ) / (w1 + w2 + w3)

High R and A lower U; high C raises U. Dividing by (w1 + w2 + w3) keeps U
between 0 and 1, so the stopping threshold is easy to interpret.

NOTE: this is a research HYPOTHESIS, not an established medical measure. The
experiments tab checks whether it actually helps.
"""

import numpy as np


def normalise(value, reference):
    """Scale a raw value into 0..1 by dividing by a 'typical high' value.

    Raw scores live on different scales (BM25 can be 30, cosine is at most 1),
    so each one is divided by a reference value measured on the training set
    (its 95th percentile). Anything above the reference is capped at 1.
    """
    if reference <= 0:  # guard against division by zero
        return 0.0
    return float(min(value / reference, 1.0))


def raw_relevance(scores):
    """Average retrieval score of the passages (before normalising)."""
    return float(np.mean(scores)) if len(scores) else 0.0


def raw_agreement(passage_ids, passage_vectors):
    """Average cosine similarity between every PAIR of retrieved passages.

    passage_vectors holds a length-1 TF-IDF vector for each passage, so
    V @ V.T gives a matrix of all pairwise cosine similarities.
    """
    if len(passage_ids) < 2:
        return 0.0  # a single passage cannot confirm itself
    vectors = passage_vectors[passage_ids]  # keep only the retrieved rows
    similarity = (vectors @ vectors.T).toarray()  # n x n matrix
    # np.triu_indices(n, k=1) picks the cells above the diagonal, i.e. each
    # pair once and never a passage with itself.
    upper = np.triu_indices(len(passage_ids), k=1)
    return float(similarity[upper].mean())


def conflict(passage_ids, passage_stances):
    """Fraction of passages that disagree with the majority answer vote.

    Example: stances = ["yes", "yes", "no", "yes"] -> majority "yes" (3 votes)
             conflict = 1 - 3/4 = 0.25
    """
    if len(passage_ids) == 0:
        return 0.0
    stances = passage_stances[passage_ids]
    _, vote_counts = np.unique(stances, return_counts=True)  # votes per label
    return float(1.0 - vote_counts.max() / len(stances))


def compute_signals(passage_ids, scores, system, weights):
    """Compute R, A, C and U for one set of retrieved passages.

    passage_ids : numbers of the retrieved passages
    scores      : their retrieval scores (same order)
    system      : the MedRAGSystem (holds vectors, stances and references)
    weights     : (w1, w2, w3)
    """
    w1, w2, w3 = weights

    relevance = normalise(raw_relevance(scores), system.relevance_reference)
    agreement = normalise(
        raw_agreement(passage_ids, system.passage_vectors), system.agreement_reference
    )
    conflict_score = conflict(passage_ids, system.passage_stances)

    total_weight = w1 + w2 + w3
    if total_weight == 0:  # all weights zero -> no information -> stay unsure
        uncertainty = 1.0
    else:
        uncertainty = (
            w1 * (1 - relevance) + w2 * (1 - agreement) + w3 * conflict_score
        ) / total_weight

    return {"R": relevance, "A": agreement, "C": conflict_score, "U": uncertainty}
