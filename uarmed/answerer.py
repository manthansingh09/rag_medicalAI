"""A small answer model that predicts "yes", "no" or "maybe" from text.

We do not use a paid LLM API. Instead, a logistic-regression classifier reads
the question plus the retrieved passages and predicts the answer. It is weak
compared with a large language model, but it is:
    - free and fast (runs on CPU in seconds)
    - the SAME for every retrieval policy, so comparisons are fair
You can later swap it for a small open-source LLM if your results justify it.

Two tricks stop it from answering "yes" to everything:
    1. FINDING CUES: extra features that count phrases like "no significant
       difference" (points to NO) or "significantly reduced" (points to YES).
       Plain word counts miss these because "no" and "significant" are
       treated as separate, unrelated words.
    2. BALANCED CLASS WEIGHTS: 55% of training answers are "yes", so a plain
       classifier learns that "yes" is a safe guess. class_weight="balanced"
       makes a mistake on a rare answer ("no", "maybe") cost more.
Without them the model said "yes" for 296 of 300 test questions.
"""

import re  # regular expressions, used to find the cue phrases

import numpy as np
import scipy.sparse as sp  # sparse matrices (TF-IDF features are sparse)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Phrases that usually report a NEGATIVE finding (no effect / no difference).
NEGATIVE_CUES = [
    r"\bno (statistically )?significant", r"\bnot (statistically )?significant",
    r"\bdid not\b", r"\bwas not\b", r"\bwere not\b", r"\bno difference",
    r"\bno association", r"\bnot associated", r"\bno effect", r"\bsimilar\b",
    r"\bcomparable\b", r"\bfailed to\b", r"\bnor\b", r"\bno\b", r"\bnot\b",
]
# Phrases that usually report a POSITIVE finding (a real effect).
POSITIVE_CUES = [
    r"\bsignificantly (higher|lower|greater|reduced|increased|improved|better|associated|more|less)",
    r"\bp ?< ?0\.0", r"\bp ?<", r"\bimproved\b", r"\bincreased\b", r"\breduced\b",
    r"\bassociated with\b", r"\beffective\b",
]


def cue_features(texts):
    """One row per text, one column per cue phrase: how often it appears.

    log1p(x) = log(1 + x) keeps large counts from dominating (0 stays 0).
    """
    counts = [
        [len(re.findall(pattern, text.lower())) for pattern in NEGATIVE_CUES + POSITIVE_CUES]
        for text in texts
    ]
    return sp.csr_matrix(np.log1p(np.array(counts, dtype=float)))


class AnswerModel:
    name = "Classifier (TF-IDF + finding cues)"

    def features(self, texts):
        """TF-IDF word features and cue features, side by side."""
        return sp.hstack([self.vectorizer.transform(texts), cue_features(texts)]).tocsr()

    def fit(self, questions):
        """Train on (question + its own abstract) -> correct answer."""
        texts = [q["question"] + " " + " ".join(q["contexts"]) for q in questions]
        labels = [q["answer"] for q in questions]

        # We do NOT remove stop words here: words like "no" and "not" carry
        # meaning for a yes/no answer. ngram_range=(1, 2) also uses word pairs
        # such as "no significant" or "not associated".
        # min_df=2 ignores words seen in only one training example (noise).
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.vectorizer.fit(texts)

        # max_iter=2000 gives the optimiser enough steps to finish.
        self.classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.classifier.fit(self.features(texts), labels)
        self.labels = list(self.classifier.classes_)  # e.g. ["maybe", "no", "yes"]
        return self

    def predict_proba(self, question, passage_texts):
        """Return a dict like {"yes": 0.7, "no": 0.2, "maybe": 0.1}."""
        text = question + " " + " ".join(passage_texts)  # glue everything together
        probabilities = self.classifier.predict_proba(self.features([text]))[0]
        return dict(zip(self.labels, probabilities))

    def predict(self, question, passage_texts):
        """Return only the most likely answer, e.g. "yes"."""
        probabilities = self.predict_proba(question, passage_texts)
        return max(probabilities, key=probabilities.get)  # label with highest prob

    def passage_stances(self, question, passage_texts):
        """Predict which answer EACH passage on its own points towards.

        The uncertainty module then checks whether the retrieved passages
        "vote" for the same answer or for different answers (conflict).
        This simple classifier was trained on whole abstracts, so it looks at
        each passage alone and ignores `question`. The LLM answerer
        (llm_answerer.py) does use the question.
        """
        if len(passage_texts) == 0:
            return np.array([])
        predictions = self.classifier.predict(self.features(passage_texts))
        return np.array(predictions)  # one label per passage
