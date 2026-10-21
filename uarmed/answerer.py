"""A small answer model that predicts "yes", "no" or "maybe" from text.

We do not use a paid LLM API. Instead, a logistic-regression classifier reads
the question plus the retrieved passages and predicts the answer. It is weak
compared with a large language model, but it is:
    - free and fast (runs on CPU in seconds)
    - the SAME for every retrieval policy, so comparisons are fair
You can later swap it for a small open-source LLM if your results justify it.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class AnswerModel:
    def fit(self, questions):
        """Train on (question + its own abstract) -> correct answer."""
        texts = [q["question"] + " " + " ".join(q["contexts"]) for q in questions]
        labels = [q["answer"] for q in questions]

        # We do NOT remove stop words here: words like "no" and "not" carry
        # meaning for a yes/no answer. ngram_range=(1, 2) also uses word pairs
        # such as "no significant" or "not associated".
        # min_df=2 ignores words seen in only one training example (noise).
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        features = self.vectorizer.fit_transform(texts)

        # max_iter=1000 gives the optimiser enough steps to finish.
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(features, labels)
        self.labels = list(self.classifier.classes_)  # e.g. ["maybe", "no", "yes"]
        return self

    def predict_proba(self, question, passage_texts):
        """Return a dict like {"yes": 0.7, "no": 0.2, "maybe": 0.1}."""
        text = question + " " + " ".join(passage_texts)  # glue everything together
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        return dict(zip(self.labels, probabilities))

    def predict(self, question, passage_texts):
        """Return only the most likely answer, e.g. "yes"."""
        probabilities = self.predict_proba(question, passage_texts)
        return max(probabilities, key=probabilities.get)  # label with highest prob

    def passage_stances(self, passage_texts):
        """Predict which answer EACH passage on its own points towards.

        We call this once for the whole corpus and store the results. The
        uncertainty module then checks whether retrieved passages "vote" for
        the same answer (agreement) or for different answers (conflict).
        """
        features = self.vectorizer.transform(passage_texts)
        predictions = self.classifier.predict(features)
        return np.array(predictions)  # one label per passage
