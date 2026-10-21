"""Three retrievers ("search engines") that rank passages for a question.

Every retriever has the same two methods, so the rest of the code does not
care which one is used:
    fit(texts)      -> learn from the corpus (build the index)
    score(query)    -> give every passage a relevance score (higher = better)
    search(query,k) -> return the k best passage numbers and their scores
"""

import numpy as np  # fast maths on arrays
from sklearn.decomposition import TruncatedSVD  # compresses TF-IDF into dense vectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize  # makes vectors length 1 (for cosine)


class BaseRetriever:
    """Shared code for all retrievers. Child classes only write fit() and score()."""

    name = "base"  # a human-readable name, overwritten by each child class

    def fit(self, texts):
        raise NotImplementedError  # child classes must implement this

    def score(self, query):
        raise NotImplementedError  # child classes must implement this

    def search(self, query, k):
        """Return the numbers (indices) and scores of the k best passages."""
        scores = self.score(query)  # one score per passage in the corpus
        k = min(k, len(scores))  # cannot return more passages than we have
        # np.argsort sorts from small to large, so we sort -scores to get the
        # largest scores first, then keep the first k positions.
        top_indices = np.argsort(-scores)[:k]
        return top_indices, scores[top_indices]


class TfidfRetriever(BaseRetriever):
    """Classic keyword search with TF-IDF vectors and cosine similarity.

    TF  (term frequency)        : how often a word appears in a passage
    IDF (inverse doc frequency) : rare words (e.g. "dizziness") count more than
                                  common words (e.g. "patients")
    """

    name = "TF-IDF"

    def fit(self, texts):
        # stop_words="english" removes words like "the", "and", "of".
        # sublinear_tf=True uses 1 + log(tf) so a word repeated 10 times does
        # not count 10 times more than a word seen once.
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        # Each row of this sparse matrix is one passage. Rows already have
        # length 1, so a dot product between two rows equals cosine similarity.
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def score(self, query):
        query_vector = self.vectorizer.transform([query])  # 1 x vocabulary
        # (passages x vocab) @ (vocab x 1) -> one cosine score per passage
        scores = self.matrix @ query_vector.T
        return scores.toarray().ravel()  # sparse -> flat numpy array


class BM25Retriever(BaseRetriever):
    """BM25: the standard keyword ranking formula used by search engines.

    For every query word t found in passage d:
        score += IDF(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg_len))
    where f = how many times t appears in d.
        k1 controls how fast repeated words stop adding score (saturation)
        b  controls how much long passages are penalised
    """

    name = "BM25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, texts):
        # CountVectorizer gives raw word counts (not TF-IDF weights).
        self.vectorizer = CountVectorizer(stop_words="english")
        counts = self.vectorizer.fit_transform(texts)  # passages x vocabulary

        n_passages = counts.shape[0]
        # Length of each passage = total number of (non-stop) words in it.
        self.passage_lengths = np.asarray(counts.sum(axis=1)).ravel()
        self.average_length = self.passage_lengths.mean()

        # Document frequency: in how many passages does each word appear?
        doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
        # BM25 IDF formula (the +1 inside the log keeps it positive).
        self.idf = np.log((n_passages - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        # "csc" format stores the matrix column by column, which makes it fast
        # to fetch all passages that contain a given word.
        self.counts = counts.tocsc()
        return self

    def score(self, query):
        scores = np.zeros(self.counts.shape[0])  # start every passage at 0
        # .indices gives the vocabulary numbers of the words in the query.
        query_words = self.vectorizer.transform([query]).indices
        for word in query_words:
            column = self.counts[:, word]  # counts of this word in all passages
            rows = column.indices  # passages that contain the word
            freq = column.data  # how many times it appears in each of them
            length_ratio = self.passage_lengths[rows] / self.average_length
            denominator = freq + self.k1 * (1 - self.b + self.b * length_ratio)
            scores[rows] += self.idf[word] * freq * (self.k1 + 1) / denominator
        return scores


class DenseLSARetriever(BaseRetriever):
    """A lightweight "dense" retriever using LSA (Latent Semantic Analysis).

    Real dense retrievers use neural sentence embeddings, which need PyTorch and
    are heavy for free Streamlit hosting. LSA compresses TF-IDF vectors into
    a few hundred numbers per passage, so passages that use *related* words
    (not only identical words) end up close together. It is a fair, CPU-only
    stand-in for embedding search.
    """

    name = "Dense (LSA)"

    def __init__(self, n_components=200):
        self.n_components = n_components  # length of each dense vector

    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        tfidf = self.vectorizer.fit_transform(texts)
        # random_state fixes the randomness so results are repeatable.
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=0)
        # fit_transform learns the compression and applies it to all passages,
        # then normalize() makes every vector length 1 for cosine similarity.
        self.vectors = normalize(self.svd.fit_transform(tfidf))
        return self

    def score(self, query):
        query_vector = normalize(self.svd.transform(self.vectorizer.transform([query])))
        return self.vectors @ query_vector.ravel()  # cosine with every passage


# A dictionary so the app can create a retriever from its name.
RETRIEVERS = {
    "BM25": BM25Retriever,
    "TF-IDF": TfidfRetriever,
    "Dense (LSA)": DenseLSARetriever,
}
