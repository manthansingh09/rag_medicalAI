"""Safety checks that decide whether the system should answer at all.

A medical question-answering system must know when to say "I can't answer
that". Saying "YES" to "Do I have cancer?" is dangerous, even if the
retrieved papers are about cancer. We check three things:

    1. Is it a PERSONAL question?  ("Do I have...", "Should I...", "my ...")
       Research papers describe groups of patients; they cannot diagnose one
       person. -> do not answer, suggest asking a doctor.
    2. Is the topic actually IN the evidence?  If the question's key words do
       not appear in the retrieved passages, the database does not cover it.
       -> say "not found in the database".
    3. Is the evidence STRONG and the model CONFIDENT enough for a yes/no
       verdict? -> if relevance R < MIN_RELEVANCE or the model's probability
       < MIN_VERDICT_CONFIDENCE, we say the evidence is unclear.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # "the", "is", ...

# Share of the question's key words that must appear in the evidence.
MIN_TOPIC_COVERAGE = 0.6
# Relevance R below this means the passages barely match the question, so no
# yes/no verdict is given, however confident the answer model looks.
MIN_RELEVANCE = 0.3
# Probability the answer model needs before we show a yes/no verdict. On the
# validation questions, verdicts above 0.5 were right 56% of the time, versus
# 48% for all verdicts, and about 39% of questions passed this bar.
MIN_VERDICT_CONFIDENCE = 0.5

# Patterns that show the question is about the user personally.
PERSONAL_PATTERNS = [
    r"\b(do|did|does|could|can|might|may|will|would|should|am|was|have) i\b",
    r"\bi (have|had|am|feel|felt|got|think|take|took|need)\b",
    r"\bi'm\b", r"\bmy\b", r"\bme\b", r"\bmyself\b",
]

# Words that appear in almost any medical question and say nothing about the
# topic. They are ignored when checking whether the topic is in the evidence.
GENERIC_WORDS = {
    "use", "used", "uses", "using", "effect", "effects", "affect", "affects",
    "cause", "causes", "help", "helps", "good", "bad", "people", "patient",
    "patients", "study", "studies", "risk", "increase", "increases", "reduce",
    "reduces", "treatment", "treat", "work", "works", "safe", "better",
    # question words that the stop-word list does not include
    "does", "did", "role", "play", "plays", "true", "really",
}


def is_personal_question(question):
    """True for questions about the user, e.g. "Do I have cancer?"."""
    text = question.lower()
    return any(re.search(pattern, text) for pattern in PERSONAL_PATTERNS)


def key_terms(question):
    """The topic words of a question: no stop words, no generic words."""
    words = re.findall(r"[a-z][a-z0-9-]{2,}", question.lower())  # words of 3+ letters
    return [w for w in words if w not in ENGLISH_STOP_WORDS and w not in GENERIC_WORDS]


def term_found(term, evidence_words):
    """Match the first 5 letters, so "diabetes" also finds "diabetic"."""
    if len(term) <= 5:
        return term in evidence_words
    return any(word.startswith(term[:5]) for word in evidence_words)


def topic_coverage(question, passage_texts):
    """Return (share of key terms found in the evidence, list of missing terms)."""
    terms = key_terms(question)
    if not terms:
        return 1.0, []  # nothing specific to check
    evidence_words = set(re.findall(r"[a-z][a-z0-9-]+", " ".join(passage_texts).lower()))
    missing = [t for t in terms if not term_found(t, evidence_words)]
    return 1 - len(missing) / len(terms), missing
