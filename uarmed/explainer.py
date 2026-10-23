"""Write a readable answer from the retrieved passages (no API, no LLM).

This is EXTRACTIVE answering: instead of inventing new text, we pick the
sentences from the evidence that best answer the question and join them,
each with a citation number like [1] that points to its source passage.
Extractive answers are a safe choice in medicine because every sentence can
be traced back to a real paper.

Steps:
    1. Split every retrieved passage into sentences.
    2. Score each sentence by TF-IDF similarity to the question, plus a small
       bonus for "answer-like" wording (e.g. "suggest", "is a"), and a penalty
       for "aim" sentences that only repeat the question.
    3. Take the best sentences, skipping near-duplicates and using at most
       two sentences per passage so the answer draws on several sources.
    4. Put them back in reading order so the paragraph flows.
"""

import re  # regular expressions, used to split text into sentences

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Questions starting with these words can be answered with yes / no / maybe.
YES_NO_STARTS = {
    "is", "are", "was", "were", "does", "do", "did", "can", "could", "should",
    "will", "would", "has", "have", "had", "may", "might", "must",
}

# Words that often appear in sentences that state a finding or a definition.
ANSWER_CUES = (
    "conclusion", "suggest", "indicate", "demonstrate", "show", "found",
    "associated", "effective", "significant", "is a ", "is an ", "are ",
)

# Sentences starting like this state the study's goal ("To investigate whether
# X affects Y"). They look very similar to the question but do not answer it.
AIM_STARTS = (
    "to investigate", "to evaluate", "to determine", "to assess", "to examine",
    "to compare", "to study", "the aim", "the purpose", "the objective",
    "we aimed", "this study aimed", "we investigated", "we sought",
)


def is_yes_no_question(question):
    """True for questions like "Does X cause Y?", False for "What is X?".

    Paper titles often put the question after a colon, e.g.
    "Vanishing testes syndrome: is surgical management necessary?", so we
    check the start of every part split at ":", ";" or "--".
    """
    for part in re.split(r"[:;]|--", question.lower()):
        words = part.split()
        if words and words[0] in YES_NO_STARTS:
            return True
    return False


def split_sentences(text):
    """Split text after ". ", "? " or "! " when the next sentence starts with
    a capital letter, a digit or a bracket (so "e.g. aspirin" is not split)."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def written_answer(question, passage_ids, passages, max_sentences=4, max_per_source=2,
                   min_score=0.1):
    """Return the chosen sentences as dicts: {"sentence", "source", "pmid"}.

    "source" is the 1-based position of the passage in the evidence list, so
    the app can show it as a citation number [source].
    """
    # 1) Collect candidate sentences from every retrieved passage.
    candidates = []
    for source, passage_id in enumerate(passage_ids, start=1):
        for sentence in split_sentences(passages[passage_id]["text"]):
            if len(sentence.split()) >= 6:  # skip fragments like "Methods."
                candidates.append(
                    {"sentence": sentence, "source": source, "pmid": passages[passage_id]["pmid"]}
                )
    if not candidates:
        return []

    # 2) Score them. A small vectorizer fitted on just these sentences is
    #    enough, and it works for any corpus (also live PubMed results).
    sentences = [c["sentence"] for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english").fit([question] + sentences)
    sentence_vectors = vectorizer.transform(sentences)
    question_vector = vectorizer.transform([question])
    scores = (sentence_vectors @ question_vector.T).toarray().ravel()  # cosine
    for i, sentence in enumerate(sentences):
        lower = sentence.lower()
        if any(cue in lower for cue in ANSWER_CUES):
            scores[i] += 0.05  # small bonus, so similarity still matters most
        if lower.startswith(AIM_STARTS):
            scores[i] *= 0.3  # keep it only if nothing better exists

    # 3) Greedily take the best sentences, skipping near-duplicates.
    chosen = []
    used_per_source = {}  # source number -> how many sentences taken from it
    for i in np.argsort(-scores):  # best score first
        if len(chosen) == max_sentences or scores[i] < min_score:
            break  # enough sentences, or the rest barely relate to the question
        source = candidates[i]["source"]
        if used_per_source.get(source, 0) == max_per_source:
            continue  # already took enough from this passage
        similarity_to_chosen = [
            (sentence_vectors[i] @ sentence_vectors[j].T).toarray()[0, 0] for j in chosen
        ]
        if any(sim > 0.5 for sim in similarity_to_chosen):
            continue  # says nearly the same as a sentence we already have
        chosen.append(i)
        used_per_source[source] = used_per_source.get(source, 0) + 1

    # 4) Candidates were collected in reading order, so sorting the chosen
    #    positions restores that order.
    return [candidates[i] for i in sorted(chosen)]
