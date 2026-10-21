"""Loading the PubMedQA dataset and building the retrieval corpus.

PubMedQA (https://pubmedqa.github.io) has 1,000 expert-labelled questions.
Each question comes from one PubMed abstract and has:
    - QUESTION        : the research question (usually the paper title)
    - CONTEXTS        : the abstract split into parts (background, methods, ...)
    - final_decision  : the correct answer, one of "yes", "no", "maybe"
"""

import json  # reads the .json dataset file
import os  # builds file paths that work on Windows, Linux and Mac
import random  # shuffles questions with a fixed seed so splits are repeatable
import urllib.request  # downloads the dataset if the file is missing

# Where the official PubMedQA file lives on GitHub (used only as a fallback).
PUBMEDQA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"

# __file__ is this file (uarmed/data.py). Going up two folders gives the project
# root, and from there we point to data/ori_pqal.json.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ori_pqal.json")


def load_pubmedqa(path=DEFAULT_DATA_PATH):
    """Read PubMedQA and return a list of simple dictionaries, one per question."""
    # If the file is not on disk yet, download it once and save it.
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # create data/ folder
        urllib.request.urlretrieve(PUBMEDQA_URL, path)  # download the file

    # Open the file and parse the JSON text into a Python dictionary.
    # The dictionary looks like {"21645374": {"QUESTION": ..., ...}, ...}
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)

    questions = []  # we will fill this list
    for pmid, item in raw.items():  # pmid = PubMed ID of the source paper
        questions.append(
            {
                "pmid": pmid,
                "question": item["QUESTION"],
                "contexts": item["CONTEXTS"],  # list of abstract parts
                "answer": item["final_decision"],  # "yes" / "no" / "maybe"
                "long_answer": item["LONG_ANSWER"],  # the paper's conclusion
            }
        )
    return questions


def build_corpus(questions):
    """Turn every abstract part of every question into one searchable passage.

    All 1,000 abstracts are mixed into ONE big pool (about 3,300 passages), so
    the retriever has to find the right passages among many wrong ones.
    """
    passages = []
    for q in questions:
        for position, text in enumerate(q["contexts"]):
            passages.append(
                {
                    "pmid": q["pmid"],  # which paper this passage came from
                    "position": position,  # 0 = first part of the abstract
                    "text": text,
                }
            )
    return passages


def gold_passage_ids(passages):
    """Map each PubMed ID to the passage numbers that belong to it.

    These are the "correct" (gold) passages for that question. We use them only
    for measuring retrieval quality (Recall@k, MRR), never for answering.
    Example result: {"21645374": {0, 1, 2}, "16418930": {3, 4, 5, 6}, ...}
    """
    gold = {}
    for index, passage in enumerate(passages):
        # setdefault creates an empty set the first time we see this pmid.
        gold.setdefault(passage["pmid"], set()).add(index)
    return gold


def split_questions(questions, train_fraction=0.5, val_fraction=0.2, seed=42):
    """Split questions into train / validation / test sets.

    train      -> teaches the answer model and calibrates the score scales
    validation -> used to tune the UAR-Med weights and threshold
    test       -> touched only at the very end to report final results
    """
    shuffled = list(questions)  # copy, so the original order is not changed
    random.Random(seed).shuffle(shuffled)  # same seed -> same split every run

    n_train = int(len(shuffled) * train_fraction)  # 1000 * 0.5 = 500
    n_val = int(len(shuffled) * val_fraction)  # 1000 * 0.2 = 200

    train = shuffled[:n_train]  # first 500
    val = shuffled[n_train : n_train + n_val]  # next 200
    test = shuffled[n_train + n_val :]  # remaining 300
    return train, val, test
