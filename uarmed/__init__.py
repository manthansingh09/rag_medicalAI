"""UAR-Med: Uncertainty-Guided Adaptive Retrieval for medical question answering.

This package is split into small files so each idea lives in one place:

    data.py         -> load PubMedQA and turn it into a searchable corpus
    retrievers.py   -> three search engines (TF-IDF, BM25, LSA "dense")
    answerer.py     -> a small yes/no/maybe classifier (no API, no GPU)
    uncertainty.py  -> the R, A, C signals and the uncertainty score U
    policies.py     -> HOW MANY passages to retrieve (baselines + UAR-Med)
    evaluation.py   -> metrics and the weight-tuning grid search
    pipeline.py     -> glues everything together into one "system" object
"""
