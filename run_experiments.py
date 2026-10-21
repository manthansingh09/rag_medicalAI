"""Run the full UAR-Med experiment from the command line (or Google Colab).

Usage:
    python run_experiments.py                  # BM25 retriever (default)
    python run_experiments.py --retriever TF-IDF
    python run_experiments.py --all            # all three retrievers
    python run_experiments.py --answerer "Flan-T5 (LLM)"   # needs requirements-llm.txt

Results are printed and saved to results/<retriever>_<answerer>.csv
"""

import argparse  # reads options typed after the script name
import os

import pandas as pd

from uarmed.pipeline import ANSWERERS, build_system, run_comparison, tune_on_validation
from uarmed.retrievers import RETRIEVERS


def run(retriever_name, answerer_name, llm_name, start_k, step, budget, cost_weight):
    print(f"\n=== Retriever: {retriever_name} | Answer model: {answerer_name} ===")
    system = build_system(retriever_name, answerer_name, llm_name=llm_name)
    print(f"{len(system.passages)} passages | train {len(system.train)} | "
          f"val {len(system.val)} | test {len(system.test)}")

    # Step 1: tune on validation data (never on test data).
    best_uar, best_threshold = tune_on_validation(system, start_k, step, budget, cost_weight)
    print("Tuned UAR-Med:", best_uar)
    print("Tuned threshold baseline:", best_threshold)

    # Step 2: evaluate everything once on the test set.
    summaries, curve = run_comparison(
        system, best_uar["weights"], best_uar["tau"], best_threshold["tau"],
        start_k, step, budget,
    )
    table = pd.DataFrame(summaries)
    print(table.round(3).to_string(index=False))
    print("\nFixed top-k curve:")
    print(pd.DataFrame(curve)[["policy", "accuracy", "recall@k", "avg passages"]]
          .round(3).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    # e.g. "Dense (LSA)" + "Classifier" -> "Dense_LSA_Classifier"
    file_name = f"{retriever_name}_{answerer_name}".replace(" ", "_").replace("(", "").replace(")", "")
    table.to_csv(os.path.join("results", f"{file_name}.csv"), index=False)


if __name__ == "__main__":  # only runs when the file is executed directly
    parser = argparse.ArgumentParser(description="UAR-Med experiments")
    parser.add_argument("--retriever", default="BM25", choices=list(RETRIEVERS))
    parser.add_argument("--all", action="store_true", help="run every retriever")
    # ANSWERERS contains "Flan-T5 (LLM)" only when torch + transformers are installed.
    parser.add_argument("--answerer", default="Classifier", choices=list(ANSWERERS))
    parser.add_argument("--llm", default=None,
                        help="Hugging Face model name or folder, e.g. google/flan-t5-large")
    parser.add_argument("--start-k", type=int, default=3)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    args = parser.parse_args()

    names = list(RETRIEVERS) if args.all else [args.retriever]
    for name in names:
        run(name, args.answerer, args.llm, args.start_k, args.step, args.budget, args.cost_weight)
