"""OPTIONAL: answer with a small open-source language model (Flan-T5).

This is a drop-in replacement for AnswerModel: it has the same methods
(fit, predict_proba, predict, passage_stances), so nothing else changes.

It needs extra packages that are too heavy for free Streamlit hosting, so use
it in Google Colab (free GPU) or on your own computer:
    pip install -r requirements-llm.txt

How it answers (no text generation needed):
    1. Build a prompt: question + retrieved passages + "Answer yes, no, or maybe."
    2. Run the model ONE step and read its score for the first word it would write.
    3. Compare only the scores of "yes", "no" and "maybe" and turn them into
       probabilities with softmax.
This "zero-shot" approach needs no training data at all.
"""

import importlib.util  # checks whether a package is installed without importing it

LABELS = ["yes", "no", "maybe"]


def llm_available():
    """True if torch and transformers are installed."""
    return all(importlib.util.find_spec(p) is not None for p in ("torch", "transformers"))


class LLMAnswerModel:
    name = "Flan-T5 (zero-shot LLM)"

    def __init__(self, model_name="google/flan-t5-base", max_context_words=350):
        # Imports are inside __init__ so the rest of the project still works
        # when these packages are not installed.
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        # Use the GPU if there is one (Colab), otherwise the CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()  # switch off training-only behaviour (dropout)
        self.max_context_words = max_context_words  # keeps prompts short enough

        # The vocabulary number of the first token of each answer word.
        self.label_token_ids = [
            self.tokenizer(label, add_special_tokens=False).input_ids[0] for label in LABELS
        ]
        # The same (question, passages) pair is asked many times during the
        # experiments, so we remember answers instead of recomputing them.
        self.cache = {}

    def fit(self, questions):
        return self  # zero-shot: nothing to train

    def build_prompt(self, question, passage_texts):
        words = " ".join(passage_texts).split()[: self.max_context_words]
        context = " ".join(words) if words else "(no context)"
        return (
            f"Read the medical abstract and answer the question.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer yes, no, or maybe."
        )

    def predict_proba(self, question, passage_texts):
        """Return a dict like {"yes": 0.7, "no": 0.2, "maybe": 0.1}."""
        key = (question, tuple(passage_texts))
        if key in self.cache:
            return self.cache[key]

        prompt = self.build_prompt(question, passage_texts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        # The decoder starts from its special "start" token; we ask for the
        # scores of the very first word it would produce.
        start = self.torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)

        with self.torch.no_grad():  # no gradients needed, saves memory and time
            logits = self.model(**inputs, decoder_input_ids=start).logits[0, -1]

        label_scores = logits[self.label_token_ids]  # keep only yes / no / maybe
        probabilities = self.torch.softmax(label_scores, dim=0).tolist()
        result = dict(zip(LABELS, probabilities))
        self.cache[key] = result
        return result

    def predict(self, question, passage_texts):
        probabilities = self.predict_proba(question, passage_texts)
        return max(probabilities, key=probabilities.get)

    def generate_answer(self, question, passage_texts, max_new_tokens=120):
        """Write a short free-text answer (used in the app's answer box)."""
        context = " ".join(" ".join(passage_texts).split()[: self.max_context_words])
        prompt = (
            "Using only the context, answer the question in two or three sentences.\n"
            f"Context: {context}\nQuestion: {question}"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def passage_stances(self, question, passage_texts):
        """Ask the model to answer from EACH passage alone (used for conflict C)."""
        return [self.predict(question, [text]) for text in passage_texts]
