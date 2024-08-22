from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from openai import OpenAI
from typing import List, Dict


class LanguageEvaluation:
    """
    Class to evaluate language generation using BLEU, ROUGE, and CIDEr metrics.
    """

    @staticmethod
    def evaluate(predictions: List[str], gts: List[str]) -> Dict[str, float]:
        evals = {}
        gt = {}
        prediction = {}

        # Process predictions and ground truth into the required format
        for i, (pred, ref) in enumerate(zip(predictions, gts)):
            gt[str(i)] = [ref.lower()]
            prediction[str(i)] = [pred.lower()]

        # Set up scorers
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        # Compute scores
        for scorer, method in scorers:
            score, _ = scorer.compute_score(gt, prediction)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    evals[m] = sc
            else:
                evals[method] = score

        return evals


class GPTEvaluation:
    """
    Class to evaluate predictions using OpenAI's GPT model for human-like scoring.
    """

    def __init__(self):
        self.client = OpenAI(api_key="openai_api_key")

    def evaluate(self, prediction: str, gt: str) -> int:
        try:
            gpt_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.6,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an evaluator who rates my answer based on the correct answer. "
                        "Rate my answer based on the correct answer out of 100, with higher scores indicating "
                        "that the answer is closer to the correct answer, and you should be accurate to single digits "
                        "like 62, 78, 41, etc. Only output the number.",
                    },
                    {
                        "role": "user",
                        "content": f"'This is the correct answer:{gt}, This is my answer:{prediction}'",
                    },
                ],
            )
            return int(gpt_response.choices[0].message.content.strip())
        except ValueError:
            raise ValueError("GPT returned a non-integer response.")
