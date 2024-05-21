import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# from openai import OpenAI
# CLIENT = OpenAI()


class LanguageEvaluation:

    @staticmethod
    def evaluate(predictions, gts):
        
        eval = {}
        gt = {}
        prediction = {}
        for i in range(len(predictions)):
            gt[str(i)] = [gts[i].lower()]
            prediction[str(i)] = [predictions[i].lower()]

        # =================================================
        # Set up scorers
        # =================================================
        # start_time = time.time()
        # gt  = tokenizer.tokenize(gt)
        # prediction = tokenizer.tokenize(prediction)
        # print(f"____Tokenizer took {str(time.time() - start_time)} seconds.____")

        # =================================================
        # Set up scorers
        # =================================================
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gt, prediction)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    eval[m] = sc
            else:
                eval[method] = score
        
        return eval

class GPTEvaluation:
    
    @staticmethod
    def evaluate(prediction, gt):
        gpt_response = CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.6,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluator who rates my answer based on the correct answer. "
                    "Rate my answer based on the correct answer out of 100, with higher scores indicating "
                    "that the answer is closer to the correct answer, and you should be accurate to single digits "
                    "like 62, 78, 41, etc. Only output the number."
                },
                {
                    "role": "user",
                    "content": f"'This is the correct answer:{gt}, This is my answer:{prediction}'",
                },
            ],
        )
        return gpt_response.choices[0].message.content
    

if __name__ == "__main__":
    g = "The ego vehicle is driving in the rain. It is night time. There are parked cars around."
    preds = {
    "same": "the ego vehicle is driving in the rain. it is night time. there are parked cars around.",
    }

    """gpt = GPTEvaluation()
    for key, val in preds.items():
        print(f"{key}: {gpt.evaluate(val, g)}")
    tk = PTBTokenizer()
    results = {}
    for key, val in preds.items():
        results[key] = LanguageEvaluation.evaluate(val, g, tokenizer=tk).copy()
        
    print(json.dumps(results, indent=4))"""
    
    results = LanguageEvaluation.evaluate(list(preds.values()), [g] * 1)
    print(results)
    