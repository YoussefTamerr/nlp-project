import requests
from sentence_transformers import util
from langchain_huggingface import HuggingFaceEmbeddings
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')
from datasets import load_from_disk
import pandas as pd
from datasets import Dataset

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

def compute_f1(pred, gt):
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_rouge_l(pred, gt):
    scores = scorer.score(gt, pred)
    return scores['rougeL'].fmeasure

def compute_bleu(pred, gt):
    reference = [gt.lower().split()]
    candidate = pred.lower().split()
    return sentence_bleu(reference, candidate, smoothing_function=smoothie)

def evaluate_models(api_url, dataset, model1, model2):
    results = []

    c = 0
    for item in dataset:
        question = item['question']['text']
        ground_truth = item['answers'][0]['text']

        ## first model
        response = requests.get(api_url, params={
            "prompt": question,
            "model": model1
        })

        if response.status_code != 200:
            print(f"Error from API for model {model1}: {response.text}")
            continue

        data = response.json()
        predicted_answer_1 = data["response"]

        # Compute all metrics
        f1_1 = compute_f1(predicted_answer_1, ground_truth)
        rouge_l_1 = compute_rouge_l(predicted_answer_1, ground_truth)

        print(c, model1)
        c += 1

        ## second model
        response = requests.get(api_url, params={
            "prompt": question,
            "model": model2
        })

        if response.status_code != 200:
            print(f"Error from API for model {model2}: {response.text}")
            continue

        data = response.json()
        predicted_answer_2 = data["response"]

        # Compute all metrics
        f1_2 = compute_f1(predicted_answer_2, ground_truth)
        rouge_l_2 = compute_rouge_l(predicted_answer_2, ground_truth)

        print(c, model2)
        c += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction llama": predicted_answer_1,
            "prediction gemma": predicted_answer_2,
            "f1 llama": f1_1,
            "rougeL llama": rouge_l_1,
            "f1 gemma": f1_2,
            "rougeL gemma": rouge_l_2,
        })

    return results

def main():
    samples = load_from_disk("narrativeqa_1000")
    # get 30 random rows
    samples = samples.to_pandas()
    samples = samples.sample(n=30, random_state=42)
    samples = Dataset.from_pandas(samples)
    # ds_train_context = [row['document']['summary']['text'] for row in samples]
    # ds_train_question = [row['question']['text'] for row in samples]
    # ds_train_answer = [row['answers'][0]['text'] for row in samples]


    api_url = "http://localhost:8000/chat"
    model1 = "llama"
    model2 = "gemma"
    results = evaluate_models(api_url, samples, model1, model2)
        
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv("model_evaluation_results.csv", index=False)

if __name__ == "__main__":
    main()



