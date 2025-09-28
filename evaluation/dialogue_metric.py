import json
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Path to your current dataset
dataset_path = "evaluation/evaluation_results.json"

# Load dataset
with open(dataset_path, "r") as f:
    data = json.load(f)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

# Function to compute dialogue metrics
def compute_dialogue_metrics(generated: str, reference: str):
    # BLEU
    bleu = sentence_bleu([reference.split()], generated.split())
    # ROUGE
    rouge_scores = scorer.score(reference, generated)
    rouge1 = rouge_scores["rouge1"].fmeasure
    rougel = rouge_scores["rougeL"].fmeasure
    return {"BLEU": bleu, "ROUGE-1": rouge1, "ROUGE-L": rougel}

# Update dataset
for case in data:
    for context_type in ["With Context", "Without Context"]:
        generated = case[context_type]["qualitative"].get("generated_insight", "")
        # If 'generated_insight' does not exist, fallback to reference
        reference = generated if generated else generated
        # Here, you can replace 'generated' with your model's actual output
        dialogue_metrics = compute_dialogue_metrics(
            generated=generated,
            reference=case[context_type].get("qualitative", {}).get("reasoning_steps", [""])[-1]
        )
        case[context_type]["dialogue_metrics"] = dialogue_metrics

# Save updated dataset
updated_path = "evaluation/evaluation_results_updated.json"
with open(updated_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated dataset saved at: {updated_path}")
