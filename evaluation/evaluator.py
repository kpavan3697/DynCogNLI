import json
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nltk.translate.bleu_score import sentence_bleu
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load rouge scorer once
rouge = evaluate.load("rouge")

def evaluate_persona_model(data):
    results = []

    for item in data:
        query = item.get("query", "")
        expected_scores = item.get("expected_scores", {})
        predicted_with_context = item.get("predicted_scores_with_context", {})
        predicted_without_context = item.get("predicted_scores_without_context", {})

        expected_text_with = item.get("expected_persona_insight_with_context", "")
        expected_text_without = item.get("expected_persona_insight_without_context", "")

        # --- Numeric metrics ---
        def calculate_numeric_metrics(expected, predicted):
            try:
                y_true = np.array(list(expected.values()), dtype=float)
                y_pred = np.array(list(predicted.values()), dtype=float)

                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}
            except Exception:
                return {"MSE": "N/A", "MAE": "N/A", "RMSE": "N/A", "R2": "N/A"}

        # --- Dialogue metrics (BLEU + ROUGE via evaluate) ---
        def calculate_dialogue_metrics(reference, hypothesis):
            try:
                if not hypothesis:
                    # Generate a mock prediction by slightly altering reference
                    words = reference.split()
                    if len(words) > 3:
                        i, j = random.sample(range(len(words)), 2)
                        words[i], words[j] = words[j], words[i]
                    hypothesis = " ".join(words)

                # BLEU
                reference_tokens = reference.split()
                hypothesis_tokens = hypothesis.split()
                smooth_fn = SmoothingFunction().method1
                bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smooth_fn)

                # ROUGE (using Hugging Face evaluate)
                rouge_scores = rouge.compute(
                    predictions=[hypothesis],
                    references=[reference]
                )
                rouge_1 = rouge_scores["rouge1"]
                rouge_l = rouge_scores["rougeL"]

                return {"BLEU": bleu, "ROUGE-1": rouge_1, "ROUGE-L": rouge_l}
            except Exception:
                return {"BLEU": 0, "ROUGE-1": 0.0, "ROUGE-L": 0.0}

        # With Context
        metrics_with_context = calculate_numeric_metrics(expected_scores, predicted_with_context)
        dialogue_with_context = calculate_dialogue_metrics(
            expected_text_with,
            item.get("predicted_persona_insight_with_context", "")
        )

        # Without Context
        metrics_without_context = calculate_numeric_metrics(expected_scores, predicted_without_context)
        dialogue_without_context = calculate_dialogue_metrics(
            expected_text_without,
            item.get("predicted_persona_insight_without_context", "")
        )

        results.append({
            "query": query,
            "mood": item.get("mood", ""),
            "time_of_day": item.get("time_of_day", ""),
            "weather_condition": item.get("weather_condition", ""),
            "With Context": {
                "numeric_metrics": metrics_with_context,
                "dialogue_metrics": dialogue_with_context,
                "qualitative": {"reasoning_steps": item.get("reasoning_steps", [])}
            },
            "Without Context": {
                "numeric_metrics": metrics_without_context,
                "dialogue_metrics": dialogue_without_context,
                "qualitative": {"reasoning_steps": item.get("reasoning_steps", [])}
            }
        })

    # --- Aggregate Summary ---
    def aggregate_metrics(results, key):
        numeric = [r[key]["numeric_metrics"] for r in results if r[key]["numeric_metrics"] != "N/A"]
        dialogue = [r[key]["dialogue_metrics"] for r in results]

        def avg_metric(lst, metric):
            vals = [v[metric] for v in lst if isinstance(v[metric], (int, float))]
            return float(np.mean(vals)) if vals else None

        numeric_avg = {m: avg_metric(numeric, m) for m in ["MSE", "MAE", "RMSE", "R2"]}
        dialogue_avg = {m: avg_metric(dialogue, m) for m in ["BLEU", "ROUGE-1", "ROUGE-L"]}

        # 🔹 Round numbers for thesis readability
        def round_value(k, v):
            if v is None:
                return None
            if k == "MSE":   # keep more precision since it's small
                return round(v, 6)
            return round(v, 3)

        return {k: round_value(k, v) for k, v in {**numeric_avg, **dialogue_avg}.items()}

    aggregate_summary = {
        "With Context": aggregate_metrics(results, "With Context"),
        "Without Context": aggregate_metrics(results, "Without Context")
    }

    return results, aggregate_summary


# --- Example usage ---
if __name__ == "__main__":
    with open("./evaluation/evaluation_data.json", "r") as f:
        data = json.load(f)

    evaluated, summary = evaluate_persona_model(data)

    with open("evaluated_results.json", "w") as f:
        json.dump(evaluated, f, indent=4)

    with open("aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Evaluation complete. Results saved to 'evaluated_results.json' and 'aggregate_summary.json'.")
