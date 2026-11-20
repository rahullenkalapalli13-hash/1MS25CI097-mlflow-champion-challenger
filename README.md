# Champion–Challenger Model Comparison: Sentiment Classification (Hugging Face + MLflow)

**USN:** <1MS25CI097> · **Course:** Skill Lab (MLOps with MLflow), Ramaiah Institute of Technology

## 1. Task and models
- **Task:** Binary sentiment classification (NEGATIVE = 0, POSITIVE = 1)
- **Dataset:** GLUE SST-2 *validation* split, random subset of **200** sentences (seed 42), saved in `data/eval_inputs.csv`
- **Model A (Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters)
- **Model B (Challenger):** `siebert/sentiment-roberta-large-english` (~355M parameters)
- Both models are run on the identical, identically ordered inputs on CPU.

## 2. How to run
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Terminal 1: MLflow server (port 5001 because macOS AirPlay uses 5000)
mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

# Terminal 2 (same project folder, venv active)
N_SAMPLES=10 python evaluate.py     # optional quick debug run
python evaluate.py                  # full run: evaluates, logs, decides winner
python register_champion.py         # registers both models, sets aliases
python predict_from_mlflow.py --both  # loads @champion / @challenger, predicts new data
```

## 3. Champion rule (written BEFORE the final run, not changed afterwards)
- **Primary metric:** macro-F1 on the 200-sample evaluation set. **Higher is better.**
- **Minimum improvement margin:** the Challenger must beat the Champion by **≥ 0.01 (1 percentage point)**. With 200 samples, one sentence is 0.5 points, so a smaller gap is within noise.
- **Guardrail:** reject the Challenger if its average latency is **> 10×** the Champion's.
- **Tie-breaker:** if the macro-F1 difference is smaller than the margin, the **faster** model (lower `avg_latency_ms`, lower is better) is the Champion.
- Other metrics (accuracy, precision, recall, F1): higher is better. `avg_latency_ms`, `total_inference_s`: lower is better.

## 4. Results
| Metric | A: DistilBERT | B: RoBERTa-large | Δ (B − A) |
|---|---|---|---|
| accuracy | 0.935 | 0.945 | +0.010 |
| precision | 0.9286 | 0.9375 | +0.0089 |
| recall | 0.9541 | 0.9633 | +0.0092 |
| f1 | 0.9412 | 0.9502 | +0.0090 |
| **macro_f1 (primary)** | 0.9343 | 0.9444 | **+0.0101** |
| avg_latency_ms | 7.63 | 40.58 | +32.95 (about 5.3× slower) |
| total_inference_s | 1.53 | 8.12 | +6.59 |

**Winner by the rule: Model B (RoBERTa-large)**, which is now `@champion` (version 2). DistilBERT is `@challenger` (version 1).

Screenshots are in `screenshots/`. Qualitative examples where the models disagree are in `outputs/disagreements.csv`.

## 5. Decision (150–250 words)
The Challenger, RoBERTa-large, does improve on the Champion, DistilBERT, but only slightly. Its macro-F1 was 0.9444 against 0.9343, a gain of 0.0101, which clears my pre-set 1-point margin by just 0.0001. Because the rule was fixed before the final run and the 10x latency guardrail was not exceeded, the decision follows the rule: RoBERTa-large is now the Champion (version 2) and DistilBERT is the Challenger (version 1).

However, I do not think this difference is strongly meaningful. The two models disagreed on only 18 of 200 sentences. RoBERTa was right on about 10 of them and DistilBERT on about 8, so the accuracy gap (0.945 vs 0.935) is just two sentences. On a sample this small, that could easily be noise.

The trade-offs are clear. RoBERTa-large is about 5.3 times slower (40.6 ms vs 7.6 ms per sentence) and roughly five times larger (355M vs 67M parameters). If speed or cost matters, DistilBERT is the better value. A larger evaluation set would be needed to confirm that RoBERTa's advantage is real.

**Stretch goal:** I moved the `champion` alias to version 1, reran `predict_from_mlflow.py` without editing it, and it loaded DistilBERT instead of RoBERTa (see `screenshots/stretch_flipped_alias.png`). I then restored the aliases to match the decision rule.

**Note:** The MLflow 3.18 UI alias popup showed an error, so I set the aliases with the `MlflowClient` Python API instead.

## 6. Limitations and declaration
- Only 200 test sentences: a 1-point difference is ~2 sentences and may be noise.
- SST-2 sentences are short movie-review snippets; results may not transfer to other domains. DistilBERT was fine-tuned on SST-2 train; the RoBERTa-large model was trained on many sentiment datasets, which may include SST-2.
- Latency was measured on my laptop CPU, one sample at a time, so it will differ on other hardware.
- **AI declaration:** I used an AI assistant (Claude) to help scaffold the code. I have read and can explain every line.
