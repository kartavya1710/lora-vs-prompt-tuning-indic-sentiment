# LoRA vs Prompt-Tuning for Low-Resource Sentiment Classification

Full fine-tuning, LoRA, prompt-tuning and a linear probe on **IndicBERT v2** for **Gujarati** and **Marathi** binary sentiment classification (IndicSentiment), with training sets from 16 to 500 examples and 3 seeds (180 runs, plus 36 runs of the Prompt2 variant).

**Authors:** Bhavya Janak Shah (PRN 26070149031), Kartavya Divyakant Master (PRN 26070149027)
**Affiliation:** Symbiosis Institute of Technology, Pune Campus, Symbiosis International (Deemed University), Pune, India
**Programme:** M.Tech, Artificial Intelligence and Machine Learning (2026–28)

---

## Key results (test macro-F1 with 500 training examples, mean of 3 seeds)

| Method | Method-specific trainable params | Gujarati | Marathi | Train time (s) | Peak GPU memory (MB) |
|---|---|---|---|---|---|
| Full fine-tuning | 278,042,882 | 0.957 | 0.969 | 54.4 | 5,361 |
| LoRA (r = 8) | 294,912 | 0.956 | 0.964 | 15.6 | 1,829 |
| Prompt (20 tokens) | 15,360 | 0.782 | 0.777 | 14.9 | 1,829 |
| Prompt2 (separate head LR) | 15,360 | 0.759 | 0.940 | 15.0 | 1,829 |
| Linear probe | 0 | 0.853 | 0.881 | 6.2 | 1,267 |

LoRA matches full fine-tuning while training 0.106% of the parameters, about 3.5x faster and with about a third of the GPU memory. Prompt-tuning is unstable at this model size: several runs collapse to chance level.

---

## Repository structure

```
.
├── notebooks/
│   └── lora_vs_prompt_tuning_indic_sentiment.ipynb   # all code: data, training, analysis, model saving
├── models/
│   ├── lora/      # trained LoRA models (Gujarati, Marathi; 500 examples, seed 0)
│   └── prompt/    # trained Prompt2 models (Gujarati, Marathi; 500 examples, seed 0)
├── results/       # results.csv (every run), summary tables, chosen learning rates, figures
├── report/        # project report (.docx)
├── requirements.txt
└── README.md
```

## Dataset

- **IndicSentiment** (AI4Bharat), part of IndicXTREME: https://huggingface.co/datasets/ai4bharat/IndicSentiment
- The notebook downloads the Gujarati and Marathi JSON files automatically with `huggingface_hub`, so no data is stored in this repo.
- Splits: 2 unlabelled test rows are removed (998 left). A stratified split with seed 42 gives a 500-review **training pool** and a 498-review **test set**. The official validation set (156 reviews) is the **dev set**, used only for the learning-rate search.

## Model and methods

- Encoder: [`ai4bharat/IndicBERTv2-mlm-only`](https://huggingface.co/ai4bharat/IndicBERTv2-mlm-only) (278M parameters) with a 2-class head
- `full`: every weight is trained
- `lora`: rank-8 adapters (alpha 16, dropout 0.1) on attention `query` and `value`
- `prompt`: 20 trainable virtual tokens, one learning rate for prompt and head
- `prompt2`: same as `prompt`, but the head has its own fixed learning rate (1e-3)
- `probe`: frozen encoder
- In every method except `full`, the pooler and classifier (592,130 parameters) are also trained.

## Dependencies

Python 3.10+, PyTorch, Transformers, PEFT, Accelerate, huggingface_hub, safetensors, sentencepiece, scikit-learn, SciPy, NumPy, pandas, Matplotlib. See `requirements.txt`. The notebook's first cell installs everything in Colab.

## Setup

### Google Colab (recommended, free T4 GPU)
1. Open Colab → **File → Open notebook → GitHub**, paste this repository's URL and open `notebooks/lora_vs_prompt_tuning_indic_sentiment.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Allow Google Drive access when asked. Results are saved to `MyDrive/lora_vs_prompt_sentiment/` so a disconnect does not lose them.

### Local machine (NVIDIA GPU with CUDA)
```bash
git clone https://github.com/kartavya1710/lora-vs-prompt-tuning-indic-sentiment.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/lora_vs_prompt_tuning_indic_sentiment.ipynb
```
Outside Colab, results are written to `./outputs`.

## Steps to execute

Run the notebook from top to bottom:

1. **Setup**: install packages, imports, configuration (`SMOKE_TEST = True` first does a tiny check run).
2. **Data**: download IndicSentiment, check that Gujarati and Marathi are aligned, make the seed-42 splits.
3. **Tokenisation**: IndicBERT v2 tokenizer, max length 128.
4. **Models and training code**: `build_model` and `train_run` (AdamW, weight decay 0.01, linear schedule with 10% warm-up, gradient clipping 1.0, batch 16, fp16).
5. **Part 1–2**: parameter-count check, then the real configuration (`SMOKE_TEST = False`).
6. **Part 3**: learning-rate search on the dev set (128 examples, seed 0) → `best_lr.json`.
7. **Part 4**: main experiment, 4 methods × 6 sizes × 3 seeds × 2 languages, resumable → `results.csv`.
8. **Part 5**: tables, learning curves, retention, AULC, cost and Welch t-tests.
9. **Cells A–C / Cell 1–2**: Prompt2 variant, train/test overlap checks, final analysis with medians.
10. **Section 9**: trains and saves the example models in `models/`.

The full experiment takes a few hours on a free T4.

## Using the saved models

Each model folder contains:

| File | What it is |
|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | PEFT adapter (LoRA or prompt tokens) |
| `trained_weights.safetensors` | every trained tensor: adapter or prompt + pooler + classifier |
| `run_info.json` | language, method, learning rate, steps and test scores |

Load a model (run the notebook's setup, data and tokenisation cells, plus `build_peft` from Section 9):

```python
from safetensors.torch import load_file
model = build_peft("lora")      # or "prompt2"
model.load_state_dict(load_file("models/lora/lora_gu_500_seed0/trained_weights.safetensors"), strict=False)
model.to(DEVICE).eval()
acc, f1 = evaluate(model, ENC[("indicbert2", "gu", "test")], TOK["indicbert2"])
```

Fully fine-tuned models are about 1.1 GB each, which is over GitHub's file limit, so they are not included. They can be recreated with the notebook.

## Reproducibility

Split seed 42; training seeds 0, 1, 2. Scores can vary slightly between GPU types and library versions.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning*, EMNLP 2021
- Doddapaneni et al., *Towards Leaving No Indic Language Behind* (IndicBERT v2, IndicXTREME), ACL 2023
