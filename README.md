# Darija Script Detector

A fine-tuned XLM-RoBERTa model that classifies Algerian Arabic (Darija) text into three categories: **Darija written in Latin script**, **Darija written in Arabic script**, and **Not Darija**.

Built to handle the code-switching and script-mixing common in Algerian social media, chat, and speech transcripts, where the same language is written inconsistently across Arabic and Latin alphabets.

## Example

| Input | Prediction |
|---|---|
| `كي راك خويا؟` | Darija (Arabic script) |
| `wach rak mlih wla la?` | Darija (Latin script) |
| `How are you doing today?` | Not Darija |
| `Je suis très fatigué aujourd'hui.` | Not Darija |

## Results

Fine-tuned `xlm-roberta-base` for 3 epochs on a custom-labeled dataset.

| Metric | Score |
|---|---|
| Accuracy | 99.7% |
| F1 (macro) | 99.7% |
| Precision (macro) | 99.8% |
| Recall (macro) | 99.6% |

## Project structure

```
├── data/           # sample of the labeled dataset (full dataset not published)
├── notebooks/       # original development notebook
├── src/
│   ├── preprocess.py   # data loading, cleaning, tokenization
│   ├── train.py         # fine-tuning script
│   └── predict.py       # inference
├── examples/         # sample inputs/outputs
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Train
python src/train.py path/to/darija_dataset.csv

# Predict (automatically downloads the model from Hugging Face Hub on first run)
python src/predict.py "wach rak mlih?"
```

## Dataset

Custom dataset built and labeled across three classes: `darija_latin`, `darija_arabic`, `not_darija`. A sample is included under `data/`; the full dataset is not published.

## Model

Base model: [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base), fine-tuned for sequence classification.

Fine-tuned weights hosted on Hugging Face Hub: [Chaima-KHENAFIF/algerian-darija-script-detector](https://huggingface.co/Chaima-KHENAFIF/algerian-darija-script-detector)
