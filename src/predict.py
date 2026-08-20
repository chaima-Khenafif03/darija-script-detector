"""
Load the fine-tuned 3-class Darija detector and classify text.

Usage:
    python predict.py "kifach rak"
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_PATH = "Chaima-KHENAFIF/algerian-darija-script-detector"

LABEL_NAMES = {
    "LABEL_0": "Darija (Latin script)",
    "LABEL_1": "Darija (Arabic script)",
    "LABEL_2": "Not Darija",
}

_classifier = None  # lazy-loaded so importing this file doesn't load the model


def _get_classifier():
    global _classifier
    if _classifier is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return _classifier


def predict(text: str) -> str:
    """Return the predicted class label for the given text."""
    classifier = _get_classifier()
    result = classifier(text)[0]
    return LABEL_NAMES.get(result["label"], result["label"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = sys.argv[1]
        print(f"Text: {text}\nPrediction: {predict(text)}")
    else:
        # quick sanity check with the notebook's original examples
        examples = [
            "كي راك خويا؟",
            "How are you doing today?",
            "أنا جيت البارح مع الليل",
            "Je suis très fatigué aujourd'hui.",
            "machi kamel mli7a.",
        ]
        for text in examples:
            print(f"Text: {text}\nPrediction: {predict(text)}\n")
