"""
Data loading and cleaning for the 3-class Darija detector
(darija_latin / darija_arabic / not_darija).
"""

import pandas as pd

LABEL_MAPPING = {
    "darija_latin": 0,
    "darija_arabic": 1,
    "not_darija": 2,
}


def load_and_clean(raw_csv_path: str, save_path: str = "cleaned_darija_dataset.csv") -> pd.DataFrame:
    """
    Load the raw dataset, map labels, drop duplicates/NaNs, and save the cleaned version.

    Expects a CSV with 'text' and 'lang' columns, where 'lang' is one of
    'darija_latin', 'darija_arabic', or 'not_darija'.
    """
    df = pd.read_csv(raw_csv_path)
    df = df[["text", "lang"]]

    df["label"] = df["lang"].map(LABEL_MAPPING)

    if df["label"].isnull().any():
        print("Warning: some rows have an unexpected 'lang' value!")
        print(df[df["label"].isnull()]["lang"].value_counts())

    df.dropna(subset=["text", "lang"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(df["label"].value_counts())
    print(df.head())

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128, test_size: float = 0.2):
    """
    Convert a cleaned DataFrame into a tokenized HuggingFace train/test DatasetDict.
    """
    from datasets import Dataset

    dataset = Dataset.from_pandas(df[["text", "label"]])

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print(tokenized_dataset)
    return tokenized_dataset


if __name__ == "__main__":
    import sys
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "darija_dataset.csv"
    load_and_clean(raw_path)
