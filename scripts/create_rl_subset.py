"""Create a leakage-controlled subset for a small RL experiment."""

from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "medix-rl-data"
    / "data"
    / "train-00005-of-00046.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "medix-rl-data-small"
TRAIN_SIZE = 64
VAL_SIZE = 16


def is_eligible(row):
    source = str(row.get("source", "")).lower()
    return (
        "vqa-rad" not in source
        and bool(row.get("image"))
        and bool(row.get("problem"))
        and bool(row.get("solution"))
    )


def main():
    if not SOURCE_FILE.exists():
        raise SystemExit(f"Source parquet not found: {SOURCE_FILE}")

    dataset = load_dataset("parquet", data_files=str(SOURCE_FILE), split="train")
    dataset = dataset.filter(is_eligible, num_proc=1).shuffle(seed=42)

    required = TRAIN_SIZE + VAL_SIZE
    if len(dataset) < required:
        raise ValueError(f"Only {len(dataset)} eligible samples found; {required} required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.select(range(TRAIN_SIZE)).to_parquet(OUTPUT_DIR / "train.parquet")
    dataset.select(range(TRAIN_SIZE, required)).to_parquet(OUTPUT_DIR / "test.parquet")
    print(f"Created {TRAIN_SIZE} train and {VAL_SIZE} validation samples in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
