"""Create a tiny local split for an end-to-end MediX-R1 smoke test."""
from pathlib import Path

from datasets import load_dataset, load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "medix-rl-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "medix-rl-data-smoke"


def load_source():
    if not SOURCE_DIR.exists():
        raise SystemExit(
            f"Training data not found at {SOURCE_DIR}. Download MBZUAI/medix-rl-data first."
        )

    parquet_files = sorted(SOURCE_DIR.rglob("*.parquet"))
    if parquet_files:
        train_files = [str(path) for path in parquet_files if "train" in path.name]
        test_files = [
            str(path)
            for path in parquet_files
            if "test" in path.name or "validation" in path.name
        ]
        if not train_files:
            train_files = [str(path) for path in parquet_files]
        if not test_files:
            test_files = train_files
        return (
            load_dataset("parquet", data_files=train_files, split="train"),
            load_dataset("parquet", data_files=test_files, split="train"),
        )

    dataset = load_from_disk(str(SOURCE_DIR))
    if hasattr(dataset, "keys"):
        train = dataset["train"]
        test = dataset.get("test") or dataset.get("validation") or train
        return train, test
    return dataset, dataset


def main():
    train, test = load_source()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.select(range(min(4, len(train)))).to_parquet(OUTPUT_DIR / "train.parquet")
    test.select(range(min(2, len(test)))).to_parquet(OUTPUT_DIR / "test.parquet")
    print(f"Created smoke data in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
