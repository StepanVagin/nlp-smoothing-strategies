"""Download WikiText-2 via the datasets library, tokenize, and write train/valid/test splits."""

from pathlib import Path

from datasets import load_dataset


def tokenize_and_write(split_data: list[str], output_path: Path) -> int:
    """Lowercase, whitespace-tokenize, insert <eos> at sentence boundaries, write one token per line.

    Returns the total token count written.
    """
    token_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for line in split_data:
            line = line.strip()
            if not line:
                f.write("<eos>\n")
                token_count += 1
                continue
            tokens = line.lower().split()
            for tok in tokens:
                f.write(tok + "\n")
                token_count += 1
    return token_count


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Downloading WikiText-2 dataset...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    splits = {"train": "train", "valid": "validation", "test": "test"}

    for name, ds_split in splits.items():
        output_path = data_dir / f"{name}.txt"
        texts = ds[ds_split]["text"]
        count = tokenize_and_write(texts, output_path)
        print(f"{name}: {count:,} tokens -> {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
