import argparse
import random
from pathlib import Path

import pandas as pd


def read_tsv_sentences(path: Path):
    if not path.exists():
        return []

    df = pd.read_csv(path, sep='\t', encoding='utf-8')
    if 'sentence' not in df.columns:
        raise ValueError(f"Missing 'sentence' column in {path}")

    sentences = []
    for text in df['sentence'].fillna('').astype(str).tolist():
        text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ').strip()
        if text:
            sentences.append(text)
    return sentences


def build_corpus(data_dir: Path, include_test: bool, deduplicate: bool, shuffle: bool, seed: int):
    all_sentences = []
    all_sentences.extend(read_tsv_sentences(data_dir / 'train.tsv'))
    all_sentences.extend(read_tsv_sentences(data_dir / 'dev.tsv'))
    if include_test:
        all_sentences.extend(read_tsv_sentences(data_dir / 'test.tsv'))

    if deduplicate:
        seen = set()
        unique_sentences = []
        for text in all_sentences:
            if text not in seen:
                seen.add(text)
                unique_sentences.append(text)
        all_sentences = unique_sentences

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_sentences)

    return all_sentences


def main():
    parser = argparse.ArgumentParser(description='Build a plain-text MLM corpus from train/dev/test TSV files.')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing train.tsv, dev.tsv, and optionally test.tsv')
    parser.add_argument('--output-file', type=str, required=True, help='Output .txt file, one sentence per line')
    parser.add_argument('--include-test', action='store_true', help='Include test.tsv if it exists')
    parser.add_argument('--no-deduplicate', action='store_true', help='Keep duplicated sentences')
    parser.add_argument('--no-shuffle', action='store_true', help='Keep original order')
    parser.add_argument('--seed', type=int, default=44)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sentences = build_corpus(
        data_dir=data_dir,
        include_test=args.include_test,
        deduplicate=not args.no_deduplicate,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    if not sentences:
        raise RuntimeError('No sentences were found. Please check train.tsv/dev.tsv/test.tsv.')

    with output_file.open('w', encoding='utf-8') as f:
        for text in sentences:
            f.write(text + '\n')

    print(f'[done] Wrote {len(sentences)} sentences to: {output_file}')


if __name__ == '__main__':
    main()