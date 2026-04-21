import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class TextSample:
    text: str


class MLMTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.examples = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, idx):
        item = {k: self.examples[k][idx] for k in self.examples.keys()}
        return item


def read_lines(path: Path) -> List[str]:
    lines = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def split_train_valid(lines: List[str], valid_ratio: float, seed: int):
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError('--valid-ratio must be between 0 and 1')
    if len(lines) < 2:
        raise ValueError('Need at least 2 lines to create a validation split')

    rng = random.Random(seed)
    shuffled = list(lines)
    rng.shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * valid_ratio))
    valid_lines = shuffled[:valid_size]
    train_lines = shuffled[valid_size:]
    if not train_lines:
        raise ValueError('Validation split is too large; training set became empty')
    return train_lines, valid_lines


def write_backward_compatible_files(output_dir: Path):
    config_json = output_dir / 'config.json'
    legacy_config = output_dir / 'bert_config.json'
    if config_json.exists() and not legacy_config.exists():
        shutil.copyfile(config_json, legacy_config)

    safetensors_file = output_dir / 'model.safetensors'
    bin_file = output_dir / 'pytorch_model.bin'
    if safetensors_file.exists() and not bin_file.exists():
        raise RuntimeError(
            'Expected pytorch_model.bin for pytorch_pretrained_bert compatibility. '
            'Please re-run with save_safetensors=False.'
        )


def main():
    parser = argparse.ArgumentParser(description='Continue pretraining bert-base-uncased with MLM on transaction text.')
    parser.add_argument('--train-file', type=str, required=True, help='Plain-text corpus: one sentence per line')
    parser.add_argument('--validation-file', type=str, default=None, help='Optional validation corpus; if omitted, split from train-file')
    parser.add_argument('--output-dir', type=str, required=True, help='Folder to save the adapted BERT checkpoint')
    parser.add_argument('--model-name-or-path', type=str, default='bert-base-uncased')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--valid-ratio', type=float, default=0.05)
    parser.add_argument('--mlm-probability', type=float, default=0.15)
    parser.add_argument('--per-device-train-batch-size', type=int, default=16)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-train-epochs', type=float, default=3.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--logging-steps', type=int, default=50)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--save-total-limit', type=int, default=2)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_lines = read_lines(Path(args.train_file))
    if not train_lines:
        raise RuntimeError('Training corpus is empty')

    if args.validation_file:
        valid_lines = read_lines(Path(args.validation_file))
        if not valid_lines:
            raise RuntimeError('Validation corpus is empty')
    else:
        train_lines, valid_lines = split_train_valid(train_lines, args.valid_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    train_dataset = MLMTextDataset(train_lines, tokenizer, args.max_length)
    eval_dataset = MLMTextDataset(valid_lines, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy='steps',
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=args.fp16,
        report_to='none',
        save_safetensors=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_backward_compatible_files(output_dir)

    metrics = train_result.metrics
    metrics['num_train_examples'] = len(train_lines)
    metrics['num_eval_examples'] = len(valid_lines)
    with (output_dir / 'mlm_train_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('[done] Saved domain-adapted BERT to:', output_dir)
    print('[compat] pytorch_model.bin, vocab.txt, and bert_config.json are ready for pytorch_pretrained_bert.')
    print('\nUse these environment variables before running the original trainModel.py:')
    print(f'  export TRANSFORMERS_OFFLINE=1')
    print(f'  export HUGGING_LOCAL_MODEL_FILES_PATH={output_dir.parent}')
    print('And make sure the model directory name is exactly: hf-maintainers_bert-base-uncased')


if __name__ == '__main__':
    main()