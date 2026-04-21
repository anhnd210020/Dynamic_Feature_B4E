import argparse
import gc
import os
import pickle as pkl
import random
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from env_config import env_config
from ETH_GBert import ETH_GBertModel
from utils import (
    CorpusDataset,
    InputExample,
    get_class_count_and_weight,
    sparse_scipy2torch,
)


# =========================
# Global setup
# =========================
random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)
torch.manual_seed(env_config.GLOBAL_SEED)

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)

DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# Argument / config helpers
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="Dataset")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--sw", type=int, default=0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="ETH_GBert")
    parser.add_argument("--validate_program", action="store_true")

    # NEW: explicit local BERT path
    parser.add_argument("--bert_dir", type=str, default="")

    return parser.parse_args()


def build_config(args):
    cfg = {
        "dataset": args.ds,
        "model_type": args.model,
        "use_stop_words": args.sw == 1,
        "resume_from_checkpoint": args.load == 1,
        "gcn_embedding_dim": args.dim,
        "learning_rate": args.lr,
        "weight_decay": args.l2,
        "total_train_epochs": 9,
        "dropout_rate": 0.2,
        "gradient_accumulation_steps": 1,
        "bert_model_scale": "bert-base-uncased",
        "do_lower_case": True,
        "warmup_proportion": 0.1,
        "output_dir": "./output/",
        "perform_metrics_str": ["weighted avg", "f1-score"],
        "do_softmax_before_mse": True,
        "loss_criterion": "cle",
    }

    if args.ds == "Dataset":
        cfg["batch_size"] = 16
        cfg["learning_rate"] = 8e-6
        cfg["weight_decay"] = 0.001
    else:
        cfg["batch_size"] = 16

    if args.validate_program:
        cfg["total_train_epochs"] = 1

    # Priority 1: explicitly passed MLM/local checkpoint
    if args.bert_dir:
        cfg["bert_model_scale"] = args.bert_dir

    # Priority 2: old offline logic
    elif env_config.TRANSFORMERS_OFFLINE == 1:
        cfg["bert_model_scale"] = os.path.join(
            env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
            f"hf-maintainers_{cfg['bert_model_scale']}",
        )

    cfg["max_seq_length"] = 200 + cfg["gcn_embedding_dim"]
    cfg["data_dir"] = f"data/preprocessed/{args.ds}"
    cfg["model_file"] = (
        f"{cfg['model_type']}{cfg['gcn_embedding_dim']}_model_{args.ds}_{cfg['loss_criterion']}"
        f"_sw{int(cfg['use_stop_words'])}.pt"
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg

def print_config(cfg, args):
    print(cfg["model_type"] + " Start at:", time.asctime())
    print(
        "\n----- Configure -----",
        f"\n  args.ds: {args.ds}",
        f"\n  stop_words: {cfg['use_stop_words']}",
        f"\n  Vocab GCN_hidden_dim: vocab_size -> 128 -> {cfg['gcn_embedding_dim']}",
        f"\n  Learning_rate0: {cfg['learning_rate']}",
        f"\n  weight_decay: {cfg['weight_decay']}",
        f"\n  Loss_criterion {cfg['loss_criterion']}",
        f"\n  softmax_before_mse: {cfg['do_softmax_before_mse']}",
        f"\n  Dropout: {cfg['dropout_rate']}",
        f"\n  gcn_act_func: Relu",
        f"\n  MAX_SEQ_LENGTH: {cfg['max_seq_length']}",
        f"\n  perform_metrics_str: {cfg['perform_metrics_str']}",
        f"\n  model_file_4save: {cfg['model_file']}",
        f"\n  validate_program: {args.validate_program}",
        f"\n  DEVICE: {DEVICE}",
        f"\n  BERT source: {cfg['bert_model_scale']}",
    )


# =========================
# Data helpers
# =========================
def load_pickle_bundle(data_dir, dataset_name):
    print("\n----- Prepare data set -----")
    print(
        f"  Load/shuffle/seperate {dataset_name} dataset, and vocabulary graph adjacent matrix"
    )

    names = [
        "labels",
        "train_y",
        "train_y_prob",
        "valid_y",
        "valid_y_prob",
        "test_y",
        "test_y_prob",
        "shuffled_clean_docs",
        "address_to_index",
    ]

    objects = []
    for name in names:
        path = f"./{data_dir}/data_{dataset_name}.{name}"
        with open(path, "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))

    (
        labels_list,
        train_y,
        train_y_prob,
        valid_y,
        valid_y_prob,
        test_y,
        test_y_prob,
        shuffled_clean_docs,
        address_to_index,
    ) = tuple(objects)

    return {
        "labels_list": labels_list,
        "train_y": train_y,
        "train_y_prob": train_y_prob,
        "valid_y": valid_y,
        "valid_y_prob": valid_y_prob,
        "test_y": test_y,
        "test_y_prob": test_y_prob,
        "shuffled_clean_docs": shuffled_clean_docs,
        "address_to_index": address_to_index,
    }


def build_examples(bundle):
    labels_list = bundle["labels_list"]
    train_y = bundle["train_y"]
    train_y_prob = bundle["train_y_prob"]
    valid_y = bundle["valid_y"]
    valid_y_prob = bundle["valid_y_prob"]
    test_y = bundle["test_y"]
    test_y_prob = bundle["test_y_prob"]
    shuffled_clean_docs = bundle["shuffled_clean_docs"]

    label2idx = labels_list[0]
    idx2label = labels_list[1]

    y = np.hstack((train_y, valid_y, test_y))
    y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

    examples = []
    for i, text in enumerate(shuffled_clean_docs):
        examples.append(InputExample(i, text.strip(), confidence=y_prob[i], label=y[i]))

    train_size = len(train_y)
    valid_size = len(valid_y)
    test_size = len(test_y)

    indices = np.arange(0, len(examples))
    train_examples = [examples[i] for i in indices[:train_size]]
    valid_examples = [
        examples[i] for i in indices[train_size: train_size + valid_size]
    ]
    test_examples = [
        examples[i]
        for i in indices[
            train_size + valid_size: train_size + valid_size + test_size
        ]
    ]

    return {
        "label2idx": label2idx,
        "idx2label": idx2label,
        "all_examples": examples,
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "test_examples": test_examples,
        "num_classes": len(label2idx),
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
    }


def load_gcn_adj_list(dataset_name):
    npz = np.load(f"data/preprocessed/{dataset_name}/norm_adj_coo.npz")
    a_hat = sp.coo_matrix(
        (npz["data"], (npz["row"], npz["col"])),
        shape=tuple(npz["shape"]),
    )
    return [sparse_scipy2torch(a_hat).to(DEVICE)]


def build_tokenizer(cfg):
    print(f"[Tokenizer] loading from: {cfg['bert_model_scale']}")
    return BertTokenizer.from_pretrained(
        cfg["bert_model_scale"],
        do_lower_case=cfg["do_lower_case"],
    )


def _extract_label_from_dataset_item(item):
    # CorpusDataset item format is expected to end with label or contain it at index 4.
    if isinstance(item, (list, tuple)):
        if len(item) >= 5:
            return item[4]
        return item[-1]
    raise ValueError("Unexpected dataset item format while building weighted sampler")


def build_dataloader(
    examples,
    tokenizer,
    address_to_index,
    max_seq_length,
    gcn_embedding_dim,
    batch_size,
    shuffle_choice,
    classes_weight=None,
    total_resample_size=-1,
):
    dataset = CorpusDataset(
        examples,
        tokenizer,
        address_to_index,
        max_seq_length,
        gcn_embedding_dim,
    )

    if shuffle_choice == 0:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=dataset.pad,
        )

    if shuffle_choice == 1:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.pad,
        )

    if shuffle_choice == 2:
        assert classes_weight is not None
        assert total_resample_size > 0
        weights = [classes_weight[_extract_label_from_dataset_item(item)] for item in dataset]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=total_resample_size,
            replacement=True,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=dataset.pad,
        )

    raise ValueError(f"Unsupported shuffle_choice: {shuffle_choice}")


# =========================
# Eval / predict helpers
# =========================
def forward_model(model, gcn_adj_list, batch):
    input_ids, input_mask, segment_ids, _, label_ids, gcn_swop_eye = batch
    logits = model(
        gcn_adj_list,
        gcn_swop_eye,
        input_ids,
        segment_ids,
        input_mask,
    )
    return logits, label_ids


def predict(model, examples, tokenizer, cfg, address_to_index, gcn_adj_list):
    dataloader = build_dataloader(
        examples=examples,
        tokenizer=tokenizer,
        address_to_index=address_to_index,
        max_seq_length=cfg["max_seq_length"],
        gcn_embedding_dim=cfg["gcn_embedding_dim"],
        batch_size=cfg["batch_size"],
        shuffle_choice=0,
    )

    predict_out = []
    confidence_out = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            logits, _ = forward_model(model, gcn_adj_list, batch)

            if cfg["loss_criterion"] == "mse" and cfg["do_softmax_before_mse"]:
                logits = F.softmax(logits, dim=-1)

            predict_out.extend(logits.max(1)[1].tolist())
            confidence_out.extend(logits.max(1)[0].tolist())

    return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(-1)


def evaluate(
    model,
    gcn_adj_list,
    predict_dataloader,
    epoch_th,
    dataset_name,
    cfg,
    num_classes,
    loss_weight,
):
    model.eval()
    predict_out = []
    all_label_ids = []
    ev_loss = 0.0
    total = 0
    correct = 0
    start = time.time()

    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch

            logits = model(
                gcn_adj_list,
                gcn_swop_eye,
                input_ids,
                segment_ids,
                input_mask,
            )

            if cfg["loss_criterion"] == "mse":
                if cfg["do_softmax_before_mse"]:
                    logits = F.softmax(logits, dim=-1)
                loss = F.mse_loss(logits, y_prob)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, num_classes),
                    label_ids,
                    weight=loss_weight,
                )

            ev_loss += loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            correct += predicted.eq(label_ids).sum().item()
            total += len(label_ids)

    y_true = np.array(all_label_ids).reshape(-1)
    y_pred = np.array(predict_out).reshape(-1)
    f1_metrics = f1_score(y_true, y_pred, average="weighted")

    print("Report:\n" + classification_report(y_true, y_pred, digits=4))

    ev_acc = correct / total
    end = time.time()
    print(
        "Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation"
        % (
            epoch_th,
            " ".join(cfg["perform_metrics_str"]),
            100 * f1_metrics,
            100.0 * ev_acc,
            dataset_name,
            (end - start) / 60.0,
        )
    )
    print("--------------------------------------------------------------")
    return ev_loss, ev_acc, f1_metrics


# =========================
# Model helpers
# =========================
def load_or_init_model(cfg, label2idx, gcn_vocab_size, gcn_adj_list):
    checkpoint_path = os.path.join(cfg["output_dir"], cfg["model_file"])
    print(f"[Model] loading BERT from: {cfg['bert_model_scale']}")

    if cfg["resume_from_checkpoint"] and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "step" in checkpoint:
            prev_save_step = checkpoint["step"]
            start_epoch = checkpoint["epoch"]
        else:
            prev_save_step = -1
            start_epoch = checkpoint["epoch"] + 1

        valid_acc_prev = checkpoint["valid_acc"]
        perform_metrics_prev = checkpoint["perform_metrics"]

        model = ETH_GBertModel.from_pretrained(
            cfg["bert_model_scale"],
            state_dict=checkpoint["model_state"],
            gcn_adj_dim=gcn_vocab_size,
            gcn_adj_num=len(gcn_adj_list),
            gcn_embedding_dim=cfg["gcn_embedding_dim"],
            num_labels=len(label2idx),
        )

        pretrained_dict = checkpoint["model_state"]
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {
            k: v for k, v in pretrained_dict.items() if k in net_state_dict
        }
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)

        print(
            f"Loaded the pretrain model: {cfg['model_file']}",
            f", epoch: {checkpoint['epoch']}",
            f"step: {prev_save_step}",
            f"valid acc: {checkpoint['valid_acc']}",
            f"{' '.join(cfg['perform_metrics_str'])}_valid: {checkpoint['perform_metrics']}",
        )
    else:
        model = ETH_GBertModel.from_pretrained(
            cfg["bert_model_scale"],
            gcn_adj_dim=gcn_vocab_size,
            gcn_adj_num=len(gcn_adj_list),
            gcn_embedding_dim=cfg["gcn_embedding_dim"],
            num_labels=len(label2idx),
        )
        start_epoch = 0
        prev_save_step = -1
        valid_acc_prev = 0
        perform_metrics_prev = 0

    model.to(DEVICE)
    print(f"[Model] first parameter device: {next(model.parameters()).device}")
    return model, start_epoch, prev_save_step, valid_acc_prev, perform_metrics_prev


# =========================
# Training
# =========================
def train():
    args = parse_args()
    cfg = build_config(args)
    print_config(cfg, args)

    bundle = load_pickle_bundle(cfg["data_dir"], cfg["dataset"])
    example_info = build_examples(bundle)
    label2idx = example_info["label2idx"]
    train_examples = example_info["train_examples"]
    valid_examples = example_info["valid_examples"]
    test_examples = example_info["test_examples"]
    num_classes = example_info["num_classes"]
    address_to_index = bundle["address_to_index"]
    gcn_vocab_size = len(address_to_index)

    if args.validate_program:
        train_examples = [train_examples[0]]
        valid_examples = [valid_examples[0]]
        test_examples = [test_examples[0]]

    gcn_adj_list = load_gcn_adj_list(cfg["dataset"])
    tokenizer = build_tokenizer(cfg)

    gc.collect()

    train_classes_num, train_classes_weight = get_class_count_and_weight(
        bundle["train_y"],
        len(label2idx),
    )
    loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(DEVICE)

    train_dataloader = build_dataloader(
        examples=train_examples,
        tokenizer=tokenizer,
        address_to_index=address_to_index,
        max_seq_length=cfg["max_seq_length"],
        gcn_embedding_dim=cfg["gcn_embedding_dim"],
        batch_size=cfg["batch_size"],
        shuffle_choice=0,
    )
    valid_dataloader = build_dataloader(
        examples=valid_examples,
        tokenizer=tokenizer,
        address_to_index=address_to_index,
        max_seq_length=cfg["max_seq_length"],
        gcn_embedding_dim=cfg["gcn_embedding_dim"],
        batch_size=cfg["batch_size"],
        shuffle_choice=0,
    )
    test_dataloader = build_dataloader(
        examples=test_examples,
        tokenizer=tokenizer,
        address_to_index=address_to_index,
        max_seq_length=cfg["max_seq_length"],
        gcn_embedding_dim=cfg["gcn_embedding_dim"],
        batch_size=cfg["batch_size"],
        shuffle_choice=0,
    )

    total_train_steps = int(
        len(train_dataloader)
        / cfg["gradient_accumulation_steps"]
        * cfg["total_train_epochs"]
    )

    print("  Train_classes count:", train_classes_num)
    print(
        f"  Num examples for train = {len(train_examples)}",
        f", after weight sample: {len(train_dataloader) * cfg['batch_size']}",
    )
    print("  Num examples for validate = %d" % len(valid_examples))
    print("  Batch size = %d" % cfg["batch_size"])
    print("  Num steps = %d" % total_train_steps)
    print("\n----- Running training -----")

    model, start_epoch, prev_save_step, valid_acc_prev, perform_metrics_prev = load_or_init_model(
        cfg=cfg,
        label2idx=label2idx,
        gcn_vocab_size=gcn_vocab_size,
        gcn_adj_list=gcn_adj_list,
    )

    optimizer = BertAdam(
        model.parameters(),
        lr=cfg["learning_rate"],
        warmup=cfg["warmup_proportion"],
        t_total=total_train_steps,
        weight_decay=cfg["weight_decay"],
    )

    train_start = time.time()
    global_step_th = int(
        len(train_examples)
        / cfg["batch_size"]
        / cfg["gradient_accumulation_steps"]
        * start_epoch
    )

    all_loss_list = {"train": [], "valid": [], "test": []}
    all_f1_list = {"train": [], "valid": [], "test": []}

    test_f1_best = 0.0
    test_f1_best_epoch = -1
    test_f1_when_valid_best = 0.0
    valid_f1_best_epoch = -1

    for epoch in range(start_epoch, cfg["total_train_epochs"]):
        tr_loss = 0.0
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            if prev_save_step > -1:
                if step <= prev_save_step:
                    continue
                prev_save_step = -1

            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch

            logits = model(
                gcn_adj_list,
                gcn_swop_eye,
                input_ids,
                segment_ids,
                input_mask,
            )

            if cfg["loss_criterion"] == "mse":
                if cfg["do_softmax_before_mse"]:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, num_classes),
                    label_ids,
                    weight=loss_weight,
                )

            if cfg["gradient_accumulation_steps"] > 1:
                loss = loss / cfg["gradient_accumulation_steps"]

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

            if step % 40 == 0:
                print(
                    "Epoch:{}-{}/{}, Train {} Loss: {}, Cumulated time: {}m ".format(
                        epoch,
                        step,
                        len(train_dataloader),
                        cfg["loss_criterion"],
                        loss.item(),
                        (time.time() - train_start) / 60.0,
                    )
                )

        print("--------------------------------------------------------------")
        valid_loss, valid_acc, perform_metrics = evaluate(
            model=model,
            gcn_adj_list=gcn_adj_list,
            predict_dataloader=valid_dataloader,
            epoch_th=epoch,
            dataset_name="Valid_set",
            cfg=cfg,
            num_classes=num_classes,
            loss_weight=loss_weight,
        )
        test_loss, _, test_f1 = evaluate(
            model=model,
            gcn_adj_list=gcn_adj_list,
            predict_dataloader=test_dataloader,
            epoch_th=epoch,
            dataset_name="Test_set",
            cfg=cfg,
            num_classes=num_classes,
            loss_weight=loss_weight,
        )

        if test_f1 > test_f1_best:
            test_f1_best = test_f1
            test_f1_best_epoch = epoch

        all_loss_list["train"].append(tr_loss)
        all_loss_list["valid"].append(valid_loss)
        all_loss_list["test"].append(test_loss)
        all_f1_list["valid"].append(perform_metrics)
        all_f1_list["test"].append(test_f1)

        print(
            "Epoch:{} completed, Total Train Loss:{}, Valid Loss:{}, Spend {}m ".format(
                epoch,
                tr_loss,
                valid_loss,
                (time.time() - train_start) / 60.0,
            )
        )

        if perform_metrics > perform_metrics_prev:
            to_save = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "valid_acc": valid_acc,
                "lower_case": cfg["do_lower_case"],
                "perform_metrics": perform_metrics,
            }
            torch.save(to_save, os.path.join(cfg["output_dir"], cfg["model_file"]))
            perform_metrics_prev = perform_metrics
            valid_acc_prev = valid_acc
            test_f1_when_valid_best = test_f1
            valid_f1_best_epoch = epoch

    print("\n**Optimization Finished!,Total spend:", (time.time() - train_start) / 60.0)
    print("**Valid weighted F1: %.3f at %d epoch." % (100 * perform_metrics_prev, valid_f1_best_epoch))
    print("**Test weighted F1 when valid best: %.3f" % (100 * test_f1_when_valid_best))
    print("**Test weighted F1 (absolute best): %.3f at %d epoch." % (100 * test_f1_best, test_f1_best_epoch))


if __name__ == "__main__":
    train()