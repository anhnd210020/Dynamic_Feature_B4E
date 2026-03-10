# trainModel.py
import argparse
import gc
import os
import pickle as pkl
import random
import time
import warnings
import re
from collections import Counter
import inspect as pyinspect

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

# huggingface transformers 0.6.2 (pytorch_pretrained_bert)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from env_config import env_config
from ETH_GBert import ETH_GBertModel
from utils import InputExample, CorpusDataset, sparse_scipy2torch, get_class_count_and_weight

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Reproducibility / device
# -------------------------
random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)
torch.manual_seed(env_config.GLOBAL_SEED)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)
device = torch.device("cuda:0" if cuda_yes else "cpu")


# -------------------------
# Configuration
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="Dataset")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--sw", type=int, default=0)
parser.add_argument("--dim", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--l2", type=float, default=0.01)
parser.add_argument("--model", type=str, default="ETH_GBert")
parser.add_argument("--validate_program", action="store_true")
args = parser.parse_args()

print("A3_ENABLE=", env_config.A3_ENABLE, "AUX_MLM=", env_config.AUX_MLM,
      "TOPK=", env_config.A3_TOPK_ADDR, "VAL_BINS=", env_config.A3_VAL_BINS)

cfg_model_type = args.model
cfg_stop_words = (args.sw == 1)  # kept for filename compatibility
will_train_mode_from_checkpoint = (args.load == 1)

gcn_embedding_dim = args.dim
learning_rate0 = args.lr
l2_decay = args.l2

total_train_epochs = 9
dropout_rate = 0.2

if args.ds == "Dataset":
    batch_size = 16
    learning_rate0 = 8e-6
    l2_decay = 0.001

MAX_SEQ_LENGTH = 200 + gcn_embedding_dim
gradient_accumulation_steps = 1

bert_model_scale = env_config.BERT_NAME or "bert-base-uncased"
if env_config.TRANSFORMERS_OFFLINE == 1:
    bert_model_scale = os.path.join(
        env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
        f"hf-maintainers_{bert_model_scale}",
    )

do_lower_case = True
warmup_proportion = 0.1

data_dir = f"data/preprocessed/{args.ds}"
output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)

perform_metrics_str = ["weighted avg", "f1-score"]
do_softmax_before_mse = True
cfg_loss_criterion = "cle"

# A2 (aux MLM)
AUX_MLM = env_config.AUX_MLM
MLM_PROB = env_config.MLM_PROB
MLM_LAMBDA = env_config.MLM_LAMBDA

# A3 (multi-modal encoding)
A3_ENABLE = env_config.A3_ENABLE
A3_TOPK_ADDR = env_config.A3_TOPK_ADDR
A3_VAL_BINS = env_config.A3_VAL_BINS
A3_DT_BINS = env_config.A3_DT_BINS
A3_UNUSED_BUDGET = env_config.A3_UNUSED_BUDGET

model_file_4save = (
    f"{cfg_model_type}{gcn_embedding_dim}_model_{args.ds}_{cfg_loss_criterion}"
    f"_sw{int(cfg_stop_words)}"
    f"_A3{int(A3_ENABLE)}_A2{int(AUX_MLM)}.pt"
)

if args.validate_program:
    total_train_epochs = 1

print(f"{cfg_model_type} Start at: {time.asctime()}")
print(
    "\n----- Configure -----"
    f"\n  args.ds: {args.ds}"
    f"\n  stop_words: {cfg_stop_words}"
    f"\n  Vocab GCN_hidden_dim: vocab_size -> 128 -> {gcn_embedding_dim}"
    f"\n  Learning_rate0: {learning_rate0}"
    f"\n  weight_decay: {l2_decay}"
    f"\n  Loss_criterion: {cfg_loss_criterion}"
    f"\n  Dropout: {dropout_rate}"
    f"\n  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}"
    f"\n  model_file_4save: {model_file_4save}"
    f"\n  validate_program: {args.validate_program}"
    f"\n  A3_ENABLE: {A3_ENABLE} (TOPK_ADDR={A3_TOPK_ADDR}, VAL_BINS={A3_VAL_BINS}, DT_BINS={A3_DT_BINS})"
    f"\n  AUX_MLM: {AUX_MLM} (MLM_PROB={MLM_PROB}, MLM_LAMBDA={MLM_LAMBDA})"
)


# -------------------------
# Prepare data set
# -------------------------
print("\n----- Prepare data set -----")
print(f"  Load/shuffle/seperate {args.ds} dataset, and vocabulary graph adjacent matrix")

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
    datafile = f"./{data_dir}/data_{args.ds}.{name}"
    with open(datafile, "rb") as f:
        objects.append(pkl.load(f, encoding="latin1"))

(
    lables_list,
    train_y,
    train_y_prob,
    valid_y,
    valid_y_prob,
    test_y,
    test_y_prob,
    shuffled_clean_docs,
    address_to_index,
) = tuple(objects)

label2idx, idx2label = lables_list[0], lables_list[1]

y = np.hstack((train_y, valid_y, test_y))
y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

examples = []
for i, ts in enumerate(shuffled_clean_docs):
    examples.append(InputExample(i, ts.strip(), confidence=y_prob[i], label=y[i]))

num_classes = len(label2idx)
gcn_vocab_size = len(address_to_index)

train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)

indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [examples[i] for i in indexs[train_size: train_size + valid_size]]
test_examples = [examples[i] for i in indexs[train_size + valid_size: train_size + valid_size + test_size]]

# ds size=1 for validating the program
if args.validate_program:
    train_examples = [train_examples[0]]
    valid_examples = [valid_examples[0]]
    test_examples = [test_examples[0]]

# === Load normalized adjacency (COO) and build gcn_adj_list ===
npz = np.load(f"data/preprocessed/{args.ds}/norm_adj_coo.npz") if os.path.exists(f"data/preprocessed/{args.ds}/norm_adj_coo.npz") \
    else np.load("data/preprocessed/Dataset/norm_adj_coo.npz")
A_hat = sp.coo_matrix(
    (npz["data"], (npz["row"], npz["col"])),
    shape=tuple(npz["shape"]),
)
gcn_adj_list = [sparse_scipy2torch(A_hat).to(device)]

gc.collect()

train_classes_num, train_classes_weight = get_class_count_and_weight(train_y, len(label2idx))
loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(device)

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

# -------------------------
# A3: build token config + top-K addr mapping (optional)
# -------------------------
addr_pat = re.compile(r"0x[a-fA-F0-9]{40}")

TOK = None
addr_to_tok = None

if A3_ENABLE:
    # allocate [unused] tokens budget
    unused = [f"[unused{i}]" for i in range(min(100, A3_UNUSED_BUDGET))]
    ptr = 0

    def take(n):
        nonlocal_ptr = None  # just to keep linter quiet
        nonlocal_ptr = 0
        return []

    # python doesn't allow nonlocal in this scope in older versions; do it manually:
    def _take(n):
        nonlocal_ptr = None
        return n

    def take_n(n):
        nonlocal_ptr = None
        return n

    # use a simple pointer dict to avoid nonlocal
    _p = {"ptr": 0}

    def take(n):
        out = unused[_p["ptr"]: _p["ptr"] + n]
        _p["ptr"] += n
        return out

    TOK = {}

    # OOV addr
    TOK["ADDR_OOV"] = take(1)[0]
    # tag / inout
    TOK["TAG0"], TOK["TAG1"] = take(2)
    TOK["IN0"], TOK["IN1"] = take(2)
    # bins
    TOK["VAL_BINS"] = take(A3_VAL_BINS)
    TOK["DT2_BINS"] = take(A3_DT_BINS)
    TOK["DT3_BINS"] = take(A3_DT_BINS)
    TOK["DT4_BINS"] = take(A3_DT_BINS)
    TOK["DT5_BINS"] = take(A3_DT_BINS)
    # topK addr tokens
    TOK["ADDR_TOP"] = take(A3_TOPK_ADDR)

    # sanity: ensure tokens exist (should, for [unusedX])
    for k, v in TOK.items():
        toks = v if isinstance(v, list) else [v]
        ids = tokenizer.convert_tokens_to_ids(toks)
        if any(tid == tokenizer.vocab.get("[UNK]") for tid in ids):
            raise ValueError(f"A3 token(s) for {k} mapped to [UNK]. Check unused budget / tokenizer.")

    # build top addresses from TRAIN only
    cnt = Counter()
    for ex in train_examples:
        cnt.update(addr_pat.findall(ex.text_a))
    top_addrs = [a for a, _ in cnt.most_common(A3_TOPK_ADDR)]
    addr_to_tok = {a: TOK["ADDR_TOP"][i] for i, a in enumerate(top_addrs)}

    print(f"[A3] built top address map: {len(addr_to_tok)} addresses, unused_ptr={_p['ptr']}/{len(unused)}")


# -------------------------
# DataLoader
# -------------------------
def build_dataset(examples_):
    """
    Create CorpusDataset, passing A3 params only if CorpusDataset supports them.
    This makes trainModel.py compatible with both old/new utils.py.
    """
    init_sig = pyinspect.signature(CorpusDataset.__init__)
    kwargs = {}
    if "tok_cfg" in init_sig.parameters and A3_ENABLE:
        kwargs["tok_cfg"] = TOK
    if "addr_to_tok" in init_sig.parameters and A3_ENABLE:
        kwargs["addr_to_tok"] = addr_to_tok

    ds = CorpusDataset(
        examples_,
        tokenizer,
        address_to_index,
        MAX_SEQ_LENGTH,
        gcn_embedding_dim,
        **kwargs,
    )
    return ds


def get_pytorch_dataloader(examples_, batch_size_, shuffle_choice, classes_weight=None, total_resample_size=-1):
    ds = build_dataset(examples_)

    if shuffle_choice == 0:
        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size_,
            shuffle=False,
            num_workers=4,
            collate_fn=ds.pad,
        )

    if shuffle_choice == 1:
        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size_,
            shuffle=True,
            num_workers=4,
            collate_fn=ds.pad,
        )

    if shuffle_choice == 2:
        # FIX: ds item has 6 fields; label is item[4]
        assert classes_weight is not None
        assert total_resample_size > 0
        weights = [classes_weight[int(item[4])] for item in ds]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=total_resample_size, replacement=True)
        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size_,
            sampler=sampler,
            num_workers=4,
            collate_fn=ds.pad,
        )

    raise ValueError(f"Unknown shuffle_choice={shuffle_choice}")


batch_size = 16
train_dataloader = get_pytorch_dataloader(train_examples, batch_size, shuffle_choice=0)
valid_dataloader = get_pytorch_dataloader(valid_examples, batch_size, shuffle_choice=0)
test_dataloader = get_pytorch_dataloader(test_examples, batch_size, shuffle_choice=0)

total_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * total_train_epochs)

print("  Train_classes count:", train_classes_num)
print(f"  Num examples for train = {len(train_examples)}, batches={len(train_dataloader)}")
print(f"  Num examples for validate = {len(valid_examples)}, batches={len(valid_dataloader)}")
print(f"  Batch size = {batch_size}")
print(f"  Num steps = {total_train_steps}")


# -------------------------
# Helpers: batch unpack (6-tuple only)
# -------------------------
def unpack_batch(batch):
    batch = tuple(t.to(device) for t in batch)
    if len(batch) != 6:
        raise ValueError(f"Expected batch size 6, got {len(batch)}")
    input_ids, input_mask, segment_ids, y_prob_or_dummy, label_ids, gcn_swop_eye = batch
    return input_ids, input_mask, segment_ids, y_prob_or_dummy, label_ids, gcn_swop_eye


# -------------------------
# A2: MLM masking
# -------------------------
def mask_inputs_for_mlm(input_ids, attention_mask, tokenizer_, mlm_prob=0.15):
    """
    Returns masked_input_ids, mlm_labels
      - masked_input_ids: input_ids with MLM corruption
      - mlm_labels: -100 for non-masked, original token id for masked positions
    """
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, fill_value=-100)

    eligible = attention_mask.bool()

    cls_id = tokenizer_.vocab.get("[CLS]")
    sep_id = tokenizer_.vocab.get("[SEP]")
    pad_id = tokenizer_.vocab.get("[PAD]")
    mask_id = tokenizer_.vocab.get("[MASK]")

    if cls_id is not None:
        eligible &= (input_ids != cls_id)
    if sep_id is not None:
        eligible &= (input_ids != sep_id)
    if pad_id is not None:
        eligible &= (input_ids != pad_id)

    prob = torch.rand_like(input_ids.float())
    mask_pos = eligible & (prob < mlm_prob)

    labels[mask_pos] = input_ids[mask_pos]

    rand = torch.rand_like(input_ids.float())
    random_tokens = torch.randint(low=0, high=len(tokenizer_.vocab), size=input_ids.shape, device=input_ids.device)

    # 80% -> [MASK]
    masked[mask_pos & (rand < 0.8)] = mask_id
    # 10% -> random
    masked[mask_pos & (rand >= 0.8) & (rand < 0.9)] = random_tokens[mask_pos & (rand >= 0.8) & (rand < 0.9)]
    # 10% -> keep

    return masked, labels


# -------------------------
# Eval / Predict
# -------------------------
@torch.no_grad()
def predict(model, examples_, batch_size_):
    dataloader = get_pytorch_dataloader(examples_, batch_size_, shuffle_choice=0)
    predict_out, confidence_out = [], []

    model.eval()
    for batch in dataloader:
        input_ids, input_mask, segment_ids, _, label_ids, gcn_swop_eye = unpack_batch(batch)

        score_out = model(
            gcn_adj_list,
            gcn_swop_eye,
            input_ids,
            segment_ids,
            input_mask,
        )

        if cfg_loss_criterion == "mse" and do_softmax_before_mse:
            score_out = F.softmax(score_out, dim=-1)

        predict_out.extend(score_out.max(1)[1].tolist())
        confidence_out.extend(score_out.max(1)[0].tolist())

    return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(-1)


@torch.no_grad()
def evaluate(model, predict_dataloader, epoch_th, dataset_name):
    model.eval()
    predict_out, all_label_ids = [], []
    ev_loss, total, correct = 0.0, 0, 0
    start = time.time()

    for batch in predict_dataloader:
        input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = unpack_batch(batch)

        logits = model(
            gcn_adj_list,
            gcn_swop_eye,
            input_ids,
            segment_ids,
            input_mask,
        )

        if cfg_loss_criterion == "mse":
            if do_softmax_before_mse:
                logits = F.softmax(logits, -1)
            loss = F.mse_loss(logits, y_prob)
        else:
            # keep consistent with train weights for comparability
            loss = F.cross_entropy(logits.view(-1, num_classes), label_ids, weight=loss_weight)

        ev_loss += loss.item()

        _, predicted = torch.max(logits, -1)
        predict_out.extend(predicted.tolist())
        all_label_ids.extend(label_ids.tolist())

        total += len(label_ids)
        correct += predicted.eq(label_ids).sum().item()

    f1_metrics = f1_score(
        np.array(all_label_ids).reshape(-1),
        np.array(predict_out).reshape(-1),
        average="weighted",
    )
    print(
        "Report:\n"
        + classification_report(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            digits=4,
        )
    )

    ev_acc = correct / total
    end = time.time()
    print(
        "Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation"
        % (
            epoch_th,
            " ".join(perform_metrics_str),
            100 * f1_metrics,
            100.0 * ev_acc,
            dataset_name,
            (end - start) / 60.0,
        )
    )
    print("--------------------------------------------------------------")
    return ev_loss, ev_acc, f1_metrics


# -------------------------
# Train
# -------------------------
print("\n----- Running training -----")

ckpt_path = os.path.join(output_dir, model_file_4save)
if will_train_mode_from_checkpoint and os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    start_epoch = checkpoint.get("epoch", 0) + (0 if "step" in checkpoint else 1)
    prev_save_step = checkpoint.get("step", -1)

    valid_acc_prev = checkpoint.get("valid_acc", 0)
    perform_metrics_prev = checkpoint.get("perform_metrics", 0)

    model = ETH_GBertModel.from_pretrained(
        bert_model_scale,
        state_dict=checkpoint["model_state"],
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )

    # safe load intersecting keys
    pretrained_dict = checkpoint["model_state"]
    net_state_dict = model.state_dict()
    net_state_dict.update({k: v for k, v in pretrained_dict.items() if k in net_state_dict})
    model.load_state_dict(net_state_dict)

    print(
        f"Loaded checkpoint: {model_file_4save}",
        f"epoch: {checkpoint.get('epoch')}, step: {prev_save_step}, valid acc: {valid_acc_prev}, valid F1: {perform_metrics_prev}",
    )
else:
    start_epoch = 0
    prev_save_step = -1
    valid_acc_prev = 0
    perform_metrics_prev = 0

    model = ETH_GBertModel.from_pretrained(
        bert_model_scale,
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=gcn_embedding_dim,
        num_labels=len(label2idx),
    )

model.to(device)

optimizer = BertAdam(
    model.parameters(),
    lr=learning_rate0,
    warmup=warmup_proportion,
    t_total=total_train_steps,
    weight_decay=l2_decay,
)

train_start = time.time()

all_loss_list = {"train": [], "valid": [], "test": []}
all_f1_list = {"valid": [], "test": []}
test_f1_best = 0.0
test_f1_best_epoch = 0
test_f1_when_valid_best = 0.0
valid_f1_best_epoch = 0

for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0.0
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        if prev_save_step > -1 and step <= prev_save_step:
            continue
        if prev_save_step > -1:
            prev_save_step = -1

        input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = unpack_batch(batch)

        # --- A2: MLM aux (optional) ---
        if AUX_MLM:
            masked_input_ids, mlm_labels = mask_inputs_for_mlm(input_ids, input_mask, tokenizer, mlm_prob=MLM_PROB)
            out = model(
                gcn_adj_list,
                gcn_swop_eye,
                masked_input_ids,
                segment_ids,
                input_mask,
                mlm_labels=mlm_labels,
                return_mlm=True,
            )
            logits = out["logits"]
            mlm_logits = out["mlm_logits"]

            cls_loss = F.cross_entropy(logits.view(-1, num_classes), label_ids, weight=loss_weight)
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            loss = cls_loss + MLM_LAMBDA * mlm_loss
        else:
            logits = model(
                gcn_adj_list,
                gcn_swop_eye,
                input_ids,
                segment_ids,
                input_mask,
            )
            if cfg_loss_criterion == "mse":
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), label_ids, weight=loss_weight)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % 40 == 0:
            spent = (time.time() - train_start) / 60.0
            if AUX_MLM:
                print(f"Epoch:{epoch}-{step}/{len(train_dataloader)}, Loss(total):{loss.item():.6f} (spent {spent:.2f}m)")
            else:
                print(f"Epoch:{epoch}-{step}/{len(train_dataloader)}, Loss:{loss.item():.6f} (spent {spent:.2f}m)")

    print("--------------------------------------------------------------")
    valid_loss, valid_acc, valid_f1 = evaluate(model, valid_dataloader, epoch, "Valid_set")
    test_loss, _, test_f1 = evaluate(model, test_dataloader, epoch, "Test_set")

    if test_f1 > test_f1_best:
        test_f1_best = test_f1
        test_f1_best_epoch = epoch

    all_loss_list["train"].append(tr_loss)
    all_loss_list["valid"].append(valid_loss)
    all_loss_list["test"].append(test_loss)
    all_f1_list["valid"].append(valid_f1)
    all_f1_list["test"].append(test_f1)

    print(f"Epoch:{epoch} done. TrainLoss:{tr_loss:.4f}, ValidLoss:{valid_loss:.4f}, TestLoss:{test_loss:.4f}")

    # Save checkpoint if valid improves
    if valid_f1 > perform_metrics_prev:
        to_save = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "valid_acc": valid_acc,
            "lower_case": do_lower_case,
            "perform_metrics": valid_f1,
            "a3_enable": int(A3_ENABLE),
            "aux_mlm": int(AUX_MLM),
        }
        torch.save(to_save, os.path.join(output_dir, model_file_4save))
        perform_metrics_prev = valid_f1
        test_f1_when_valid_best = test_f1
        valid_f1_best_epoch = epoch

print("\n**Optimization Finished!,Total spend:", (time.time() - train_start) / 60.0)
print("**Valid weighted F1: %.3f at %d epoch." % (100 * perform_metrics_prev, valid_f1_best_epoch))
print("**Test weighted F1 when valid best: %.3f" % (100 * test_f1_when_valid_best))
print("**Test weighted F1 (absolute best): %.3f at %d epoch." % (100 * test_f1_best, test_f1_best_epoch))