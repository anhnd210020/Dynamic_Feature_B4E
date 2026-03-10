import re
import math

import numpy as np
import scipy.sparse as sp
import torch
from nltk.tokenize import TweetTokenizer
from torch.utils.data import Dataset

# ----------------------------
# (2.1) Binning helpers
# ----------------------------
addr_pat = re.compile(r"0x[a-fA-F0-9]{40}")

def to_val_bin(x, n_bins):
    # log1p binning, clamp
    x = max(float(x), 0.0)
    b = int(min(n_bins - 1, max(0, math.floor(math.log1p(x) * (n_bins / 10.0)))))
    return b

def to_dt_bin(x, n_bins):
    # dt often large; use log1p too
    x = max(float(x), 0.0)
    b = int(min(n_bins - 1, max(0, math.floor(math.log1p(x) * (n_bins / 10.0)))))
    return b


"""
General functions
"""

def del_http_user_tokenize(tweet):
    space_pattern = r"\s+"
    url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    mention_regex = r"@[\w\-]+"
    tweet = re.sub(space_pattern, " ", tweet)
    tweet = re.sub(url_regex, "", tweet)
    tweet = re.sub(mention_regex, "", tweet)
    return tweet

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_tweet_tokenize(string):
    tknzr = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=False)
    tokens = tknzr.tokenize(string.lower())
    return " ".join(tokens).strip()

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    rowsum[rowsum == 0] = 1.0
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def sparse_scipy2torch(coo_sparse):
    coo = coo_sparse.tocoo()
    i = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    v = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, size=coo.shape)

def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    for i in range(n_classes):
        count = np.sum(y == i)
        classes_count.append(count)
        # tránh chia 0 nếu dataset có class trống
        weight.append(len(y) / (n_classes * count)) if count > 0 else weight.append(0.0)
    return classes_count, weight


"""
Functions and Classes for read and organize data set
"""

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.confidence = confidence
        self.label = label

class InputFeatures(object):
    def __init__(
        self,
        guid,
        tokens,
        input_ids,
        gcn_vocab_ids,
        input_mask,
        segment_ids,
        confidence,
        label_id,
    ):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.gcn_vocab_ids = gcn_vocab_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.confidence = confidence
        self.label_id = label_id


def example2feature(
    example,
    tokenizer,
    gcn_vocab_map,
    max_seq_len,
    gcn_embedding_dim,
    tok_cfg=None,
    addr_to_tok=None,
):
    """
    Field-aware tokenization:
    - address: map via addr_to_tok (topK) else ADDR_OOV (default "[UNK]")
    - key=value parsing: tag/in_out/value(amount)/2-gram..5-gram -> bin tokens from tok_cfg
    - fallback: keep token as-is
    """
    tok_cfg = tok_cfg or {}
    addr_to_tok = addr_to_tok or {}

    tokens_a_raw = example.text_a.strip().split()
    assert example.text_b is None

    out_tokens = []
    gcn_vocab_ids = []

    def push(tok: str):
        out_tokens.append(tok)
        if tok in gcn_vocab_map:
            gcn_vocab_ids.append(gcn_vocab_map[tok])
        else:
            gcn_vocab_ids.append(gcn_vocab_map.get("UNK", -1))

    def is_true(v: str) -> bool:
        return v in ("1", "true", "True", "yes", "Yes", "y", "Y")

    for w in tokens_a_raw:
        # address?
        if addr_pat.fullmatch(w):
            tok = addr_to_tok.get(w, tok_cfg.get("ADDR_OOV", "[UNK]"))
            push(tok)
            continue

        # key=value?
        if "=" in w:
            k, v = w.split("=", 1)
            k = k.lower()

            # TAG
            if k in ("tag",):
                # require tok_cfg["TAG1"], tok_cfg["TAG0"]
                push(tok_cfg.get("TAG1", "[UNK]") if is_true(v) else tok_cfg.get("TAG0", "[UNK]"))
                continue

            # IN/OUT
            if k in ("in_out", "inout"):
                push(tok_cfg.get("IN1", "[UNK]") if is_true(v) else tok_cfg.get("IN0", "[UNK]"))
                continue

            # VALUE/AMOUNT
            if k in ("value", "amount", "amt"):
                bins = tok_cfg.get("VAL_BINS", None)
                if isinstance(bins, (list, tuple)) and len(bins) > 0:
                    try:
                        b = to_val_bin(v, len(bins))
                        push(bins[b])
                    except Exception:
                        push(bins[0])
                else:
                    push("[UNK]")
                continue

            # DT bins
            if k in ("2-gram", "2gram", "dt2"):
                bins = tok_cfg.get("DT2_BINS", None)
                if isinstance(bins, (list, tuple)) and len(bins) > 0:
                    try:
                        push(bins[to_dt_bin(v, len(bins))])
                    except Exception:
                        push(bins[0])
                else:
                    push("[UNK]")
                continue

            if k in ("3-gram", "3gram", "dt3"):
                bins = tok_cfg.get("DT3_BINS", None)
                if isinstance(bins, (list, tuple)) and len(bins) > 0:
                    try:
                        push(bins[to_dt_bin(v, len(bins))])
                    except Exception:
                        push(bins[0])
                else:
                    push("[UNK]")
                continue

            if k in ("4-gram", "4gram", "dt4"):
                bins = tok_cfg.get("DT4_BINS", None)
                if isinstance(bins, (list, tuple)) and len(bins) > 0:
                    try:
                        push(bins[to_dt_bin(v, len(bins))])
                    except Exception:
                        push(bins[0])
                else:
                    push("[UNK]")
                continue

            if k in ("5-gram", "5gram", "dt5"):
                bins = tok_cfg.get("DT5_BINS", None)
                if isinstance(bins, (list, tuple)) and len(bins) > 0:
                    try:
                        push(bins[to_dt_bin(v, len(bins))])
                    except Exception:
                        push(bins[0])
                else:
                    push("[UNK]")
                continue

        # fallback
        push(w)

    # truncate (keep same as old behavior)
    max_a = max_seq_len - 1 - gcn_embedding_dim
    if len(out_tokens) > max_a:
        out_tokens = out_tokens[:max_a]
        gcn_vocab_ids = gcn_vocab_ids[:max_a]

    # Build BERT-like tokens
    tokens = ["[CLS]"] + out_tokens + ["[SEP]" for _ in range(gcn_embedding_dim + 1)]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence=example.confidence,
        label_id=example.label,
    )
    return feat


class CorpusDataset(Dataset):
    # (2.2) updated signature
    def __init__(
        self,
        examples,
        tokenizer,
        gcn_vocab_map,
        max_seq_len,
        gcn_embedding_dim,
        tok_cfg=None,
        addr_to_tok=None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim
        self.gcn_vocab_map = gcn_vocab_map

        self.tok_cfg = tok_cfg or {}
        self.addr_to_tok = addr_to_tok or {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
            tok_cfg=self.tok_cfg,
            addr_to_tok=self.addr_to_tok,
        )
        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
        )

    def pad(self, batch):
        gcn_vocab_size = len(self.gcn_vocab_map)
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = int(np.max(seqlen_list))

        f_collect = lambda x: [sample[x] for sample in batch]
        f_pad = lambda x, seqlen: [
            sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch
        ]
        f_pad2 = lambda x, seqlen: [
            [-1] + sample[x] + [-1] * (seqlen - len(sample[x]) - 1)
            for sample in batch
        ]

        batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
        batch_confidences = torch.tensor(f_collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(f_collect(4), dtype=torch.long)

        batch_gcn_vocab_ids_paded = np.array(f_pad2(5, maxlen)).reshape(-1)

        # NOTE: -1 indexes the last element in torch indexing.
        # This matches your old behavior where -1 maps to "padding row" and then we slice [:, :-1]
        batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[batch_gcn_vocab_ids_paded][:, :-1]
        batch_gcn_swop_eye = batch_gcn_swop_eye.view(len(batch), -1, gcn_vocab_size).transpose(1, 2)

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
        )