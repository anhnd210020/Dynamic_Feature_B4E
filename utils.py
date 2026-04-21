import re

import numpy as np
import scipy.sparse as sp
import torch
from nltk.tokenize import TweetTokenizer
from torch.utils.data import Dataset


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
    tknzr = TweetTokenizer(
        reduce_len=True,
        preserve_case=False,
        strip_handles=False,
    )
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
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape)


def get_class_count_and_weight(y, n_classes):
    classes_count = []
    weight = []
    total = len(y)

    for i in range(n_classes):
        count = np.sum(y == i)
        classes_count.append(count)
        weight.append(0.0 if count == 0 else total / (n_classes * count))

    return classes_count, weight


"""
Functions and classes for dataset processing
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


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def example2feature(example, tokenizer, gcn_vocab_map, max_seq_len, gcn_embedding_dim):
    tokens_a = example.text_a.split()
    assert example.text_b is None

    if len(tokens_a) > max_seq_len - 1 - gcn_embedding_dim:
        tokens_a = tokens_a[: (max_seq_len - 1 - gcn_embedding_dim)]

    gcn_vocab_ids = []
    for word in tokens_a:
        if word in gcn_vocab_map:
            gcn_vocab_ids.append(gcn_vocab_map[word])
        else:
            gcn_vocab_ids.append(gcn_vocab_map.get("UNK", -1))

    tokens = ["[CLS]"] + tokens_a + ["[SEP]" for _ in range(gcn_embedding_dim + 1)]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    return InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        gcn_vocab_ids=gcn_vocab_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        confidence=example.confidence,
        label_id=example.label,
    )


class CorpusDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        gcn_vocab_map,
        max_seq_len,
        gcn_embedding_dim,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim
        self.gcn_vocab_map = gcn_vocab_map

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
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
        seq_lens = [len(sample[0]) for sample in batch]
        max_len = np.max(seq_lens)

        def collect(i):
            return [sample[i] for sample in batch]

        def pad_1d(i, target_len):
            return [
                sample[i] + [0] * (target_len - len(sample[i]))
                for sample in batch
            ]

        def pad_gcn_ids(i, target_len):
            return [
                [-1] + sample[i] + [-1] * (target_len - len(sample[i]) - 1)
                for sample in batch
            ]

        batch_input_ids = torch.tensor(pad_1d(0, max_len), dtype=torch.long)
        batch_input_mask = torch.tensor(pad_1d(1, max_len), dtype=torch.long)
        batch_segment_ids = torch.tensor(pad_1d(2, max_len), dtype=torch.long)
        batch_confidences = torch.tensor(collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(collect(4), dtype=torch.long)

        batch_gcn_vocab_ids_padded = np.array(pad_gcn_ids(5, max_len)).reshape(-1)
        batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[batch_gcn_vocab_ids_padded][
            :, :-1
        ]
        batch_gcn_swop_eye = batch_gcn_swop_eye.view(
            len(batch), -1, gcn_vocab_size
        ).transpose(1, 2)

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
        )