import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from pytorch_pretrained_bert.modeling import (
    BertEmbeddings,
    BertEncoder,
    BertModel,
    BertPooler,
)


class VocabGraphConvolution(nn.Module):
    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(self, f"W{i}_vh", nn.Parameter(torch.randn(voc_dim, hid_dim)))

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if (
                name.startswith("W")
                or name.startswith("a")
                or name in ("W", "a", "dense")
            ):
                init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(self, vocab_adj_list, x_dv, add_linear_mapping_term=False):
        fused_h = None

        for i in range(self.num_adj):
            adj = vocab_adj_list[i]
            if not isinstance(adj, torch.Tensor) or not adj.is_sparse:
                raise TypeError("Expected vocab_adj_list elements to be PyTorch sparse tensors")

            w_vh = getattr(self, f"W{i}_vh")
            h_vh = torch.sparse.mm(adj.float(), w_vh)
            h_vh = self.dropout(h_vh)

            h_dh = x_dv.matmul(h_vh)

            if add_linear_mapping_term:
                h_linear = x_dv.matmul(w_vh)
                h_linear = self.dropout(h_linear)
                h_dh = h_dh + h_linear

            fused_h = h_dh if fused_h is None else fused_h + h_dh

        out = self.fc_hc(fused_h)
        return out


def diff_softmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim=dim)
    if not hard:
        return y_soft

    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


class DynamicFusionLayer(nn.Module):
    """
    Token-level fusion between:
    - original BERT token embeddings
    - GCN-enhanced token embeddings
    """

    def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_embeddings, gcn_enhanced_embeddings):
        concat_embeddings = torch.cat(
            [bert_embeddings, gcn_enhanced_embeddings], dim=-1
        )

        gate_logits = self.gate_network(concat_embeddings)
        gate_values = diff_softmax(
            gate_logits,
            tau=self.tau,
            hard=self.hard_gate,
            dim=-1,
        )

        gate_bert_only = gate_values[:, :, 0].unsqueeze(-1)
        gate_gcn_only = gate_values[:, :, 1].unsqueeze(-1)
        gate_weighted = gate_values[:, :, 2].unsqueeze(-1)

        bert_only = bert_embeddings
        gcn_only = gcn_enhanced_embeddings
        weighted_mix = (
            self.fusion_weight * bert_embeddings
            + (1 - self.fusion_weight) * gcn_enhanced_embeddings
        )

        fused_embeddings = (
            gate_bert_only * bert_only
            + gate_gcn_only * gcn_only
            + gate_weighted * weighted_mix
        )
        return fused_embeddings


class ETH_GBertEmbeddings(BertEmbeddings):
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super().__init__(config)
        assert gcn_embedding_dim >= 0

        self.gcn_embedding_dim = gcn_embedding_dim
        self.vocab_gcn = VocabGraphConvolution(
            gcn_adj_dim,
            gcn_adj_num,
            128,
            gcn_embedding_dim,
        )
        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        words_embeddings = self.word_embeddings(input_ids)

        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)

        gcn_words_embeddings = words_embeddings.clone()

        for i in range(self.gcn_embedding_dim):
            tmp_pos = (
                attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
            ) + torch.arange(0, input_ids.shape[0], device=input_ids.device) * input_ids.shape[1]

            gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = (
                gcn_vocab_out[:, :, i]
            )

        fused_word_embeddings = self.dynamic_fusion_layer(
            words_embeddings,
            gcn_words_embeddings,
        )

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = fused_word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ETH_GBertModel(BertModel):
    def __init__(
        self,
        config,
        gcn_adj_dim,
        gcn_adj_num,
        gcn_embedding_dim,
        num_labels,
        output_attentions=False,
        keep_multihead_output=False,
    ):
        super().__init__(config)

        self.embeddings = ETH_GBertEmbeddings(
            config,
            gcn_adj_dim,
            gcn_adj_num,
            gcn_embedding_dim,
        )
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.output_attentions = (
            config.output_attentions if hasattr(config, "output_attentions")
            else output_attentions
        )
        self.keep_multihead_output = (
            config.keep_multihead_output if hasattr(config, "keep_multihead_output")
            else keep_multihead_output
        )

        self.will_collect_cls_states = False
        self.all_cls_states = []

        self.apply(self.init_bert_weights)

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        head_mask=None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            token_type_ids,
            attention_mask,
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers,
                    -1,
                    -1,
                    -1,
                    -1,
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_args = {}
        if "head_mask" in inspect.signature(self.encoder.forward).parameters:
            encoder_args["head_mask"] = head_mask

        if self.output_attentions:
            output_all_encoded_layers = True

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **encoder_args,
        )

        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        logits = self.classifier(self.dropout(pooled_output))

        if self.output_attentions:
            return all_attentions, logits

        return logits