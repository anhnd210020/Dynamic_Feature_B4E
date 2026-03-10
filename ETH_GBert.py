# ETH_GBert.py
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from env_config import env_config

# for huggingface transformers 0.6.2
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
        for n, p in self.named_parameters():
            if n.startswith("W") or n.startswith("a") or n in ("W", "a", "dense"):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        fused_H = None
        for i in range(self.num_adj):
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")

            H_vh = torch.sparse.mm(vocab_adj_list[i].float(), getattr(self, f"W{i}_vh"))  # [V,h]
            H_vh = self.dropout(H_vh)

            # X_dv: [B, V, D], H_vh: [V,h] -> [B, V, h] or compatible depending on X_dv
            H_dh = X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(getattr(self, f"W{i}_vh"))
                H_linear = self.dropout(H_linear)
                H_dh = H_dh + H_linear

            fused_H = H_dh if fused_H is None else (fused_H + H_dh)

        out = self.fc_hc(fused_H)
        return out


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Soft or straight-through hard softmax.
    NOTE: This is NOT true Gumbel-Softmax (no gumbel noise), kept to minimize changes.
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class DynamicFusionLayer(nn.Module):
    """
    Token-level fusion of:
      - bert_embeddings (original token embeddings)
      - gcn_enhanced_embeddings (token embeddings with injected vocab-gcn outputs)
    with 3 options: bert-only / gcn-enhanced / weighted-mix
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
        # bert_embeddings, gcn_enhanced_embeddings: [B, L, D]
        concat_embeddings = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)  # [B,L,2D]

        gate_logits = self.gate_network(concat_embeddings)  # [B,L,3]
        gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

        gate_bert_only = gate_values[:, :, 0].unsqueeze(-1)
        gate_gcn_enhanced = gate_values[:, :, 1].unsqueeze(-1)
        gate_weighted = gate_values[:, :, 2].unsqueeze(-1)

        embeddings_bert_only = bert_embeddings
        embeddings_gcn_enhanced = gcn_enhanced_embeddings
        embeddings_weighted = self.fusion_weight * bert_embeddings + (1 - self.fusion_weight) * gcn_enhanced_embeddings

        fused_embeddings = (
            gate_bert_only * embeddings_bert_only
            + gate_gcn_enhanced * embeddings_gcn_enhanced
            + gate_weighted * embeddings_weighted
        )
        return fused_embeddings


class ETH_GBertEmbeddings(BertEmbeddings):
    """
    BERT embeddings with vocab-GCN injection and token-level dynamic fusion.
    """
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super().__init__(config)
        assert gcn_embedding_dim >= 0
        self.gcn_embedding_dim = gcn_embedding_dim

        self.vocab_gcn = VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim)
        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
        # words embeddings: [B,L,D]
        words_embeddings = self.word_embeddings(input_ids)

        # vocab_input: [B, V, D]
        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)  # expected [B, V, gcn_dim]-compatible

        # inject gcn outputs into the last gcn_embedding_dim "pseudo token" slots
        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            tmp_pos = (
                (attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i)
                + torch.arange(0, input_ids.shape[0], device=input_ids.device) * input_ids.shape[1]
            )
            gcn_words_embeddings.flatten(0, 1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        new_words_embeddings = self.dynamic_fusion_layer(words_embeddings, gcn_words_embeddings)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = new_words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ETH_GBertModel(BertModel):
    """
    - No SET branch.
    - GRU pooled fusion stays (as you already adopted).
    - Adds MLM head for A2 (aux MLM): return dict when return_mlm=True or mlm_labels is provided.
    """

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

        self.embeddings = ETH_GBertEmbeddings(config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.output_attentions = config.output_attentions if hasattr(config, "output_attentions") else False
        self.keep_multihead_output = config.keep_multihead_output if hasattr(config, "keep_multihead_output") else False

        # ---- GRU branch (Cách 1): GRU over last hidden states ----
        self.use_gru = True
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # ---- pooled-level 2-way fusion gate: BERT pooled vs GRU pooled ----
        self.gru_fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 2),  # logits for {BERT, GRU}
        )
        self.latest_gru_gate = None

        # ---- A2: MLM head (BERT-style) ----
        self.mlm_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_act = nn.GELU()
        self.mlm_ln = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # tie weights with word embedding (common MLM trick)
        self.mlm_decoder.weight = self.embeddings.word_embeddings.weight

        self.apply(self.init_bert_weights)

    @staticmethod
    def _last_valid_from_mask(seq_out: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        seq_out: [B, L, D]
        attention_mask: [B, L] (1 for valid, 0 for pad)
        returns: [B, D] last valid hidden per sample
        """
        lengths = attention_mask.long().sum(dim=1).clamp(min=1)  # [B]
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, seq_out.size(-1))  # [B,1,D]
        return seq_out.gather(1, idx).squeeze(1)  # [B,D]

    def forward(
        self,
        vocab_adj_list,
        gcn_swop_eye,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        head_mask=None,
        return_branches: bool = False,
        *,
        # A2 flags
        mlm_labels=None,     # [B,L] with -100 for ignore positions
        return_mlm: bool = False,
        **kwargs,
    ):
        # optional disable GCN injection
        if getattr(env_config, "GCN_DISABLE_IN_EMB", 0):
            vocab_adj_list = [adj * 0 for adj in vocab_adj_list]

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
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
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

        encoded = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **encoder_args,
        )

        if self.output_attentions:
            all_attentions, encoded_layers = encoded
        else:
            encoded_layers = encoded

        if isinstance(encoded_layers, (list, tuple)):
            last_hidden = encoded_layers[-1]  # [B,L,D]
        else:
            last_hidden = encoded_layers      # [B,L,D]

        # ----- pooled BERT -----
        eB = self.pooler(last_hidden)  # [B,D]

        # ----- GRU branch -----
        gru_out, _ = self.gru(last_hidden)  # [B,L,D]
        eR = self._last_valid_from_mask(gru_out, attention_mask)  # [B,D]

        # ----- pooled fusion -----
        gate_logits = self.gru_fusion_gate(torch.cat([eB, eR], dim=-1))  # [B,2]
        gate = F.softmax(gate_logits, dim=-1)                            # [B,2]
        self.latest_gru_gate = gate.detach()

        fused = gate[:, 0:1] * eB + gate[:, 1:2] * eR  # [B,D]

        # ----- classification logits -----
        logits = self.classifier(self.dropout(fused))  # [B,C]

        # ----- optional MLM logits -----
        need_mlm = return_mlm or (mlm_labels is not None)
        if need_mlm:
            x = self.mlm_transform(last_hidden)
            x = self.mlm_act(x)
            x = self.mlm_ln(x)
            mlm_logits = self.mlm_decoder(x) + self.mlm_bias  # [B,L,V]
            out = {"logits": logits, "mlm_logits": mlm_logits}
            if return_branches:
                out.update({"eB": eB, "eR": eR, "gate": gate})
            if self.output_attentions:
                return all_attentions, out
            return out

        if self.output_attentions:
            return all_attentions, logits

        if return_branches:
            return {"logits": logits, "eB": eB, "eR": eR, "gate": gate}

        return logits
