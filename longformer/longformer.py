from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from longformer.sliding_chunks import (
    sliding_chunks_no_overlap_matmul_qk,
    sliding_chunks_no_overlap_matmul_pv,
)
from transformers.modeling_roberta import RobertaConfig, RobertaModel, RobertaForMaskedLM


class Longformer(RobertaModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        if config.attention_mode == "n2":
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == "n2":
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerConfig(RobertaConfig):
    def __init__(
        self,
        attention_window: List[int] = None,
        attention_dilation: List[int] = None,
        autoregressive: bool = False,
        attention_mode: str = "sliding_chunks",
        **kwargs
    ):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        assert self.attention_mode in ["tvm", "sliding_chunks", "n2", "sliding_chunks_no_overlap"]


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id, bias=True, attention_dim_scale=True):
        super(LongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.attention_dim_scale = attention_dim_scale

        self.query = nn.Linear(config.hidden_size, self.embed_dim, bias=bias)
        self.key = nn.Linear(config.hidden_size, self.embed_dim, bias=bias)
        self.value = nn.Linear(config.hidden_size, self.embed_dim, bias=bias)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim, bias=bias)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim, bias=bias)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim, bias=bias)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive

        if hasattr(config, "relative_attention_num_buckets") and layer_id == 0:
            self.has_relative_attention_bias = True
            self.relative_attention_num_buckets = config.long_relative_attention_num_buckets
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)
        else:
            self.has_relative_attention_bias = False

        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ["tvm", "sliding_chunks", "sliding_chunks_no_overlap"]
        if self.attention_mode in ["sliding_chunks", "sliding_chunks_no_overlap"]:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        """
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        """
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(
                    0, max_num_extra_indices_per_batch, device=num_extra_indices_per_batch.device
                )
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        if self.attention_dim_scale:
            q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        if self.attention_mode == "tvm":
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise False
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(
                remove_from_windowed_attention_mask, -10000.0
            )
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == "tvm":
                d_mask = diagonaled_mm_tvm(
                    ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False,
                )
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [
            self.attention_window * 2 + 1,
            self.attention_window * 3,
        ]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum("blhd,bshd->blhs", (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        if position_bias is None and self.has_relative_attention_bias:
            window_relative_position = torch.arange(
                -self.attention_window, self.attention_window + 1, 1, dtype=torch.long, device=attn_weights.device,
            )  # (2*window+1,)
            window_position_bias = (
                self.relative_attention_bias(
                    relative_position_bucket(
                        window_relative_position,
                        num_buckets=self.relative_attention_num_buckets,
                        max_distance=self.attention_window,
                    )
                )
                .permute(1, 0)[None, None, :, :]
                .repeat(bsz, seq_len, 1, 1)
            )  # (bsz, seq_len, num_heads, 2*window+1)
            perm_global_position_bias_from_g = attn_weights.new_zeros(
                bsz, max_num_extra_indices_per_batch, seq_len, self.num_heads
            )  # (bsz, max_num_extra_indices_per_batch, seq_len, num_heads)
            perm_global_position_bias_to_g = attn_weights.new_zeros(
                bsz, max_num_extra_indices_per_batch, seq_len, self.num_heads
            )  # (bsz, max_num_extra_indices_per_batch, seq_len, num_heads)
            if extra_attention_mask is not None:
                selected_global_memory_position = extra_attention_mask_nonzeros[1][
                    :, None
                ]  # (sum num_extra_indices_per_batch, 1)
                selected_global_query_position = torch.arange(seq_len, dtype=torch.long, device=attn_weights.device)[
                    None, :
                ]  # (1, seq_len)
                selected_global_relative_position = (
                    selected_global_memory_position - selected_global_query_position
                )  # (sum num_extra_indices_per_batch, seq_len)
                selected_global_position_bias_from_g = self.relative_attention_bias(
                    relative_position_bucket(
                        selected_global_relative_position,
                        num_buckets=self.relative_attention_num_buckets,
                        max_distance=self.attention_window,
                    )
                )  # (sum num_extra_indices_per_batch, seq_len, num_heads)
                perm_global_position_bias_from_g[
                    selection_padding_mask_nonzeros
                ] = selected_global_position_bias_from_g  # (bsz, max_num_extra_indices_per_batch, seq_len, num_heads)
                selected_global_position_bias_to_g = self.relative_attention_bias(
                    relative_position_bucket(
                        -selected_global_relative_position,
                        num_buckets=self.relative_attention_num_buckets,
                        max_distance=self.attention_window,
                    )
                )  # (sum num_extra_indices_per_batch, seq_len, num_heads)
                perm_global_position_bias_to_g[
                    selection_padding_mask_nonzeros
                ] = selected_global_position_bias_to_g  # (bsz, max_num_extra_indices_per_batch, seq_len, num_heads)

                global_position_bias_from_g = perm_global_position_bias_from_g.permute(0, 2, 3, 1)
                # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
                global_position_bias_to_g = perm_global_position_bias_to_g.permute(0, 3, 1, 2)
                # (bsz, num_heads, max_num_extra_indices_per_batch, seq_len)

                position_bias = {
                    "window": torch.cat((global_position_bias_from_g, window_position_bias,), dim=-1,),
                    "global": global_position_bias_to_g,
                }
                # window: (bsz, seq_len, num_heads, max_num_extra_indices_per_batch + 2*window+1),
                # global: (bsz, num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                position_bias = {"window": window_position_bias}

        if position_bias is not None:
            attn_weights += position_bias["window"]

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(
                attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0
            )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(
                selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs),
            ).transpose(1, 2)
            attn_probs = attn_probs.narrow(
                -1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch,
            ).contiguous()

        if self.attention_mode == "tvm":
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[
                extra_attention_mask_nonzeros[::-1]
            ]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            if self.attention_dim_scale:
                q /= math.sqrt(self.head_dim)

            q = (
                q.contiguous()
                .view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )  # (bsz * self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = (
                k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            )  # (bsz * self.num_heads, seq_len, head_dim)
            v = (
                v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            )  # (bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [
                bsz * self.num_heads,
                max_num_extra_indices_per_batch,
                seq_len,
            ]
            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            if position_bias is not None:
                attn_weights += position_bias["global"]
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -10000.0,)
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            )  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [
                bsz * self.num_heads,
                max_num_extra_indices_per_batch,
                self.head_dim,
            ]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[
                selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]
            ]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(
                len(selection_padding_mask_nonzeros[0]), -1
            ).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_weights are padded with -10000.0 attention scores
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)

        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)

        return outputs


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_exact=16, max_distance=128):
    """
    Imported from Huggingface transformers, with some modification
    https://github.com/huggingface/transformers/blob/a0a027c2ed53b324cf4d0179ceec88d4ff414d47/src/transformers/models/t5/modeling_t5.py#L344
    Original description below:
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on
    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer
    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        max_exact //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets
