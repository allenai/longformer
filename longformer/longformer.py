from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from transformers.modeling_roberta import RobertaConfig, RobertaModel, RobertaForMaskedLM


class Longformer(RobertaModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerConfig(RobertaConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = False, **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: if equals 'n2', use regular n^2 self-attention else use Longformer attention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(LongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        self.autoregressive = config.autoregressive

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        '''
        The `attention_mask` is changed in BertModel.forward from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        '''
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            has_same_length_extra_indices = (num_extra_indices_per_batch == max_num_extra_indices_per_batch).all()
        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).contiguous().float()
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).contiguous().float()
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False)
            attn_weights += d_mask
        assert list(attn_weights.size()) == [bsz, seq_len, self.num_heads, self.attention_window * 2 + 1]

        # the extra attention
        if extra_attention_mask is not None:
            if has_same_length_extra_indices:
                # a simplier implementation for efficiency
                # k = (bsz, seq_len, num_heads, head_dim)
                selected_k = k.masked_select(extra_attention_mask.unsqueeze(-1).unsqueeze(-1)).view(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
                # selected_k = (bsz, extra_attention_count, num_heads, head_dim)
                # selected_attn_weights = (bsz, seq_len, num_heads, extra_attention_count)
                selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            else:
                # since the number of extra attention indices varies across
                # the batch, we need to process each element of the batch
                # individually
                flat_selected_k = k.masked_select(extra_attention_mask.unsqueeze(-1).unsqueeze(-1))
                selected_attn_weights = torch.ones(
                    bsz, seq_len, self.num_heads, max_num_extra_indices_per_batch, device=k.device, dtype=k.dtype
                )
                selected_attn_weights.fill_(-10000.0)
                start = 0
                for i in range(bsz):
                    end = start + num_extra_indices_per_batch[i] * self.num_heads * self.head_dim
                    # the selected entries for this batch element
                    i_selected_k = flat_selected_k[start:end].view(-1, self.num_heads, self.head_dim)
                    # (seq_len, num_heads, num extra indices)
                    i_selected_attn_weights = torch.einsum('lhd,shd->lhs', (q[i, :, :, :], i_selected_k))
                    selected_attn_weights[i, :, :, :num_extra_indices_per_batch[i]] = i_selected_attn_weights
                    start = end

            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).contiguous().float()
        attn = 0
        if extra_attention_mask is not None and max_num_extra_indices_per_batch > 0:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            if has_same_length_extra_indices:
                selected_v = v.masked_select(
                        extra_attention_mask.unsqueeze(-1).unsqueeze(-1)
                ).view(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            else:
                flat_selected_v = v.masked_select(extra_attention_mask.unsqueeze(-1).unsqueeze(-1))
                # don't worry about masking since this is multiplied by attn_probs, and masking above
                # before softmax will remove masked entries
                selected_v = torch.zeros(
                    bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim, device=v.device, dtype=v.dtype
                )
                start = 0
                for i in range(bsz):
                    end = start + num_extra_indices_per_batch[i] * self.num_heads * self.head_dim
                    i_selected_v = flat_selected_v[start:end].view(-1, self.num_heads, self.head_dim)
                    selected_v[i, :num_extra_indices_per_batch[i], :, :] = i_selected_v
                    start = end
            attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None and max_num_extra_indices_per_batch > 0:
            if has_same_length_extra_indices:
                # query = (seq_len, bsz, dim)
                # extra_attention_mask = (bsz, seq_len)
                # selected_query = (max_num_extra_indices_per_batch, bsz, embed_dim)
                selected_hidden_states = hidden_states.masked_select(extra_attention_mask.transpose(0, 1).unsqueeze(-1)).view(max_num_extra_indices_per_batch, bsz, embed_dim)
                # if *_proj_full exists use them, otherwise default to *_proj
                q = self.query_global(selected_hidden_states)
                k = self.key_global(hidden_states)
                v = self.value_global(hidden_states)
                q /= math.sqrt(self.head_dim)

                q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
                k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
                v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
                attn_weights = torch.bmm(q, k.transpose(1, 2))
                assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]
                if key_padding_mask is not None:
                    attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2),
                        float('-inf'),
                    )
                    attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
                attn_weights_float = F.softmax(attn_weights, dim=-1)
                attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
                selected_attn = torch.bmm(attn_probs, v)
                assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]
                selected_attn = selected_attn.transpose(0, 1).contiguous().view(max_num_extra_indices_per_batch * bsz * embed_dim)

                # now update attn by filling in the relevant indices with selected_attn
                # masked_fill_ only allows floats as values so this doesn't work
                # attn.masked_fill_(extra_attention_mask.transpose(0, 1).unsqueeze(-1), selected_attn)
                attn[extra_attention_mask.transpose(0, 1).unsqueeze(-1).repeat((1, 1, embed_dim))] = selected_attn
            else:
                raise ValueError  # not implemented

        context_layer = attn.transpose(0, 1)
        if self.output_attentions:
            if extra_attention_mask is not None and max_num_extra_indices_per_batch > 0:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if self.output_attentions else (context_layer,)
        return outputs
