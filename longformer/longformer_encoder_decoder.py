from typing import List, Optional, Tuple, Dict
from torch import nn, Tensor
from longformer.longformer import LongformerSelfAttention
from transformers.modeling_bart import BartConfig, BartForConditionalGeneration
from transformers.modeling_t5 import T5Config, T5ForConditionalGeneration


class LongformerEncoderDecoderForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == "n2":
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.model.encoder.layers):
                layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)


class LongformerEncoderDecoderConfig(BartConfig):
    def __init__(
        self,
        attention_window: List[int] = None,
        attention_dilation: List[int] = None,
        autoregressive: bool = False,
        attention_mode: str = "sliding_chunks",
        gradient_checkpointing: bool = False,
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
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ["tvm", "sliding_chunks", "n2"]


class LongformerSelfAttentionForBart(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert attn_mask is None

        outputs = self.longformer_self_attn(
            query.transpose(0, 1),  # LongformerSelfAttention expects (bsz, seqlen, embd_dim)
            attention_mask=key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=1) * -1,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
        )

        attn_output = self.output(outputs[0].transpose(0, 1))

        return (attn_output,) + outputs[1:] if len(outputs) == 2 else (attn_output, None)


class LongformerT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == "n2":
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.block):
                layer.layer[0].SelfAttention = LongformerSelfAttentionForT5(config, layer_id=i)


class LongformerT5Config(T5Config):
    def __init__(
        self,
        attention_window: List[int] = None,
        attention_dilation: List[int] = None,
        autoregressive: bool = False,
        attention_mode: str = "sliding_chunks",
        gradient_checkpointing: bool = False,
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
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ["tvm", "sliding_chunks", "n2"]


class LongformerSelfAttentionForT5(nn.Module):
    """
    Replacement for T5Attention, but only for the encoder stack
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.has_relative_attention_bias = layer_id == 0
        self.longformer_self_attn = LongformerSelfAttention(
            config, layer_id=layer_id, bias=False, attention_dim_scale=False
        )
        self.output = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):

        outputs = self.longformer_self_attn(
            input, attention_mask=mask, position_bias=position_bias, output_attentions=output_attentions,
        )

        outputs = (self.output(outputs[0]), None) + outputs[1:]

        return outputs
