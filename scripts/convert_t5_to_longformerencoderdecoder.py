import argparse
import logging
import os
import copy

from transformers import T5Tokenizer, T5ForConditionalGeneration
from longformer.longformer_encoder_decoder import (
    LongformerSelfAttentionForT5,
    LongformerT5Config,
    LongformerT5ForConditionalGeneration,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(save_model_to, base_model, attention_window, max_pos):
    # load base model & tokenizer
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=max_pos)

    # setup config
    config = LongformerT5Config.from_pretrained(base_model)
    config.architectures = [
        "LongformerT5ForConditionalGeneration",
    ]
    # in T5 attention_probs_dropout_prob is dropout_rate
    config.attention_probs_dropout_prob = config.dropout_rate
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    # modify config in model
    # HF T5 includes multiple pointers to the config object
    model.config = model.encoder.config = model.decoder.config = config

    # modify tokenizer
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos

    # modify model architecture
    for i, layer in enumerate(model.encoder.block):
        self_attn = layer.layer[0].SelfAttention

        longformer_self_attn_for_t5 = LongformerSelfAttentionForT5(config, layer_id=i)

        longformer_self_attn_for_t5.longformer_self_attn.query = self_attn.q
        longformer_self_attn_for_t5.longformer_self_attn.key = self_attn.k
        longformer_self_attn_for_t5.longformer_self_attn.value = self_attn.v

        longformer_self_attn_for_t5.longformer_self_attn.query_global = copy.deepcopy(self_attn.q)
        longformer_self_attn_for_t5.longformer_self_attn.key_global = copy.deepcopy(self_attn.k)
        longformer_self_attn_for_t5.longformer_self_attn.value_global = copy.deepcopy(self_attn.v)

        longformer_self_attn_for_t5.output = self_attn.o

        if i == 0:
            longformer_self_attn_for_t5.longformer_self_attn.relative_attention_bias = self_attn.relative_attention_bias

        layer.layer[0].SelfAttention = longformer_self_attn_for_t5

    # save modified model
    logger.info(f"saving model to {save_model_to}")
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    config.save_pretrained(save_model_to)
    return


def main():
    parser = argparse.ArgumentParser(
        description="Convert T5 to LongT5. Replaces T5 encoder's T5Attention with LongformerSelfAttention"
    )
    parser.add_argument(
        "--base_model", type=str, default="t5-small", help="The name or path of the base model you want to convert",
    )
    parser.add_argument("--save_model_to", type=str, required=True, help="The path to save the converted model")
    parser.add_argument(
        "--attention_window",
        type=int,
        default=512,
        help="attention window size for longformer self attention (one sided)",
    )
    parser.add_argument("--max_pos", type=int, default=4096 * 4, help="maximum encoder positions")

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        attention_window=args.attention_window,
        max_pos=args.max_pos,
    )

    tokenizer = T5Tokenizer.from_pretrained(args.save_model_to)
    model = LongformerT5ForConditionalGeneration.from_pretrained(args.save_model_to)
    model.eval()
    model.config.gradient_checkpointing = True

    TXT = "A rose is a rose is a"
    data = tokenizer([TXT], return_tensors="pt", padding="max_length", max_length=2048)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    attention_mask[0, 0:4:2] = 2
    decoder_input_ids = model._shift_right(input_ids[:, :5])

    logits = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)[0]
    probs = logits[0, -1].softmax(dim=0)
    _, predictions = probs.topk(5)
    print(tokenizer.convert_ids_to_tokens(predictions))


if __name__ == "__main__":
    main()
