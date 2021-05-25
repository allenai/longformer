import argparse
import logging
import os
import copy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from longformer.longformer_encoder_decoder import (
    LongformerSelfAttentionForT5,
    LongformerT5Config,
    LongformerT5ForConditionalGeneration,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(save_model_to, base_model, attention_window, max_pos, relative_attention_num_buckets):
    # load base model & tokenizer
    tokenizer = T5Tokenizer.from_pretrained(base_model, model_max_length=max_pos)
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    print("Base model architecture")
    print(model)

    # setup config
    config = LongformerT5Config.from_pretrained(base_model)
    config.architectures = [
        "LongformerT5ForConditionalGeneration",
    ]
    # in T5 attention_probs_dropout_prob is dropout_rate
    config.attention_probs_dropout_prob = config.dropout_rate
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers
    config.long_relative_attention_num_buckets = relative_attention_num_buckets

    # modify config in model
    # HF T5 includes multiple pointers to the config object
    model.config = copy.deepcopy(config)
    model.encoder.config = copy.deepcopy(config)
    model.encoder.config.use_cache = False
    model.encoder.config.is_encoder_decoder = False
    model.decoder.config = copy.deepcopy(config)
    model.decoder.config.is_decoder = True
    model.decoder.config.is_encoder_decoder = False

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
            half_num_buckets = config.long_relative_attention_num_buckets // 2
            half_t5_buckets = 16
            with torch.no_grad():
                longformer_self_attn_for_t5.longformer_self_attn.relative_attention_bias.weight[
                    :half_num_buckets
                ] = self_attn.relative_attention_bias.weight[half_t5_buckets - 1]
                longformer_self_attn_for_t5.longformer_self_attn.relative_attention_bias.weight[
                    half_num_buckets:
                ] = self_attn.relative_attention_bias.weight[-1]
                longformer_self_attn_for_t5.longformer_self_attn.relative_attention_bias.weight[
                    :half_t5_buckets
                ] = self_attn.relative_attention_bias.weight[:half_t5_buckets]
                longformer_self_attn_for_t5.longformer_self_attn.relative_attention_bias.weight[
                    half_num_buckets + 1 : half_num_buckets + half_t5_buckets
                ] = self_attn.relative_attention_bias.weight[half_t5_buckets + 1 :]

        layer.layer[0].SelfAttention = longformer_self_attn_for_t5

    # save modified model
    logger.info(f"saving model to {save_model_to}")
    model.save_pretrained(save_model_to)
    config.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
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
    parser.add_argument("--num_pos_buckets", type=int, default=40, help="number of relative position buckets")

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.makedirs(args.save_model_to)

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        attention_window=args.attention_window,
        max_pos=args.max_pos,
        relative_attention_num_buckets=args.num_pos_buckets,
    )

    tokenizer = T5Tokenizer.from_pretrained(args.save_model_to)
    # tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    model = LongformerT5ForConditionalGeneration.from_pretrained(args.save_model_to)
    # model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    model.eval()
    model.config.gradient_checkpointing = True
    model.encoder.config.gradient_checkpointing = True
    model.decoder.config.gradient_checkpointing = True
    print("Converted model architecture")
    print(model)

    TXT = "A rose is a rose is a"
    data = tokenizer([TXT], return_tensors="pt", padding="max_length", max_length=args.max_pos)
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
