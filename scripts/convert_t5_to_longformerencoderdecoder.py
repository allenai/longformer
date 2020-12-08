import argparse
import logging
import os

from transformers import T5Tokenizer

from transformers import T5ForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from longformer.longformer_t5_encoder_decoder import LongformerSelfAttentionForT5, LongformerEncoderDecoderConfigT5
from longformer.longformer_t5_encoder_decoder import LongformerEncoderDecoderForConditionalGenerationT5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    attention_window,
    max_pos
):
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos)
    config = LongformerEncoderDecoderConfigT5.from_pretrained(base_model)
    model.config = config

    # in T5 attention_probs_dropout_prob is dropout_rate, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.dropout_rate
    config.architectures = ['LongformerEncoderDecoderForConditionalGenerationT5', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    # current_max_pos, embed_size = model.model.embed_positions.weight.shape
    # assert current_max_pos == config.max_position_embeddings + 2

    # config.max_encoder_position_embeddings = max_pos
    # config.max_decoder_position_embeddings = config.max_position_embeddings
    # del config.max_position_embeddings
    # # TODO: check what's the deal with T5 here.
    # max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    # assert max_pos >= current_max_pos

    # # allocate a larger position embedding matrix for the encoder
    # new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_pos - 2
    # while k < max_pos - 1:
    #     new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
    #     k += step
    # model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

    # allocate a larger position embedding matrix for the decoder
    # new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_pos - 2
    # while k < max_pos - 1:
    #     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
    #     k += step
    # model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

    # replace the `modeling_t5.T5Attention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers
    # model.encoder.block = model.encoder.block[:1]

    for i, layer in enumerate(model.encoder.block):
        self_attn = layer.layer[0].SelfAttention

        longformer_self_attn_for_t5 = LongformerSelfAttentionForT5(config, layer_id=i)

        longformer_self_attn_for_t5.longformer_self_attn.query = self_attn.q
        longformer_self_attn_for_t5.longformer_self_attn.key = self_attn.k
        longformer_self_attn_for_t5.longformer_self_attn.value = self_attn.v

        longformer_self_attn_for_t5.longformer_self_attn.query_global = self_attn.q
        longformer_self_attn_for_t5.longformer_self_attn.key_global = self_attn.k
        longformer_self_attn_for_t5.longformer_self_attn.value_global = self_attn.v

        longformer_self_attn_for_t5.output = self_attn.o

        layer.layer[0].SelfAttention = longformer_self_attn_for_t5

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert T5 to LongT5. Replaces T5 encoder's T5Attention with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='t5-large',
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default='t5-large',
        help='The name or path of the tokenizer'
    )
    parser.add_argument(
        '--save_model_to',
        type=str,
        required=True,
        help='The path to save the converted model'
    )
    parser.add_argument(
        '--attention_window',
        type=int,
        default=512,
        help='attention window size for longformer self attention (one sided)'
    )
    parser.add_argument(
        '--max_pos',
        type=int,
        default=4096 * 4,
        help='maximum encoder positions'
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        attention_window=args.attention_window,
        max_pos=args.max_pos
    )

    tokenizer = T5Tokenizer.from_pretrained(args.save_model_to)
    TXT = "My friends are <mask> but they eat too many carbs."
    model = LongformerEncoderDecoderForConditionalGenerationT5.from_pretrained(args.save_model_to)
    model.encoder.config.gradient_checkpointing = True
    model.decoder.config.gradient_checkpointing = True
    data = tokenizer([TXT], return_tensors='pt', padding='max_length', max_length=2048)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    decoder_input_ids = shift_tokens_right(input_ids[:, :5], tokenizer.pad_token_id)
    logits = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False)[0]
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(5)
    print(tokenizer.convert_ids_to_tokens(predictions))


if __name__ == "__main__":
    main()
