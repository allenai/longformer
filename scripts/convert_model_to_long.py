
import logging
import os
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)


def create_long_model(save_model_to, attention_window, max_pos):
    '''Convert a roberta model into roberta-long. This function is specific to the roberta model
    (e.g, the +2 positions), but it can be adapted to other models if needed.
    '''

    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def copy_proj_layers(model):
    '''Pretraining on MLM doesn't update the global projection layers. After pretraining,
    copy `query`, `key`, `value` into their global counterpart'''

    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model


def pretrain(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer, file_path=args.val_datapath, block_size=tokenizer.max_len)
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=args.train_datapath, block_size=tokenizer.max_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)
    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval loss: {eval_loss}')

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval loss after pretraining: {eval_loss}')


@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})


def main():
    parser = HfArgumentParser((TrainingArguments, ModelArgs,))

    # NOTE: Pretraining will take 2 days on 1 x 32GB GPU with fp32. Consider changing the config below
    # to use fp16 and to use more gpus to train faster.
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        'script.py',
        '--output_dir', 'tmp',
        '--warmup_steps', '500',
        '--learning_rate', '0.00003',
        # '--lr_scheduler', 'constant',  # in the paper: polynomial_decay with power 3. Both are not supported in HF trainer
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        # NOTE: in the paper, we train for 65k steps, but 3k is probably enough because this is where most of the training
        # happens. However, the learning rate scheduler should be `constant` after warmup
        '--max_steps', '3000',
        '--logging_steps', '500',
        '--save_steps', '500',
        '--max_grad_norm', '5.0',
        '--per_gpu_eval_batch_size', '8',
        # NOTE: num of tokens per batch = batch size (2) x number of gpus (1) x gradient accumulation (32) x seqlen (4096) = 262144
        '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
        '--device', 'cuda0',  # one GPU
        '--gradient_accumulation_steps', '32',
        '--evaluate_during_training',
        '--do_train',
        '--do_eval',
    ])
    training_args.val_datapath = 'wikitext/wikitext-103-raw/wiki.valid.raw'
    training_args.train_datapath = 'wikitext/wikitext-103-raw/wiki.train.raw'

    roberta_base = RobertaForMaskedLM.from_pretrained('roberta-base')
    roberta_base_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    logger.info('Evaluating roberta-base (seqlen: 512) for refernece ...')
    pretrain(training_args, roberta_base, roberta_base_tokenizer, eval_only=True, model_path=None)

    model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
    model, tokenizer = create_long_model(
        save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

    logger.info(f'Loading the model from {model_path}')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaLongForMaskedLM.from_pretrained(model_path)

    logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')
    pretrain(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    logger.info(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)


if __name__ == "__main__":
    main()
