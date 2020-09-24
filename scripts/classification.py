import os

import json
import os
import random
import argparse
from argparse import Namespace
import numpy as np
import glob

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.distributed as dist

from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer, AdamW

from torch.utils.data.dataset import IterableDataset
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support

from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


TEXT_FIELD_NAME = 'text'
LABEL_FIELD_NAME = 'label'

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


def calc_f1(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)
    return f1


class ClassificationDataset(Dataset):

    def __init__(self, file_path, tokenizer, seqlen, num_samples=None, mask_padding_with_zero=True):
        self.data = []
        with open(file_path) as fin:
            for i, line in enumerate(tqdm(fin, desc=f'loading input file {file_path.split("/")[-1]}', unit_scale=1)):
                self.data.append(json.loads(line))
                if num_samples and len(self.data) > num_samples:
                    break
        self.seqlen = seqlen
        self._tokenizer = tokenizer
        all_labels = list(set([e[LABEL_FIELD_NAME] for e in self.data]))
        self.label_to_idx = {e: i for i, e in enumerate(all_labels)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.mask_padding_with_zero = mask_padding_with_zero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._convert_to_tensors(self.data[idx])

    def _convert_to_tensors(self, instance):
        def tok(s):
            return self._tokenizer.tokenize(s, add_prefix_space=True)
        tokens = [self._tokenizer.cls_token_id] + tok(instance[TEXT_FIELD_NAME]) + [self._tokenizer.sep_token]
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids[:self.seqlen]
        input_len = len(token_ids)
        attention_mask = [1 if self.mask_padding_with_zero else 0] * input_len
        padding_length = self.seqlen - input_len
        token_ids = token_ids + ([self._tokenizer.pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)

        assert len(token_ids) == self.seqlen, "Error with input length {} vs {}".format(
            len(token_ids), self.seqlen
        )
        assert len(attention_mask) == self.seqlen, "Error with input length {} vs {}".format(
            len(attention_mask), self.seqlen
        )

        label = self.label_to_idx[instance[LABEL_FIELD_NAME]]

        return (torch.tensor(token_ids), torch.tensor(attention_mask), torch.tensor(label))


class LongformerClassifier(pl.LightningModule):

    def __init__(self, init_args):
        super().__init__()
        if isinstance(init_args, dict):
            # for loading the checkpoint, pl passes a dict (hparams are saved as dict)
            init_args = Namespace(**init_args)
        config = LongformerConfig.from_pretrained(init_args.model_name_or_path)
        config.attention_mode = 'sliding_chunks'
        self.model_config = config
        self.model = Longformer.from_pretrained(init_args.model_name_or_path, config=config)
        self.tokenizer = RobertaTokenizer.from_pretrained(init_args.tokenizer)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        self.hparams = init_args
        self.hparams.seqlen = self.model.config.max_position_embeddings
        self.classifier = nn.Linear(config.hidden_size, init_args.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.model_config.attention_window[0], self.tokenizer.pad_token_id)
        attention_mask[:, 0] = 2  # global attention for the first token
        output = self.model(input_ids, attention_mask=attention_mask)[0]
        # pool the entire sequence into one vector (CLS token)
        output = output[:, 0, :]
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.hparams.num_labels), labels.view(-1))

        return logits, loss

    def _get_loader(self, split, shuffle=True):
        if split == 'train':
            fname = self.hparams.train_file
        elif split == 'dev':
            fname = self.hparams.dev_file
        elif split == 'test':
            fname = self.hparams.test_file
        else:
            assert False
        is_train = split == 'train'

        dataset = ClassificationDataset(
            fname, tokenizer=self.tokenizer, seqlen=self.hparams.seqlen - 2, num_samples=self.hparams.num_samples
        )

        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=shuffle)
        return loader

    def setup(self, mode):
        self.train_loader = self._get_loader("train")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        self.val_dataloader_obj = self._get_loader('dev')
        return self.val_dataloader_obj

    def test_dataloader(self):
        return self._get_loader('test')

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.total_gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.batch_size * self.hparams.grad_accum * num_devices
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.hparams.num_epochs

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        loss = outputs[1]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        logits, tmp_eval_loss = outputs
        preds = logits
        out_label_ids = inputs["labels"]
        return {"val_loss": tmp_eval_loss, "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs) -> tuple:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        preds = torch.cat([x["pred"] for x in outputs], dim=0)
        labels = torch.cat([x["target"] for x in outputs], dim=0)
        preds = torch.argmax(preds, axis=-1)
        accuracy = (preds == labels).int().sum() / float(torch.tensor(preds.shape[-1], dtype=torch.float32, device=labels.device))
        f1 = calc_f1(preds, labels)

        if self.trainer.use_ddp:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
            torch.distributed.all_reduce(accuracy, op=torch.distributed.ReduceOp.SUM)
            accuracy /= self.trainer.world_size
            torch.distributed.all_reduce(f1, op=torch.distributed.ReduceOp.SUM)
            f1 /= self.trainer.world_size
        # accuracy = (preds == out_label_ids).int().sum() / float(torch.tensor(preds.shape[0], dtype=torch.float32, device=out_label_ids.device))
        results = {"val_loss": avg_loss, "f1": f1, "acc": accuracy}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: list) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
        results = {}
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu().item()
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"].detach().cpu().item(), "log": results, "progress_bar": results}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '--model_name_or_path', dest='model_name_or_path', default='longformer-base-4096/', help='path to the model')
    parser.add_argument('--tokenizer', default='roberta-base')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--test_file')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_checkpoint', default=None)
    parser.add_argument('--test_percent_check', default=1.0, type=float)
    parser.add_argument('--val_percent_check', default=1.0, type=float)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--do_predict', default=False, action='store_true')
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--num_labels', default=-1, type=int,
        help='if -1, it automatically finds number of labels.'
        'for larger datasets precomute this and manually set')
    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument("--lr_scheduler",
        default="linear",
        choices=arg_to_scheduler_choices,
        metavar=arg_to_scheduler_metavar,
        type=str,
        help="Learning rate scheduler")
    args = parser.parse_args()
    return args

def get_train_params(args):
    train_params = {}
    train_params["precision"] = 16 if args.fp16 else 32
    if (isinstance(args.gpus, int) and args.gpus > 1) or (isinstance(args.gpus, list ) and len(args.gpus) > 1):
        train_params["distributed_backend"] = "ddp"
    else:
        train_params["distributed_backend"] = None
    train_params["accumulate_grad_batches"] = args.grad_accum
    train_params['track_grad_norm'] = -1
    train_params['val_percent_check'] = args.val_percent_check
    train_params['val_check_interval'] = args.val_check_interval
    train_params['gpus'] = args.gpus
    train_params['max_epochs'] = args.num_epochs
    return train_params

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if ',' in args.gpus:
        args.gpus = list(map(int, args.gpus.split(',')))
        args.total_gpus = len(args.gpus)
    else:
        args.gpus = int(args.gpus)
        args.total_gpus = args.gpus

    def infer_num_labels(args):
        # Dataset will be constructred inside model, here we just want to read labels (seq len doesn't matter here)
        ds = ClassificationDataset(args.train_file, tokenizer=args.tokenizer, seqlen=4096)
        num_labels = len(ds.label_to_idx)
        return num_labels

    if args.test_only:
        print('loading model...')
        if args.num_labels == -1:
            args.num_labels = infer_num_labels(args)
        model = LongformerClassifier.load_from_checkpoint(args.test_checkpoint, num_labels=args.num_labels)
        model.hparams.num_gpus = 1
        model.hparams.total_gpus = 1
        model.hparams = args
        model.hparams.dev_file = args.dev_file
        model.hparams.test_file = args.test_file
        model.hparams.train_file = args.dev_file  # the model won't get trained, pass in the dev file instead to load faster
        trainer = pl.Trainer(gpus=1, test_percent_check=args.test_percent_check, train_percent_check=0.01, val_percent_check=0.01)
        trainer.test(model)

    else:
        if args.num_labels == -1:
            args.num_labels = infer_num_labels(args)
        model = LongformerClassifier(args)

        # default logger used by trainer
        logger = TensorBoardLogger(
            save_dir=args.save_dir,
            version=0,
            name='pl-logs'
        )

        # second part of the path shouldn't be f-string
        filepath = f'{args.save_dir}/version_{logger.version}/checkpoints/' + 'ep-{epoch}_acc-{acc:.3f}'
        checkpoint_callback = ModelCheckpoint(
            filepath=filepath,
            save_top_k=1,
            verbose=True,
            monitor='acc',
            mode='max',
            prefix=''
        )

        extra_train_params = get_train_params(args)

        trainer = pl.Trainer(logger=logger,
                            checkpoint_callback=checkpoint_callback,
                            **extra_train_params)

        trainer.fit(model)

        if args.do_predict:
            # Optionally, predict and write to output_dir
            fpath = glob.glob(checkpoint_callback.dirpath + '/*.ckpt')[0]
            model = LongformerClassifier.load_from_checkpoint(fpath)
            model.args.num_gpus = 1
            model.args.total_gpus = 1
            model.args = args
            model.args.dev_file = args.dev_file
            model.args.test_file = args.test_file
            model.args.train_file = args.dev_file  # the model won't get trained, pass in the dev file instead to load faster
            trainer = pl.Trainer(gpus=1, test_percent_check=1.0, train_percent_check=0.01, val_percent_check=0.01, precision=extra_train_params['precision'])
            trainer.test(model)

if __name__ == '__main__':
    main()
