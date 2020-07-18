import argparse
import glob
import os
import random
import logging
import numpy as np
import math
from tqdm import tqdm
import time
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import DataCollatorForLanguageModeling
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as ptl
from pytorch_lightning.logging.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DONE: reproduce RoBERTa numbers on the Longformer corpus
# DONE: testing ddp single machine
# DONE: testing ddp multiple machines
# DONE: testing resume from checkpoint
# TODO: try on a single TPU
# TODO: try on a TPU-pod


class MMapTextDataset(Dataset):
    def __init__(self, mmap_filename, chunk_size):
        self.num_instances = np.memmap(mmap_filename, mode='r', dtype=np.uint16).shape[0] // chunk_size
        # defer loading the token_ids memmap until after the first __getitem__ call.
        # when spawning new processes for ddp, there is a hard limit in python < 3.8 that
        # pickle files need to be < 4GB. By waiting until after the first __getitem__ we
        # don't have to pickle the memmap
        self.token_ids = None
        self._mmap_filename = mmap_filename
        self._chunk_size = chunk_size

    def __len__(self):
        return self.num_instances

    def __getitem__(self, i):
        if self.token_ids is None:
            self.token_ids = np.memmap(self._mmap_filename, mode='r', dtype=np.uint16,
                                       shape=(self.num_instances, self._chunk_size))
        return torch.tensor(self.token_ids[i, :].astype(np.int32), dtype=torch.long)

    @staticmethod
    def raw_text_to_mmap(args):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        assert len(tokenizer) < 65535  # will use uint16 to store token ids
        all_files = glob.glob(f'{args.input_dir}/*.txt')

        if os.path.exists(f'{args.input_dir}/cache/'):
            logger.info("Cache already exists. Remove the cache directory to regenerate")
            return
        os.mkdir(f'{args.input_dir}/cache/')

        # TODO: process each shared in a separate worker and save their output to files

        chunks_list = []
        for fname in tqdm(all_files):
            with open(fname, 'r') as fin:
                current_chunk = [tokenizer.bos_token]
                for line in tqdm(fin):
                    if line.strip() == '':  # drop empty lines
                        continue
                    tokens = tokenizer.tokenize(line)  # each line is one document
                    # generate chunks of length args.seqlen. The last chunk will be padded.
                    # padding last chunk is not great for longformer because many chunks will be mostly padding

                    for token in tokens:
                        if len(current_chunk) == args.seqlen - 1:  # chunk is full
                            current_chunk.append(tokenizer.eos_token)
                            chunks_list.append(current_chunk)
                            current_chunk = [tokenizer.bos_token]
                        current_chunk.append(token)
                    if args.padded_chunks:
                        # fill the rest of the seqlen with pad
                        current_chunk.extend([tokenizer.pad_token] * (args.seqlen - len(current_chunk)))
                        current_chunk[args.seqlen - 1] = tokenizer.eos_token
                        chunks_list.append(current_chunk)
                        current_chunk = [tokenizer.bos_token]
                    else:
                        # one long doc with sep inbetween
                        if len(current_chunk) < args.seqlen - 1:
                            current_chunk.append(tokenizer.sep_token)
        random.shuffle(chunks_list)
        val_count = int(args.train_dev_split * len(chunks_list))
        val_chunks = chunks_list[:val_count]
        train_chunks = chunks_list[val_count:]

        def _tokenized_text_to_mmap(output_fname, chunks_list):
            num_chunks = len(chunks_list)
            all_token_ids = np.empty((num_chunks, args.seqlen), dtype=np.uint16)
            for k, chunk in enumerate(tqdm(chunks_list)):
                token_ids = tokenizer.convert_tokens_to_ids(chunk)
                assert len(token_ids) == args.seqlen
                all_token_ids[k, :] = [int(t) for t in token_ids]
            fp = np.memmap(output_fname, dtype=np.uint16, mode='w+', shape=(num_chunks, args.seqlen))
            fp[:, :] = all_token_ids[:, :]
            fp.flush()
            del fp

        _tokenized_text_to_mmap(f'{args.input_dir}/cache/train.bin', train_chunks)
        _tokenized_text_to_mmap(f'{args.input_dir}/cache/val.bin', val_chunks)


class Pretrainer(ptl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.args = hparams
        self.hparams = self.args

        self.model = AutoModelWithLMHead.from_pretrained(args.model)
        self.config = self.model.config
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pad_token_id = tokenizer.pad_token_id

        logger.info(f'Creating dataset cache from dir {self.args.input_dir}. This could be slow the first time.')
        MMapTextDataset.raw_text_to_mmap(args)

        # TODO: add support for other objective functions
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=self.args.mlm_prob
        )
        self.start_time = 0

    def forward(self, input_ids=None, labels=None):
        # get the padding mask - 1 for NOT masked, 0 for MASKED/PAD
        attention_mask = (input_ids != self.pad_token_id).int()

        # output is loss, prediction_scores, hidden_states
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, masked_lm_labels=labels)
        return output[0]  # loss

    def training_step(self, batch, batch_nb):
        loss = self(**batch)
        input_ids = batch['input_ids']
        tensorboard_logs = {
            'input_size': input_ids.numel(),
            'memory': torch.cuda.memory_allocated(loss.device) / 1024 ** 3,
            'mlm_loss': loss.detach(),
            'mlm_bpc': loss.detach()/math.log(2),
            'mlm_perplexity': torch.exp(loss).detach(),
            'token_per_step': input_ids.numel() * self.args.grad_accum * self.trainer.world_size,
        }
        if self.start_time != 0:
            elapsed_time = time.time() - self.start_time
            tensorboard_logs['second_per_batch'] = elapsed_time
        self.start_time = time.time()

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # TODO: log how long evaluation takes
        self.start_time = 0  # reset training_step timer
        loss = self(**batch)
        tensorboard_logs = {
            'val_mlm_loss': loss.detach(),
        }
        return {'val_loss': tensorboard_logs["val_mlm_loss"], 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['log']['val_mlm_loss'] for x in outputs if 'val_mlm_loss' in x['log']]).mean()
        if self.use_ddp:
            # TODO: PTL is already doing this. Is it still needed here?
            # https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pytorch_lightning/metrics/converters.py#L251
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= torch.distributed.get_world_size()
        logs = {'val_mlm_loss': avg_loss}
        return {'log': logs, 'progress_bar': logs, "val_loss": avg_loss}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.train_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_loader(self, fname, is_train):
        dataset = MMapTextDataset(fname, chunk_size=self.args.seqlen)

        # TODO: consider `replace_sampler_ddp=True` and removing the following if statement
        if self.trainer.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
            shuffle = False
        else:
            sampler = None
            shuffle = is_train

        loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.args.num_workers,
                collate_fn=self.data_collator,
                drop_last=is_train,
        )
        return loader

    def train_dataloader(self):
        return self._get_loader(f'{self.args.input_dir}/cache/train.bin', True)

    def val_dataloader(self):
        return self._get_loader(f'{self.args.input_dir}/cache/val.bin', False)

    def grad_norm(self, norm_type):
        # Override PTL `grad_norm` function to only return `total_grad_norm` instead norms of individual params
        # TODO: grad_norm reporting needs to take fp16 loss scale into account
        all_norms = [float(p.grad.data.norm(float(norm_type))) for p in self.parameters() if p.grad is not None]
        return {'total_grad_norm': float(torch.tensor(all_norms).norm(norm_type))}

    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=3)

        # Dataset. Some of these params are only useful when generating the dataset cache
        parser.add_argument("--input_dir", type=str, required=True)
        parser.add_argument("--train_dev_split", type=float, default=0.05)
        parser.add_argument("--padded_chunks", type=bool, default=False)
        parser.add_argument("--seqlen", type=int, default=512)
        parser.add_argument("--mlm_prob", type=float, default=0.15)

        # HF model loading
        parser.add_argument("--tokenizer", type=str, default='roberta-base')
        parser.add_argument("--model", type=str, default='roberta-base')

        # Checkpointing and logging
        parser.add_argument("--save_dir", type=str, default='runs/')
        parser.add_argument("--save_prefix", type=str, required=True,
                            help="path of output directory is --save_dir/--save_prefix")
        parser.add_argument("--resume", type=str, default=None,  # It is better to use a different output dir.
                            help="Path to a checkpoint to load model weights and training state. It overwrites args")
        parser.add_argument("--resume_model_only", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")

        # Training hyperparams
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--train_steps", type=int, default=3000, help='# training grad. updates')
        parser.add_argument("--warmup_steps", type=int, default=1000, help='# warmup grad. updates')
        parser.add_argument("--val_every", type=int, default=1000, help='# training grad. updates between evaluations')
        parser.add_argument("--val_batches", type=int, default=1000, help='# evaluation **batches**')
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_epsilon", type=float, default=1e-6)
        parser.add_argument("--grad_clip", type=float, default=0)  # TODO: test this with fp16. Likely not working

        # RoBERTa's tokens_per_step = 2^18 = 512(seqlen) x 1(gpu_count) x 32(batch_size) x 16(grad_accum)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--grad_accum", type=int, default=16)

        # Compute resources
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--gpu_count", type=int, default=1,  # `--gpus` is reserved for internal use by PTL
                            help="Number of gpus. This respects `CUDA_VISIBLE_DEVICES`")

        # For multi-node training, use the PyTorch launch script. The script and instructions can be found here:
        # https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.
        # To run PTL in a mode compatible with the launch script, two things are needed:
        #   - pass the argument `--use_env` to `torch.distributed.launch`
        #   - make sure `--nproc_per_node` matches `--gpu_count` and `--nnodes` matches `--node_count`.
        # For example, to run on 2 nodes, 3 gpus each, the command line on node rank 1 would be like:
        #   >>>> python -m torch.distributed.launch  \
        #               --use_env  --nnodes 2  --nproc_per_node 3  \
        #               --node_rank 1  --master_addr s2-server4  --master_port 12343  \
        #               scripts/pretrain.py  \
        #               --gpu_count 2  --node_count 2  \
        #               --input_dir my_data_dir  --save_prefix test_multinode
        parser.add_argument("--node_count", type=int, default=1,
                            help="Number of nodes. It needs to match --nnodes of torch.distributed.launch")
        parser.add_argument("--num_tpu_cores", type=int, default=None)

        return parser


def main(args):
    random.seed(args.seed * 10)
    np.random.seed(args.seed * 100)
    torch.manual_seed(args.seed * 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed * 10000)

    if args.resume_model_only is not None:
        pretrainer = Pretrainer.load_from_checkpoint(args.resume_model_only, args)
    else:
        pretrainer = Pretrainer(args)

    # logger here is a SummaryWritter for tensorboard
    # it is used by the trainer, and certain return variables
    # from the model are automatically logged
    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        # model saved to filepath/prefix_....
        filepath=os.path.join(args.save_dir, args.save_prefix, 'checkpoint'),
        prefix='',
        save_top_k=3,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        period=-1,  # to allow multiple checkpoints per epoch
    )

    args.val_every *= args.grad_accum  # PTL is expecting number of batches_per_gpu
    trainer = ptl.Trainer(
        gpus=args.gpu_count,
        num_nodes=args.node_count,
        num_tpu_cores=args.num_tpu_cores,
        distributed_backend='ddp' if (args.gpu_count > 1 or args.node_count > 1) else None,
        replace_sampler_ddp=False,
        track_grad_norm=2,
        max_epochs=10000, min_epochs=0, max_steps=args.train_steps,  # run for many epochs, but stop after max_steps
        val_check_interval=args.val_every, limit_val_batches=args.val_batches,
        early_stop_callback=None,
        row_log_interval=10,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=args.grad_accum,
        resume_from_checkpoint=args.resume,
        gradient_clip_val=args.grad_clip,
        precision=16, amp_level='O2',
        callbacks=[LearningRateLogger()],
    )
    trainer.fit(pretrainer)


if __name__ == "__main__":
    parser = Pretrainer.add_args(argparse.ArgumentParser(description="pretrain"))
    args = parser.parse_args()
    main(args)
