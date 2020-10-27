import argparse
import gc
import glob
import logging
import math
import os
import random
import time

import numpy as np
import pytorch_lightning as ptl
import torch
from pytorch_lightning.logging.test_tube import TestTubeLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from longformer.ptl_model_checkpoint import ModelCheckpoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DONE: reproduce RoBERTa numbers on the Longformer corpus
# DONE: testing ddp single machine
# DONE: testing ddp multiple machines
# DONE: testing resume from checkpoint
# TODO: try on a TPU-pod
# TODO: run on beaker on ai2-server1/2
# TODO: optimize longformer for TPUs
# TODO: verify that global attention is correct
# TODO: implement non-contiguous global attention


try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class MMapTextDataset(Dataset):
    def __init__(self, mmap_filename, chunk_size, bos_token_id, eos_token_id):
        # `chunk_size - 2` to reserve space for <s> and </s>
        self.num_instances = np.memmap(mmap_filename, mode='r', dtype=np.uint16).shape[0] // (chunk_size - 2)
        # defer loading the token_ids memmap until after the first __getitem__ call.
        # when spawning new processes for ddp, there is a hard limit in python < 3.8 that
        # pickle files need to be < 4GB. By waiting until after the first __getitem__ we
        # don't have to pickle the memmap
        self.token_ids = None
        self._mmap_filename = mmap_filename
        self._chunk_size = chunk_size
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id

    def __len__(self):
        return self.num_instances

    def __getitem__(self, i):
        if self.token_ids is None:
            self.token_ids = np.memmap(self._mmap_filename, mode='r', dtype=np.uint16)
        from_index = i * (self._chunk_size - 2)
        to_index = (i + 1) * (self._chunk_size - 2)
        data = np.concatenate(([self._bos_token_id], self.token_ids[from_index:to_index], [self._eos_token_id]))
        return torch.tensor(data, dtype=torch.long)

class Pretrainer(ptl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.args = hparams
        self.hparams = self.args

        if 'longformer' in args.model and args.model.endswith('/'):  # local path on the disk
            from longformer.longformer import LongformerForMaskedLM, LongformerConfig
            self.config = LongformerConfig.from_pretrained(args.model)
            self.config.attention_mode = args.attention_mode
            self.model = LongformerForMaskedLM.from_pretrained(args.model, config=self.config)
            for i, layer in enumerate(self.model.roberta.encoder.layer):
                layer.attention.self.global_tokens = args.global_tokens
                layer.attention.self.attention_window = args.attention_window
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(args.model)
        if self.args.resize_token_embeddings:
            init_embed_size = self.model.lm_head.decoder.weight.shape[0]
            target_embed_size = init_embed_size + 10  # len(ADDITIONAL_TOKENS) for s2 data
            logger.info(f'increasing model embedding dim from: {init_embed_size} to {target_embed_size}')
            self.model.resize_token_embeddings(init_embed_size)
            logger.info(f'new embed size: {self.model.lm_head.decoder.weight.shape[0]}\n\n\n')
        self.config = self.model.config
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id

        # TODO: add support for other objective functions (whole word masking, BART, Pegasus)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=self.args.mlm_prob
        )
        self.start_time = 0

    def to(self, *args, **kwargs):
        param_count_before_to = len(list(self.parameters()))
        super().to(*args, **kwargs)
        if self.trainer.use_tpu:
            # need to re-tie the weights after moving to XLA!
            self.model.tie_weights()
            if 'roberta' in self.args.model or 'longformer' in self.args.model:
                self.model.lm_head.bias = self.model.lm_head.decoder.bias
        param_count_after_to = len(list(self.parameters()))
        assert param_count_before_to == param_count_after_to

    def forward(self, input_ids=None, labels=None):
        # get the padding mask - 1 for NOT masked, 0 for MASKED/PAD
        attention_mask = (input_ids != self.pad_token_id).int()

        # output is loss, prediction_scores, hidden_states
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output[0]  # loss

    def training_step(self, batch, batch_nb):
        loss = self(**batch)
        input_ids = batch['input_ids']
        tensorboard_logs = {
            'input_size': input_ids.numel(),
            'token_per_step': input_ids.numel() * self.args.grad_accum * self.trainer.world_size,
        }
        if not self.use_tpu:
            # logging additional losses is slow on tpu
            tensorboard_logs['mlm_loss'] = loss
            tensorboard_logs['mlm_bpc'] = loss/math.log(2)
            tensorboard_logs['mlm_perplexity'] = torch.exp(loss)

        if self.start_time != 0:
            elapsed_time = time.time() - self.start_time
            tensorboard_logs['second_per_batch'] = elapsed_time
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs['lr'] = lr
        self.start_time = time.time()
        if self.on_gpu:
            tensorboard_logs['memory'] = torch.cuda.memory_allocated(loss.device) / 1024 ** 3
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
        elif self.use_tpu:
            avg_loss = xm.all_reduce(xm.REDUCE_SUM, avg_loss) / xm.xrt_world_size()

        bpc = avg_loss / math.log(2)
        logs = {'val_mlm_loss': avg_loss, 'val_bpc': bpc}
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
        dataset = MMapTextDataset(fname, chunk_size=self.args.seqlen,
                                  bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id)

        # TODO: consider `replace_sampler_ddp=True` and removing the following if statement
        if self.trainer.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
            shuffle = False
        elif self.trainer.use_tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=is_train,
            )
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
        self.train_dataloader_obj = self._get_loader(f'{self.args.input_dir}/train.bin', True)
        return self.train_dataloader_obj

    def val_dataloader(self):
        return self._get_loader(f'{self.args.input_dir}/val.bin', True)

    def grad_norm(self, norm_type):
        # Override PTL `grad_norm` function to only return `total_grad_norm` instead norms of individual params
        # TODO: grad_norm reporting needs to take fp16 loss scale into account
        parameters = [p for p in self.parameters() if p.grad is not None]
        device = parameters[0].device
        total_norm = torch.zeros([], device=device if parameters else None)
        norm_type = float(norm_type)
        for p in parameters:
            param_norm = p.grad.data.pow(norm_type).sum()
            total_norm.add_(param_norm)
        total_norm = (total_norm ** (1.0 / norm_type))
        return {'total_grad_norm': total_norm}

    def on_epoch_start(self):
        super().on_epoch_start()
        # this works around a bug in PTL that doesn't set epoch on TPU!
        if self.use_tpu and hasattr(self.train_dataloader_obj.sampler, 'set_epoch'):
            self.train_dataloader_obj.sampler.set_epoch(self.current_epoch)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=3)

        # Dataset. Some of these params are only useful when generating the dataset cache
        parser.add_argument("--input_dir", type=str, default='/net/nfs.corp/s2-research/beltagy/longformer/data/')
        # Used only at the preprocessing phase
        parser.add_argument("--seqlen", type=int, default=512)
        parser.add_argument("--mlm_prob", type=float, default=0.15)

        # HF model loading
        parser.add_argument("--tokenizer", type=str, default='roberta-base')
        parser.add_argument("--model", type=str, default='roberta-base')

        # Longformer attention params
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks')
        parser.add_argument("--attention_window", type=int, default=256)
        parser.add_argument("--global_tokens", type=int, default=0)

        # Checkpointing and logging
        parser.add_argument("--save_dir", type=str, default='runs/')
        parser.add_argument("--save_prefix", type=str, default='test',
                            help="path of output directory is --save_dir/--save_prefix")
        parser.add_argument("--resume", type=str, default=None,  # It is better to use a different output dir.
                            help="Path to a checkpoint to load model weights and training state. It overwrites args")
        parser.add_argument("--resume_model_only", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument("--log_rate", type=int, default=10)
        parser.add_argument("--disable_checkpointing", type=bool, default=False)

        # Training hyperparams
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--train_steps", type=int, default=3000, help='# training grad. updates')
        parser.add_argument("--warmup_steps", type=int, default=1000, help='# warmup grad. updates')
        parser.add_argument("--val_every", type=int, default=1000, help='# training grad. updates between evaluations')
        parser.add_argument("--val_batches", type=int, default=1000, help='# evaluation **batches**')
        parser.add_argument("--val_percent_check", type=float, default=1.0, help='percentage of evaluation **batches**')
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_epsilon", type=float, default=1e-6)
        parser.add_argument("--grad_clip", type=float, default=0)  # TODO: test this with fp16. Likely not working

        # RoBERTa's tokens_per_step = 2^18 = 512(seqlen) x 1(gpu_count) x 32(batch_size) x 16(grad_accum)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--grad_accum", type=int, default=1)

        # Compute resources
        parser.add_argument("--fp16", type=bool, default=False)
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
        parser.add_argument("--tpu_core_count", type=int, default=None)
        parser.add_argument("--process_spawn_delay", type=int, default=0)

        parser.add_argument("--resize_token_embeddings", default=False, action='store_true', help='used for s2 data with additional vocabulary.')

        return parser

class MLM_Trainer(ptl.Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_checkpoint(self, filepath):
        def _do_save(chkpt):
            # do the actual save
            try:
                self._atomic_save(chkpt, filepath)
            except AttributeError:
                if 'hparams' in chkpt:
                    del chkpt['hparams']
                self._atomic_save(chkpt, filepath)
        if self.use_tpu and XLA_AVAILABLE:
            # we need to wait for all processes to meet
            xm.rendezvous('checkpoint_dump')
        checkpoint = self.dump_checkpoint()
        # self._atomic_save has different behavior for XLA vs
        # non-xla.  In XLA, it has a barrier and internal logic to only
        # save for rank=0, so need to call for all ranks. For non-XLA,
        # it doesn't have rank=0 logic so only call for rank = 0
        if self.use_tpu and XLA_AVAILABLE:
            _do_save(checkpoint)
        elif self.proc_rank == 0:
            _do_save(checkpoint)
        del checkpoint
        gc.collect()

    def _atomic_save(self, checkpoint, filepath: str):
        """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.
        This will create a temporary checkpoint with a suffix of ``.part``, then copy it to the final location once
        saving is finished.
        Args:
            checkpoint: The object to save.
                Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
                accepts.
            filepath: The path to which the checkpoint will be saved.
                This points to the file that the checkpoint will be stored in.
        """
        tmp_path = str(filepath) + ".part"
        if self.use_tpu and XLA_AVAILABLE:
            device = xm.xla_device()
            ordinal = xm.get_ordinal()
            local_ordinal = xm.get_ordinal()
            xm.rendezvous("saving_model_state")
            xm.save(checkpoint, tmp_path, master_only=True, global_master=True)
            if xm.is_master_ordinal(local=False):
                os.replace(tmp_path, filepath)
        else:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, filepath)



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
        filepath=os.path.join(args.save_dir, args.save_prefix, 'checkpoint', '{epoch}-{val_loss:.4f}'),
        prefix='',
        save_top_k=3,
        # save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        period=-1,  # to allow multiple checkpoints per epoch
    )

    args.val_every *= args.grad_accum  # PTL is expecting number of batches_per_gpu
    trainer = MLM_Trainer(
        gpus=args.gpu_count,
        num_nodes=args.node_count,
        num_tpu_cores=args.tpu_core_count,
        distributed_backend='ddp' if (args.gpu_count > 1 or args.node_count > 1) else None,
        replace_sampler_ddp=False,
        track_grad_norm=2 if args.tpu_core_count is None else -1,  # gradnorm logging is slow on tpus
        max_epochs=10000, min_epochs=0, max_steps=args.train_steps,  # run for many epochs, but stop after max_steps
        val_check_interval=args.val_every, limit_val_batches=args.val_batches,
        early_stop_callback=None,
        row_log_interval=args.log_rate,
        progress_bar_refresh_rate=args.log_rate,
        logger=logger,
        checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else None,
        accumulate_grad_batches=args.grad_accum,
        resume_from_checkpoint=args.resume,
        gradient_clip_val=args.grad_clip,
        precision=16 if args.fp16 else 32, amp_level='O2',
        num_sanity_val_steps=2,
        val_percent_check=args.val_percent_check,
        delay_start_process=args.process_spawn_delay,
        # reload_dataloaders_every_epoch=True
    )
    trainer.fit(pretrainer)


if __name__ == "__main__":
    parser = Pretrainer.add_args(argparse.ArgumentParser(description="pretrain"))
    args = parser.parse_args()
    main(args)
