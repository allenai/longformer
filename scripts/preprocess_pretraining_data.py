import argparse
import functools
import glob
import logging
import math
import multiprocessing
import os
import pathlib
import random
import time
from builtins import NotImplementedError

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # suppress tokenization length errors


try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import tensorflow as tf
except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True


tokenizer = None  # will be loaded later

# ========================= preprocessing code ========================= #
def _process_file(full_fname, args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=args.seq_len)  # use_fast should be false for multiprocessing
    "Step 1: tokenize an input text file then save token ids into `np.memmap` shards of size `args.shard_size`"
    fname = full_fname.split('/')[-1]
    if args.num_workers > 1:
        current = multiprocessing.current_process()
        process_identity = int(current._identity[0])
    else:
        process_identity = 1

    logging.info(f'Processing {full_fname} ...')
    if args.data_type == 'tfrecord':
        fin = tf.data.TFRecordDataset(full_fname)
        log_filename = f'{args.output_dir}/logs-{fname}.log'
    elif args.data_type == 'raw_text':
        fin = open(full_fname, 'r')
        log_filename = f'{args.output_dir}/logs-{args.shard_size}/{fname}.log'
    else:
        raise NotImplementedError

    if os.path.isfile(log_filename):
        logging.info(f'Skipping {full_fname} ...')
        return  # log file already exists. Skip current file.
    # with open(full_fname, 'r') as fin:
    token_list = []
    shard_count = 0
    tokens_count = 0

    def _write_shard():
        if len(token_list) == 0:
            return
        if token_list[-1] != tokenizer.sep_token_id:  # handle a rare case
            token_list.append(tokenizer.sep_token_id)
        if args.data_type == 'tfrecord':
            shared_filename = f'{args.output_dir}/{fname}.bin'
        elif args.data_type == 'raw_text':
            shared_filename = f'{args.output_dir}/shards-{args.shard_size}/{fname}-{shard_count}.bin'
        else:
            raise NotImplementedError
        logging.info(f'Writing {len(token_list)} tokens to shared {shared_filename}')
        fp = np.memmap(shared_filename, dtype=np.uint16, mode='w+', shape=len(token_list))
        fp[:] = token_list[:]
        del fp  # flush and close file

    if args.data_type == 'raw_text':  # the input file is one doc per line
        for line in tqdm(fin):
            line = line.strip()
            if line == '':  # drop empty lines
                continue
            tokens = tokenizer.encode(line, add_special_tokens=False)  # `__getitem__` adds special tokens
            if args.add_sep_after_doc:
                tokens.append(tokenizer.sep_token_id)
            token_list.extend(tokens)
            if len(token_list) > args.shard_size:
                _write_shard()
                tokens_count += len(token_list)
                token_list = []
                shard_count += 1
            else:
                token_list.append(tokenizer.sep_token_id)
        tokens_count += len(token_list)
        _write_shard()
        fin.close()
    elif args.data_type == 'tfrecord':  # the input file is tfrecord format of the c4 dataset
        for raw_example in tqdm(iter(fin), disable=process_identity != 1):
            parsed = tf.train.Example.FromString(raw_example.numpy())
            feature_keys = set(parsed.features.feature.keys())
            if 'text' in feature_keys:
                line = parsed.features.feature['text'].bytes_list.value[0].decode()  # raw text 
                tokens = tokenizer.encode(line, add_special_tokens=False)  # `__getitem__` adds special tokens
                if args.add_sep_after_doc:
                    tokens.append(tokenizer.sep_token_id)
                token_list.extend(tokens)
                tokens_count += len(token_list)
            shard_count += 1
        _write_shard()
    else:
        raise NotImplementedError

    with open(log_filename, 'w') as f:
        f.write(f'Generated {tokens_count} tokens in {shard_count + 1} shards')


def _combine_shards(output_fname, shards_list):
    "Step 2: combining memmap shards into one `train.bin` or `val.bin` file"
    total_size = 0
    for filename in shards_list:
        total_size += np.memmap(filename, mode='r', dtype=np.uint16).shape[0]
    logging.info(f'Writing {total_size} tokens to {output_fname}')
    all_token_ids = np.empty(total_size, dtype=np.uint16)
    last_token_index = 0
    for filename in tqdm(shards_list):
        shared = np.memmap(filename, mode='r', dtype=np.uint16)
        all_token_ids[last_token_index:last_token_index+len(shared)] = shared[:]
        last_token_index += len(shared)
    fp = np.memmap(output_fname, dtype=np.uint16, mode='w+', shape=total_size)
    fp[:] = all_token_ids[:]
    del fp

def raw_text_to_mmap(args):
    """This is the main preprocessing function. It processes all the text files in `args.input_dir` and
    outputs two np.memmap files, one for training and one for validation with ratio `args.train_dev_split`.
    Processing each input file involves tokenizing it, sharding it into shards of size `args.shard_size`,
    then writing each shard as an np.memmap file, shuffle the shards, split them into train and dev shards,
    then combine the shards of each set into one big file (train.bin and val.bin).
    Notice that only the shards are shuffled not the instances inside each shard. Therefor, it is important
    to use `args.shard_size` that's small enough to have a good train/dev split, but also not small enough
    to end up with a huge number of shards that might be difficult to work with.
    The stream of tokens in the memmap files represents documents separated with `tokenizer.sep_token`.
    In `__getitem__`, the `tokenizer.bos_token` and `tokenizer.eos_token`
    are added. The reason for not adding them at preprocessing time is to allow different sequence lengths
    later on. Notice that this is the "FULL-SENTENCES" setting in the RoBERTa paper, Table2.
    Example running the preprocessing:
        >>> python scripts/pretrain.py --input_dir dirWithTextFiles --train_dev_split 0.05  \
                                        --shard_size  268435456  --num_workers 16
    """
    # global tokenizer
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # if tokenizer is None:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # assert len(tokenizer) < 65535  # will use uint16 to store token ids

    all_files = glob.glob(f'{args.input_files}')

    args.input_dir = str(pathlib.Path(args.input_files).parent)

    if not args.data_type == 'raw_text':
        try:
            os.mkdir(f'{args.output_dir}/shards-{args.shard_size}/')
        except FileExistsError:
            pass
        try:
            os.mkdir(f'{args.output_dir}/logs-{args.shard_size}/')  # log progrss to be able to resume
        except FileExistsError:
            pass

    # STEP1: tokenizing and saving to shards
    process_fn = functools.partial(_process_file, args=args)
    if args.num_workers > 1:
        from multiprocessing.pool import Pool
        with Pool(args.num_workers) as p:
            list(tqdm(p.imap(process_fn, all_files), total=len(all_files)))
    else:
        [process_fn(f) for f in tqdm(all_files)]

    if args.data_type == 'raw_text':  # c4 tfrecords are already sharded
        # STEP2: shuffling shards and combining them into train.bin and val.bin files
        all_shards = glob.glob(f'{args.output_dir}/shards-{args.shard_size}/*.bin')
        random.shuffle(all_shards)  # shuffling based on shards not individual lines
        val_shards_count = int(args.train_dev_split * len(all_shards))
        val_shards = all_shards[:val_shards_count]
        train_shards = all_shards[val_shards_count:]
        # TODO: if MMapTextDataset._combining_shards is very slow for large files, it can be skipped but we nned to
        # update the dataset to read from multiple shards directly
        _combine_shards(f'{args.output_dir}/val.bin', val_shards)
        _combine_shards(f'{args.output_dir}/train.bin', train_shards)
    elif args.data_type == 'tfrecord':
        train_shards = glob.glob(f'{args.output_dir}/*train*.bin')
        val_shards = glob.glob(f'{args.output_dir}/*val*.bin')
        _combine_shards(f'{args.output_dir}/val.bin', val_shards)
        _combine_shards(f'{args.output_dir}/train.bin', train_shards)

# ========================= end preprocessing code ========================= #

def add_args(parser):
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input_files", type=str, help='regex like path for matching input files')
    parser.add_argument("--output_dir", type=str, help='regex like path for matching input files')
    # Used only at the preprocessing phase
    parser.add_argument("--train_dev_split", type=float, default=0.05)
    parser.add_argument("--shard_size", type=int, default=1024 ** 3 // 4)  # 250MB
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    # Used only at the training phase

    # HF model loading
    parser.add_argument("--tokenizer", type=str, default='roberta-base')
    parser.add_argument("--model", type=str, default='roberta-base')
    parser.add_argument("--tfrecord", action='store_true', default=False, help='the input files are tfrecords (for c4 dataset)')
    parser.add_argument("--add_sep_after_doc", action='store_true', default=False, help='add sep token after document')

    parser.add_argument("--data_type", default='raw_text', choices=['raw_text', 's2', 's2orc', 'tfrecord'])

    return parser


def main(args):
    random.seed(args.seed * 10)
    np.random.seed(args.seed * 100)
    torch.manual_seed(args.seed * 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed * 10000)
    raw_text_to_mmap(args)


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description="pretrain"))
    args = parser.parse_args()
    main(args)
