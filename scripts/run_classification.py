""" simple script to run classification with different seeds, lrs, multiple datasets """
import argparse
import json
import subprocess
from collections import defaultdict
import glob
import sys
import signal
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', help='path to the model dir')
    parser.add_argument('--checkpoint-path', help='name of the model')
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--save-prefix', '--save_prefix', default=None, required=True)
    parser.add_argument('--tasks', default=None, help='comma separated name of tasks')
    parser.add_argument('--data-dir', '--data_dir', default=None, help='path to the root data.')
    parser.add_argument('--save-dir', '--save_dir', default=None, help='root path to save results', required=True)
    parser.add_argument('--baseline', default=False, action='store_true')

    args = parser.parse_args()

    data_dirs = glob.glob(args.data_dir + '/*')

    if args.tasks is not None:
        tasks = args.tasks.split(',')
        full_paths = [f'{args.data_dir}/{task}' for task in tasks]
    else:
        tasks = [e.split('/')[-1] for e in data_dirs]
        full_paths = data_dirs
    print(f'running on tasks: {"; ".join(tasks)}')


    for i, (task, data_path) in enumerate(zip(tasks, full_paths)):
        print(f'running on task {i}: {task}')
        for seed in [1234, 21, 65]:
            for lr in [0.00002, 0.00003, 0.00005]:
                for epochs in [2, 3, 5]:
                    for effective_batch_size in [16, 32]:
                        save_prefix = args.save_prefix + f"{task}-seed{seed}-{str(lr)}-ep{epochs}"
                        outdir = args.save_dir + '/' + save_prefix
                        batch = 8
                        accum = effective_batch_size // batch
                        if args.baseline:
                            command = ['python',
                            'scripts/classification.py',
                            '--input_dir',
                            data_path,
                            '--gpus',
                            '1',
                            '--save_dir',
                            outdir,
                            '--num_workers',
                            '16',
                            '--batch_size',
                            str(batch),
                            '--fp16',
                            '--val_check_interval',
                            '0.5',
                            '--grad_accum',
                            str(accum),
                            '--num_epochs',
                            f'{epochs}',
                            '--lr',
                            str(lr),
                            '--do_predict',
                            '--use_roberta']
                        else:
                            command = ['python',
                            'scripts/classification.py',
                            '--input_dir',
                            data_path,
                            '--gpus',
                            '1',
                            '--save_dir',
                            outdir,
                            '--num_workers',
                            '12',
                            '--batch_size',
                            str(batch),
                            '--fp16',
                            '--val_check_interval',
                            '0.5',
                            '--grad_accum',
                            str(accum),
                            '--num_epochs',
                            f'{epochs}',
                            '--config_path',
                            args.config_path,
                            '--checkpoint_path',
                            args.checkpoint_path,
                            '--add_tokens',
                            '--attention_mode',
                            'sliding_chunks3',
                            '--attention_window',
                            '170',
                            '--lr',
                            str(lr),
                            '--do_predict']
                        print(f'running {" ".join(command)}')
                        proc = subprocess.run(command)

if __name__ == '__main__':
    main()