""" mesh/mag files are stored in gzip format with structured input
this simple script converts them into a flat format """
import gzip
import os
import json
import glob
import pathlib
import argparse

from tqdm.auto import tqdm

def get_text(record):
    title = record.get('title') or ''
    text = title
    abstract = record.get('abstract') or ''
    text += (' ' + abstract) if abstract else ''
    if 'section_titles' in record:
        for sec_title, sec in zip(record['section_titles'], record['sections']):
            sec_title = (' ' + sec_title) if sec_title else ''
            sec = ' ' + sec if sec else ''
            text += sec_title + sec
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='')
    parser.add_argument('output_dir', help='')
    args = parser.parse_args()

    files = glob.glob(args.input_dir + '/*.gz')
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for f in files:
        fname = f.split('/')[-1]
        outfile = os.path.join(args.output_dir, fname)
        with gzip.open(outfile, 'wt') as fout:
            with gzip.open(f, 'rt') as fin:
                for line in tqdm(fin, desc=f"processing {fname}"):
                    record = json.loads(line)
                    obj = {
                        'text': get_text(record),
                        'label': record['label'],
                        'label_text': record['label_text']
                    }
                    fout.write(json.dumps(obj) + '\n')

if __name__ == '__main__':
    main()
