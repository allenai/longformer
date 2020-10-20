""" script to read pyspark data from s2 and convert them into json """
import multiprocessing
import json
import os
import gzip
import argparse
import glob
import pathlib
import functools

from multiprocessing.pool import Pool
from tqdm.auto import tqdm

try:
    import pyarrow.parquet as pq
    import pyspark
    import pandas as pd
    from fastparquet import ParquetFile
except ImportError:
    print("in order to process s2 data you will need to install pyarrow and pyspark")
    print("in order to process s2 data you will need to install fastparquet, python-snappy (see python-snappy for dependencies)")


def _process_file(full_fname, args):
    # read the parquet file
    # pdf = ParquetFile(full_fname).to_pandas()
    # spark
    spark = pyspark.sql.SparkSession.builder. \
        config("spark.driver.memory", "10g"). \
        config("spark.executor.memory", "6g"). \
        config("spark.driver.maxResultSize", "0"). \
        config("spark.executor.cores", '1').getOrCreate()
    df = spark.read.load(full_fname)
    pdf = df.toPandas()
    # pdf = pd.read_parquet(full_fname, engine='fastparquet')
    data = []
    total_token_count = 0
    fname, _ = os.path.splitext(full_fname.split('/')[-1])
    outfile = args.output_dir + '/' + fname + '.json.gz'
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with gzip.open(outfile, 'wt') as fout:
        for idx, row in tqdm(pdf.iterrows(), total=pdf.shape[0]):
            token_count = 0
            obj = {
                'title': row.get('title'),
                'abstract': row.get('abstractText'),
                'corpusId': row['corpusId'],
                'sourceId': row.get('sourceId'),
            }
            sections, section_titles = [], []
            for section in row.get('sections') or []:
                sec_dict = section.asDict()
                # filter bad sections
                sections.append(sec_dict['body'])
                section_titles.append(sec_dict['header'])
            if len(sections) == 0 and not obj['abstract']:
                continue
            obj['sections'] = sections
            obj['section_titles'] = section_titles
            fout.write(json.dumps(obj) + '\n')

    return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', help='path to the input files (use wild card to match many files)')
    parser.add_argument('--output_dir', help='path to the output files (use wild card to match many files)')
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel workers')
    args = parser.parse_args()

    all_files = glob.glob(f'{args.input_files}')

    args.input_dir = str(pathlib.Path(args.input_files).parent)

    # STEP1: tokenizing and saving to shards
    process_fn = functools.partial(_process_file, args=args)
    if args.num_workers > 1:
        from multiprocessing.pool import Pool
        with Pool(args.num_workers) as p:
            list(tqdm(p.imap(process_fn, all_files), total=len(all_files)))
    else:
        [process_fn(f) for f in tqdm(all_files)]

if __name__ == '__main__':
    main()
