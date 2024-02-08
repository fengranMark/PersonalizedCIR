import csv
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

INPUT_FILE = "collection.tsv"
OUTPUT_FILE = "collection.jsonl"

def main(input_file, output_file):
    with open(output_file, 'w') as output:
        for line in tqdm(open(input_file, "r")):
            try:
                pid, passage = line.strip().split('\t')
                obj = {"id": str(pid), "contents": passage}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except:
                continue

if __name__ == "__main__":
    main(INPUT_FILE, OUTPUT_FILE)
