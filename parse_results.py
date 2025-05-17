import os
import re
import glob
import json
import pandas as pd


def summary(result_folder, output_path):
    results = []
    for file in glob.glob(f'{result_folder}/*.json'):
        print(file)
        file_bases = os.path.basename(os.path.splitext(file)[0]).split('--')
        with open(file, 'r') as f:
            records = json.load(f)
        for base in file_bases:
            key, value = base.split('=')
            records[key] = value
        results.append(records)
    pd.DataFrame(results).sort_values(
        by=['data']
    ).fillna(
        'OVR'
    ).to_csv(
        output_path, index=False
    )
    return 'Success'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="para_results", help="Grid results folder")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.dir, 'para_results_summary.csv')

    status = summary(result_folder=args.dir, output_path=args.output)
    print(status)
