import argparse
import os
import numpy as np
import pandas as pd


def compute_means(source: str):
    source = pd.read_csv(source, index_col=0)
    for column in source:
        if column == 'fold':
            continue

        print(f'{column}: {np.mean(source[column]):.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computes clinical mean AUCs')

    parser.add_argument('-s', '--source', required=False,
                        default=os.path.join('data', 'casecontrol.csv'),
                        help='Path to case-control CSV file')

    args = parser.parse_args()

    compute_means(args.source)
