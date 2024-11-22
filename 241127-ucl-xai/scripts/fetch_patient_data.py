import argparse
import os
import pandas as pd


def fetch_patient_data(source: str, destination: str):
    with open(source, 'r') as f:
        lines = f.readlines()

    starts = {
        'ADHD-HC': 3776 - 1,
        'ADHD': 3888 - 1,
        'ANX-HC': 4500 - 1,
        'ANX': 4612 - 1,
        'ASD-HC': 5224 - 1,
        'ASD': 5336 - 1,
        'BIP-HC': 5948 - 1,
        'BIP': 6060 - 1,
        'DEM-HC': 6672 - 1,
        'DEM': 6784 - 1,
        'MCI-HC': 7396 - 1,
        'MCI': 7508 - 1,
        'MDD-HC': 8120 - 1,
        'MDD': 8232 - 1,
        'MS-HC': 8844 - 1,
        'MS': 8956 - 1,
        'SCZ-HC': 9568 - 1,
        'SCZ': 9680 - 1,
    }

    prev_xs = None
    data = {}

    for key, start in starts.items():
        tokens = lines[start:start+100]
        tokens = [line.strip() for line in tokens]
        tokens = [token[1:-1] for token in tokens]
        tokens = [token.split(', ') for token in tokens]
        xs = [float(token[0]) for token in tokens]
        ys = [float(token[1]) for token in tokens]

        if prev_xs is not None and prev_xs != xs:
            raise ValueError()

        prev_xs = xs
        data[key] = ys

    data['x'] = prev_xs
    data = pd.DataFrame(data)
    data.to_csv(destination, index=False)

    print(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fetches patient data from the flux '
                                     'latex presentation')

    parser.add_argument('-s', '--source', required=False,
                        default=os.path.join(os.pardir, '230907-flux',
                                             'presentation.tex'),
                        help='The source file to parse')
    parser.add_argument('-d', '--destination', required=False,
                        default=os.path.join('data', 'disorders.csv'),
                        help='CSV where results are written')

    args = parser.parse_args()

    fetch_patient_data(args.source, args.destination)
