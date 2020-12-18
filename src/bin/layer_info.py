import sys
import os
import argparse
from tabulate import tabulate
import pandas as pd

if './' not in sys.path:
    sys.path.append('./')

from src.bin import get_classifier


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('classifier', type=str)
    parser.add_argument('--index', '-i', type=int, default=-1)
    return parser.parse_args(args)

def main(args):
    args = _parse_args(args)
    layer_info = get_classifier(args.classifier, pretrained=True).layer_info
    df = pd.DataFrame({'layer' : layer_info})

    # Show where your function goes if requested
    if args.index >= 0:
        line = pd.DataFrame({"layer": '*** YOUR FUNCTION ***'}, index=['X'])
        df = pd.concat([df.iloc[:args.index], line, df.iloc[args.index:]])

    print(tabulate(df, headers='keys', tablefmt="fancy_grid"))

if __name__ == '__main__':
    main(sys.argv[1:])