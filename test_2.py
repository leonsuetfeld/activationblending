
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-l', type=float, nargs='*')
parser.add_argument('-m', type=float, nargs='*')
parser.add_argument('-n', type=str)
parser.add_argument('-b', type=str2bool, nargs=1)
parser.add_argument('-c', type=str2bool, nargs=1)
parser.add_argument('-i', type=int, nargs=1)
parser.add_argument('-f', type=float, nargs=1)
args = vars(parser.parse_args())
print(args)
