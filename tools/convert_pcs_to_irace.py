from irace_helpers.compatibility.config_space import convert_from_config_space
from ConfigSpace.read_and_write import pcs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('pcs_file', type=str, help='Path to the PCS file.')
parser.add_argument('irace_file', type=str, help='Path to the IRACE file.')
parser.add_argument('--digit', type=int, help='Number of digits to round to.', default=4)
args = parser.parse_args()

with open(args.pcs_file, 'r') as f:
    pcs = pcs.read(f)

with open(args.irace_file, 'w') as f:
    f.write(convert_from_config_space(pcs, digit=args.digit).as_string() + '\n')

    