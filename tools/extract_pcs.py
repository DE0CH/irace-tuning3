from ConfigSpace.read_and_write import pcs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('pcs_file', type=str, help='Path to the PCS file.')
args = parser.parse_args()

with open(args.pcs_file, 'r') as f:
    pcs = pcs.read(f)


def convert_dict_to_args(d):
    return ' '.join([f'--{k} {v}' for k, v in d.items()])

print(convert_dict_to_args(pcs.sample_configuration().get_dictionary()))
