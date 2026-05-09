import os
import argparse
from utils_package.quick_start import quick_start
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='PA2PD', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')
    config_dict = {'gpu_id': 0,}
    args, _ = parser.parse_known_args()
    for i in range(1):
            quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, item_num=i, save_model=True)


