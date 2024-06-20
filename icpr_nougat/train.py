import os
import sys

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_PATH)
os.environ['RUN_ON_GPU_IDs'] = "0,1,2,3"

import argparse
import models
# import setproctitle
from utils.utils import load_config
# from experiment import get_experiment_name



def init_args():
    parser = argparse.ArgumentParser(description='trainer args')
    parser.add_argument(
        '--config_file',
        default='config.yaml',
        type=str,
    )
    parser.add_argument(
        '--experiment_name',
        default='Donut',
        type=str,
    )
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
    )
    args = parser.parse_args()
    os.environ['WORKSPACE'] = args.experiment_name
    return args

def get_experiment_name(name):
    name_split = name.split("_")
    trainer_name = "".join([tmp_name[0].upper() + tmp_name[1:] for tmp_name in name_split])
    return "{}Experiment".format(trainer_name)


def main(args):
    config = load_config(args.config_file, args.experiment_name)
    config.update({'phase': args.phase})
    print(config)
    experiment_instance = getattr(models, get_experiment_name(args.experiment_name))(config)
    if args.phase == 'train':
        experiment_instance.train()
    elif args.phase == 'evaluate':
        experiment_instance.evaluate()


if __name__ == '__main__':
    args = init_args()
    main(args)
    
    
# python -m transformers.onnx --model ./cust-data/weights --feature=vision2seq-lm hand-write-onnx --atol 1e-4