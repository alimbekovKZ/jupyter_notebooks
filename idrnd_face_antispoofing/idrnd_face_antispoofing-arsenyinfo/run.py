import subprocess

from fire import Fire
from glog import logger


def main(preload=False, parallel=True):
    models = ('densenet121',)
    dropouts = (25, 50)
    folds = (0, 1, 2)

    for d in dropouts:
        for name in models:
            for fold in folds:
                model = f'{name}.{d}'
                command = f"python train.py --name {model} --model {model} --batch_size 128 --n_fold {fold}"
                if preload:
                    command += " --train.preload --val.preload"
                if parallel:
                    command += " --parallel"
                logger.info(command)
                subprocess.call(command.split(' '))


if __name__ == '__main__':
    Fire(main)
