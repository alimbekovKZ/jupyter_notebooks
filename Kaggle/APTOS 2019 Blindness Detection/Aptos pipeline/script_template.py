import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}

for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)

run('python setup.py develop --install-dir /kaggle/working')
#run('python -m blindness.make_folds --with_size')
#run('python -m blindness.main train')
#run('python -m blindness.main valid --config_path blindness/configs/base_valid_large.json')
#run('python -m blindness.main valid --config_path blindness/configs/base_small_large.json')
run('python -m blindness.main predict --config_path ../input/resnet50regression/config.json --model_path ../input/resnet50regression/best_model.pt')
#run('python -m blindness.main submit --predictions prediction.pt')
