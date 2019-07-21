import torch
from fire import Fire
from torch import jit

from models import get_baseline


def main(model_path='model.pt', name='densenet201', out_name='model.trcd'):
    model = torch.load(model_path)
    model, = list(model.children())
    state = model.state_dict()

    base = get_baseline(name=name)
    base.load_state_dict(state)
    base.eval()

    model = jit.trace(base, example_inputs=(torch.rand(4, 3, 256, 256),))
    jit.save(model, out_name)


if __name__ == '__main__':
    Fire(main)
