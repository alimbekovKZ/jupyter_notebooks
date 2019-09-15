import os
import torch

ON_KAGGLE = 'KAGGLE_WORKING_DIR' in os.environ

def load_checkpoint(model, path, return_others=True):
    data = torch.load(path)
    model.load_state_dict(data['model'])
    epoch = data['epoch']
    best_valid_score = data['best_valid_score']
    best_valid_loss = data['best_valid_loss']
    lr = data['lr']
    print("Load model from {}".format(path))
    return epoch, best_valid_score, best_valid_loss, lr

def save_checkpoint(model, path, epoch, best_valid_score, best_valid_loss, lr):
    data = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_valid_score": best_valid_score,
        "best_valid_loss": best_valid_loss,
        "lr": lr
    }
    torch.save(data, path)