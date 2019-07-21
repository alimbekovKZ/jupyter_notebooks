import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import TopModel

from dataset import AntispoofDataset
from validation import validation
import torchvision


def save_model(model_, epoch, save_path, name_postfix=''):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, f"model_{name_postfix}.pt")

    torch.save(
        {
            'model': model_.state_dict(),
            'epoch': epoch
            },
        model_path
    )


def load_model(model, model_path):
    state = torch.load(model_path)
    epoch = state['epoch']

    model.load_state_dict(state['model'])
    print('Restored model, epoch {}'.format(epoch))

    return epoch, model


def train():
    path_data = './data/idrnd_train_data_v1/train'
    checkpoints_path = './checkpoints'

    num_epochs = 1000
    batch_size = 14

    val_batch_size = 10
    lr = 0.001
    weight_decay = 0.0000001
    model = TopModel()

    model.train()
    model = model.cuda()
    epoch = 0
    if os.path.exists(os.path.join(checkpoints_path, 'model_.pt')):
        epoch, model = load_model(model, os.path.join(checkpoints_path, 'model_.pt'))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    path_images = []

    for label in ['2dmask', 'real', 'printed', 'replay']:
        videos = os.listdir(os.path.join(path_data, label))
        for video in videos:
            path_images.append({
                'path': os.path.join(path_data, label, video),
                'label': int(label != 'real'),
                })

    split_on = int(len(path_images) * 0.9)

    train_paths = path_images[:split_on]
    val_paths = path_images[split_on:]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = AntispoofDataset(paths=train_paths, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)

    val_dataset = AntispoofDataset(paths=val_paths, transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=True,
                            num_workers=4,
                            drop_last=False)
    tq = None
    try:
        for epoch in range(epoch, num_epochs):
            tq = tqdm(total=len(train_loader) * batch_size)
            tq.set_description(f'Epoch {epoch}, lr {lr}')

            losses = []

            # iterate over data
            for inputs, labels in train_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1), labels.float())

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    tq.update(batch_size)
                    losses.append(loss.item())

                intermediate_mean_loss = np.mean(losses[-10:])
                tq.set_postfix(loss='{:.5f}'.format(intermediate_mean_loss))

                # statistics

            epoch_loss = np.mean(losses)
            epoch_metrics = validation(model, val_loader=val_loader)
            tq.close()
            print('\nLoss: {:.4f}\t Metrics: {}'.format(epoch_loss, epoch_metrics))
            save_model(model, epoch, checkpoints_path, name_postfix=f'e{epoch}')

    except KeyboardInterrupt:
        tq.close()
        print('\nCtrl+C, saving model...')
        save_model(model, epoch, checkpoints_path)


if __name__ == '__main__':
    train()
