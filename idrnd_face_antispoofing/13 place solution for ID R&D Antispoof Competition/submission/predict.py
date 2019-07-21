import argparse
import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
import cv2
from model import TopModel

PATH_MODEL = './model.pt'
BATCH_SIZE = 30


class AntispoofDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        image_info = self.paths[index]

        imgs = self.load_images(image_info['path'])

        return image_info['id'], imgs

    def __len__(self):
        return len(self.paths)

    def load_images(self, path):
        frames = os.listdir(path)
        imgs = []
        for p in frames:
            img = cv2.imread(os.path.join(path, p))
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-images-csv', type=str, required=True)
    parser.add_argument('--path-test-dir', type=str, required=True)
    parser.add_argument('--path-submission-csv', type=str, required=True)
    args = parser.parse_args()

    # prepare image paths
    test_dataset_paths = pd.read_csv(args.path_images_csv)
    path_test_dir = args.path_test_dir

    paths = [
        {
            'id': row.id,
            'path': os.path.join(path_test_dir, row.id)
        } for _, row in test_dataset_paths.iterrows()]

    # prepare dataset and loader
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_dataset = AntispoofDataset(
        paths=paths, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model

    model = TopModel()

    model.load_state_dict(torch.load(PATH_MODEL)['model'])
    model = model.to(device)
    model.eval()

    # predict
    samples, probabilities = [], []

    with torch.no_grad():
        for video, batch in dataloader:
            batch = batch.to(device)
            probability = torch.sigmoid(model(batch).view(-1))

            samples.extend(video)
            probabilities.extend(probability.cpu().numpy())

    # save
    predictions = pd.DataFrame.from_dict({
        'id': samples,
        'probability': probabilities})

    predictions = predictions.groupby('id').probability.mean().reset_index()
    predictions['prediction'] = predictions.probability
    predictions[['id', 'prediction']].to_csv(
        args.path_submission_csv, index=False)
