import argparse
import os
import pandas as pd
import torch
import torchvision


PATH_MODEL = 'weights.pt'
BATCH_SIZE = 256


class TestAntispoofDataset(torch.utils.data.dataset.Dataset):
    def __init__(
            self, paths, transform=None,
            loader=torchvision.datasets.folder.default_loader):
        self.paths = paths
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        image_info = self.paths[index]

        img = self.loader(image_info['path'])
        if self.transform is not None:
            img = self.transform(img)

        return image_info['id'], image_info['frame'], img

    def __len__(self):
        return len(self.paths)


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
            'frame': row.frame,
            'path': os.path.join(path_test_dir, row.path)
        } for _, row in test_dataset_paths.iterrows()]

    # prepare dataset and loader
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_dataset = TestAntispoofDataset(
        paths=paths, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
    model = model.to(device)
    model.eval()

    # predict
    samples, frames, probabilities = [], [], []

    with torch.no_grad():
        for video, frame, batch in dataloader:
            batch = batch.to(device)
            probability = torch.sigmoid(model(batch).view(-1))

            samples.extend(video)
            frames.extend(frame.numpy())
            probabilities.extend(probability.cpu().numpy())

    # save
    predictions = pd.DataFrame.from_dict({
        'id': samples,
        'frame': frames,
        'probability': probabilities})

    predictions = predictions.groupby('id').probability.mean().reset_index()
    predictions['prediction'] = predictions.probability
    predictions[['id', 'prediction']].to_csv(
        args.path_submission_csv, index=False)
