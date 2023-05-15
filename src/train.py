import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from model import CNN3d
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import copy
import argparse
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN3dDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = np.load(os.path.join(root, 'grasping.npz'))
        self.x = self.data['x']
        shape = np.shape(self.x)
        print(shape)
        self.y = self.data['y']
        print(sum(self.y)/len(self.y))
        self.x = self.x.reshape((shape[0], 1, 40, 40, 40))
        self.y = self.y.reshape((shape[0], 1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_split(dataset, validation_split, shuffle_dataset=True):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    print('%d data for training, %d data for validation.' % (dataset_size - split, split))
    if shuffle_dataset:
        # np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def train(args):
    # Load the data
    dataset = CNN3dDataset(args.data_dir)
    train_split, val_split = random_split(dataset, validation_split=0.1)
    dataset_splits = {'train': train_split, 'val': val_split}
    dataloaders = {x: DataLoader(dataset, batch_size=32, sampler=dataset_splits[x]) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(dataset_splits[x]) for x in ['train', 'val']}
    model = CNN3d().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    best_loss = np.inf

    # Training loop
    for epoch in range(args.nb_epoch):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_preds = 0
            running_corrects = 0

            # Iterate over data.
            bar = tqdm(dataloaders[phase])
            for start, goal in bar:
                bar.set_description(f'Epoch {epoch + 1} {phase}'.ljust(20))
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(start.to(device, dtype=torch.float))
                    loss = criterion(outputs, goal.to(device, dtype=torch.float))
                    corrects = torch.sum(torch.eq(outputs.detach().cpu().round(), goal))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_corrects += corrects
                running_loss += loss.item() * start.size(0)
                running_preds += start.size(0)
                bar.set_postfix(loss=f'{running_loss / running_preds:0.6f}',
                                acc=f'{running_corrects / running_preds:0.6f}')
            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'weights/gsp.pt')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/lou00015/ReachabilityAwareGrasping/data",
                        help='directory to where grasping data is saved')
    parser.add_argument('--nb_epoch', dest='nb_epoch', type=int, default=100,
                        help='number of data that will be processed')
    # Process data with specified arguments
    args = parser.parse_args()
    train(args)
