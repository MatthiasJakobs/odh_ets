import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from seedpy import fixedseed
from os.path import exists

from models import EncoderDecoder

def main():
    ### Hyperparameters
    epochs = 200
    n_channels = (32, 64)
    lr = 1e-4
    loss = nn.MSELoss()
    batch_size = 128
    device = 'cuda'
    ###

    save_path =  f'results/enc_dec_{n_channels[-1]}.pth'

    if exists(save_path):
        print(f'Model already created. Remove {save_path} to retrain')
        exit()

    with fixedseed(torch, seed=198471):
        model = EncoderDecoder(n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = np.load('data/X_pt_train.npy')
    X_val = np.load('data/X_pt_val.npy')
    y_train = np.load('data/y_pt_train.npy')
    y_val = np.load('data/y_pt_val.npy')

    ds_train = TensorDataset(torch.from_numpy(X_train).float().unsqueeze(1).to(device), torch.from_numpy(y_train).float().to(device))
    ds_val = TensorDataset(torch.from_numpy(X_val).float().unsqueeze(1).to(device), torch.from_numpy(y_val).float().to(device))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
        val_epoch_loss = []

        for _X, _ in dl_train:
            optimizer.zero_grad()
            reconstructed = model(_X)
            L = loss(reconstructed, _X)

            L.backward()
            optimizer.step()

            train_epoch_loss.append(L.detach())

        with torch.no_grad():
            for _X, _ in dl_val:
                model.eval()
                reconstructed = model(_X)

                L = loss(reconstructed, _X)
                val_epoch_loss.append(L)

        train_epoch_loss = torch.stack(train_epoch_loss).mean()
        val_epoch_loss = torch.stack(val_epoch_loss).mean()
        print(f'{epoch} {train_epoch_loss:.5f} | {val_epoch_loss:.5f}')

    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
