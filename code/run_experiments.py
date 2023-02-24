###
# Pipeline for model training
###

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from seedpy import fixedseed
from sklearn.metrics import mean_squared_error as mse

from models import EncoderDecoder, MultiForecaster
from deepar import DeepARWrapper

def main():

    # Global parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128

    # Load both pretrain and experiment data
    ds_experiments = np.load('data/exp_data.npy')

    X_train = np.load('data/X_pt_train.npy')
    X_val = np.load('data/X_pt_val.npy')
    y_train = np.load('data/y_pt_train.npy')
    y_val = np.load('data/y_pt_val.npy')

    ds_train = TensorDataset(torch.from_numpy(X_train).float().unsqueeze(1).to(device), torch.from_numpy(y_train).float().to(device))
    ds_val = TensorDataset(torch.from_numpy(X_val).float().unsqueeze(1).to(device), torch.from_numpy(y_val).float().to(device))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # No thresholding
    # euclidean

    # Initialize all models
    hyperparameters = {
        #'encoder_decoder': {
            # 'lr': 1e-4,
            # 'n_epochs': 50,
            # 'n_channels': (64, 32),
            # 'loss': nn.MSELoss(),
            # 'model_init_seed': 198471,
        # },
        # 'e2e_complete': {
        #     'learning_rate': 1e-4,
        #     'lagrange_multiplier': 0.1,
        #     'n_channels': (64, 32),
        #     'n_epochs': 300,
        #     'model_init_seed': 198471,
        # },
        'e2e_no_decoder': {
            'learning_rate': 1e-4,
            'lagrange_multiplier': None,
            'n_channels': (64, 32),
            'n_epochs': 300,
            'model_init_seed': 198471,
        },
        # 'e2e_finetune': {
        #     'learning_rate': 1e-4,
        #     'lagrange_multiplier': None,
        #     'n_epochs': 300,
        #     'model_init_seed': 198471,
        # },
        'deepar': {
            'n_epochs': 25,
            'model_init_seed': 198471,
        },
    }

    # Pretrain encoder / decoder
    print('== encdec ==')
    hyp = hyperparameters['encoder_decoder']

    with fixedseed(torch, seed=hyp['model_init_seed']):
        encdec = EncoderDecoder(hyp['n_channels']).to(device)
    loss = hyp['loss']
    optimizer = torch.optim.Adam(encdec.parameters(), lr=hyp['lr'])

    for epoch in range(hyp['n_epochs']):
        encdec.train()
        train_epoch_loss = []
        val_epoch_loss = []

        for _X, _ in dl_train:
            optimizer.zero_grad()
            reconstructed = encdec(_X)
            L = loss(reconstructed, _X)

            L.backward()
            optimizer.step()

            train_epoch_loss.append(L.detach())

        with torch.no_grad():
            for _X, _ in dl_val:
                encdec.eval()
                reconstructed = encdec(_X)

                L = loss(reconstructed, _X)
                val_epoch_loss.append(L)

        train_epoch_loss = torch.stack(train_epoch_loss).mean()
        val_epoch_loss = torch.stack(val_epoch_loss).mean()
        print(f'{epoch} {train_epoch_loss:.5f} | {val_epoch_loss:.5f}')

    torch.save(encdec.state_dict(), f'results/encdec.pth')

    # Run finetuning / end-to-end and comparison methods, saving trained models

    rmse = lambda a, b: mse(a, b, squared=False)
    subset_indices = [0]
    for idx in subset_indices:
        X = ds_experiments[idx]
        test_size = int(0.25 * len(X))

        mu, std = np.mean(X[:-test_size]), np.std(X[:-test_size])
        X = (X - mu) / std

        X_test = X[-test_size:]

        # e2e-complete
        print('== e2e-complete ==')
        hyp = hyperparameters['e2e_complete']
        with fixedseed(torch, seed=hyp['model_init_seed']):
            _encdec = EncoderDecoder(hyp['n_channels']).to(device)
            e2e_complete = MultiForecaster(hyp, _encdec, hyp['n_channels'][-1], 5).to(device)
            e2e_complete.fit(X, batch_size, train_encoder=True)
        torch.save(e2e_complete.state_dict(), f'results/e2e_complete_{idx}.pth')
        preds = e2e_complete.predict(X, return_mean=True)
        print(rmse(preds, X_test))
        
        # e2e-no-decoder
        print('== e2e-no-decoder ==')
        hyp = hyperparameters['e2e_no_decoder']
        with fixedseed(torch, seed=hyp['model_init_seed']):
            _encdec = EncoderDecoder(hyp['n_channels']).to(device)
            e2e_no_decoder = MultiForecaster(hyp, _encdec, hyp['n_channels'][-1], 5).to(device)
            e2e_no_decoder.fit(X, batch_size, train_encoder=True)
        torch.save(e2e_no_decoder.state_dict(), f'results/e2e_no_decoder_{idx}.pth')
        preds = e2e_no_decoder.predict(X, return_mean=True)
        print(rmse(preds, X_test))

        # e2e-finetune
        print('== e2e-finetune ==')
        hyp = hyperparameters['e2e_finetune']
        with fixedseed(torch, seed=hyp['model_init_seed']):
            e2e_finetune = MultiForecaster(hyp, encdec, hyperparameters['encoder_decoder']['n_channels'][-1], 5).to(device)
            e2e_finetune.fit(X, batch_size, train_encoder=False)
        torch.save(e2e_finetune.state_dict(), f'results/e2e_finetune_{idx}.pth')
        preds = e2e_finetune.predict(X, return_mean=True)
        print(rmse(preds, X_test))

        # deepar baseline
        print('== deepar ==')
        hyp = hyperparameters['deepar']
        with fixedseed(np, seed=hyp['model_init_seed']):
            deepar = DeepARWrapper(n_epochs=hyp['n_epochs'], lag=5)
            deepar.fit(X)
        # TODO: Save
        preds = deepar.predict(X)
        print(rmse(preds, X_test))
    

if __name__ == '__main__':
    main()
