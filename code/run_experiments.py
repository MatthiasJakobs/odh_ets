###
# Pipeline for model training
###

import numpy as np
import torch
import torch.nn as nn

from seedpy import fixedseed
from sklearn.metrics import mean_squared_error as mse
from os.path import exists

from models import EncoderDecoder, MultiForecaster
from deepar import DeepARWrapper
from datasets.dataloading import implemented_datasets, load_dataset

hyperparameters = {
    'e2e_no_decoder': {
        'learning_rate': 1e-3,
        'lagrange_multiplier': None,
        'n_channels': (64, 32),
        'n_epochs': 3000,
        'model_init_seed': 198471,
    },
    'deepar': {
        'n_epochs': 50,
        'model_init_seed': 198471,
    },
}

def main():
    # TODO: Notes
    # No thresholding
    # euclidean

    # Global parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    rmse = lambda a, b: mse(a, b, squared=False)
    lag = 5

    for ds_name, ds_index in implemented_datasets:
        print(ds_name, ds_index)
        X = load_dataset(ds_name, ds_index)

        #train_cutoff = int(len(X) * 0.5)
        #val_cutoff = train_cutoff + int(len(X) * 0.25)
        #X_train, X_val, X_test = X[0:train_cutoff], X[train_cutoff:val_cutoff], X[val_cutoff:]

        test_size = int(0.25 * len(X))

        # TODO: Normalizing? 
        # mu, std = np.mean(X[:-test_size]), np.std(X[:-test_size])
        # X = (X - mu) / std

        X_test = X[-test_size:]

        # e2e-no-decoder
        savepath = f'results/{ds_name}_#{ds_index}_e2e.pth'
        if not exists(savepath):
            hyp = hyperparameters['e2e_no_decoder']
            with fixedseed(torch, seed=hyp['model_init_seed']):
                _encdec = EncoderDecoder(hyp['n_channels']).to(device)
                e2e_no_decoder = MultiForecaster(hyp, _encdec, hyp['n_channels'][-1], lag).to(device)
                e2e_no_decoder.fit(X, batch_size, train_encoder=True, verbose=False)
            torch.save(e2e_no_decoder.state_dict(), savepath)
            preds = e2e_no_decoder.predict(X, return_mean=True)
            print(rmse(preds, X_test))

        # deepar baseline
        savepath = f'results/{ds_name}_#{ds_index}_deepar'
        if not exists(savepath):
            hyp = hyperparameters['deepar']
            with fixedseed(np, seed=hyp['model_init_seed']):
                deepar = DeepARWrapper(n_epochs=hyp['n_epochs'], lag=lag)
                deepar.fit(X)
                deepar.save(savepath)
                preds = deepar.predict(X)
                print(rmse(preds, X_test))
    

if __name__ == '__main__':
    main()
