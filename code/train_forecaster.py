import numpy as np
import torch
import matplotlib.pyplot as plt
from seedpy import fixedseed
from os.path import exists
from tsx.datasets.utils import windowing
from sklearn.metrics import mean_squared_error as mse

from models import MultiForecaster, EncoderDecoder

def main():
    ### Hyperparameters
    hyperparameters = {
        'n_epochs': 500,
        'learning_rate': 1e-4,
        'lagrange_multiplier': None,
    }
    n_channels_enc_dec = (64, 32)
    device = 'cuda'
    report_every = 10
    verbose = True
    ###

    enc_path =  f'results/enc_dec_64.pth'
    save_path =  f'results/forecaster_test.pth'

    if exists(save_path):
        print(f'Model already created. Remove {save_path} to retrain')
        exit()

    # Load experiment data
    ds_experiments = np.load('data/exp_data.npy')
    # Subset of experiments to run
    subset_indices = [0]

    for idx in subset_indices:
        # Load encoder/decoder

        with fixedseed(torch, seed=198471):
            enc_dec = EncoderDecoder(n_channels_enc_dec)
            #enc_dec.load_state_dict(torch.load(enc_path))
            model = MultiForecaster(hyperparameters, enc_dec, n_channels_enc_dec[-1], 5).to(device)

        X = ds_experiments[idx]
        test_size = int(0.25 * len(X))

        mu, std = np.mean(X[:-test_size]), np.std(X[:-test_size])
        X = (X - mu) / std

        X_test = X[-test_size:]

        log = model.fit(X, report_every=report_every, verbose=verbose)

        # Test
        rmse = lambda a, b: mse(a, b, squared=False)
        all_preds = model.predict(X, return_mean=False)
        preds = all_preds.mean(axis=-1)
        print(rmse(preds, X_test))

        # Find out how the forecaster performances are distributed
        model_was_best = np.zeros((len(model.forecasters)))
        best_prediction = np.zeros((175))
        for idx, ind_pred in enumerate(all_preds[5:]):
            losses = (ind_pred - X_test[idx])**2
            model_was_best[np.argmin(losses)] += 1
            best_prediction[idx] = losses.min()

        best_prediction[:5] = X_test[:5]

        print('fcn1 fcn2 cnn1 cnn2 sdt1 sdt2 linear')
        print(model_was_best.astype(np.uint8))
        print('best possible selection loss', rmse(X_test, best_prediction))

        forecaster_names = ['fcn1', 'fcn2', 'cnn1', 'cnn2', 'sdt1', 'sdt2', 'linear']
        for (name, pred) in zip(forecaster_names, all_preds.T):
            print(name, rmse(pred, X_test))

        plt.figure()
        plt.plot(X, color='black', label='X')
        plt.plot(np.arange(len(X_test)) + len(X) - test_size, preds, color='red', label='End-to-end (single loss)')
        plt.axvline(x=len(X)-test_size, color='red')
        plt.legend()
        plt.savefig('plots/e2e_test.png')

        plt.figure()
        plt.plot(log[:, 0], log[:, 1], color='red', label='train_loss')
        plt.plot(log[:, 0], log[:, 2], color='green', label='val_loss')
        plt.legend()
        plt.savefig('plots/e2e_train.png')

        #torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
