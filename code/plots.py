import numpy as np
import torch
import matplotlib.pyplot as plt

from seedpy import fixedseed
from tsx.distances import dtw, euclidean
from tsx.datasets import windowing
from scipy.stats import entropy

from models import EncoderDecoder, MultiForecaster
from datasets.dataloading import load_dataset
from run_experiments import hyperparameters
from utils import smape

def main():
    ds_name = 'electricity_hourly'
    ds_index = 10

    X = load_dataset(ds_name, ds_index)

    test_size = int(0.25 * len(X))
    train_size = int(0.5 * len(X))
    lag = 5

    X_train = X[:train_size]
    X_test = X[-test_size:][:-1]
    X_val = X[train_size:-test_size]
        
    savepath = f'models/{ds_name}_#{ds_index}_e2e.pth'

    device = 'cuda'
    hyp = hyperparameters['e2e_no_decoder']
    with fixedseed(torch, seed=hyp['model_init_seed']):
        _encdec = EncoderDecoder(hyp['n_channels']).to(device)
        model = MultiForecaster(hyp, _encdec, hyp['n_channels'][-1], lag).to(device)
    model.load_state_dict(torch.load(savepath))
    model.eval()
    model.to('cpu')


    rocs = model.build_rocs(X_val, only_best=False)
    model.rocs = rocs
    roc_distribution = np.array([len(roc) for roc in rocs])
    print('roc_dist', roc_distribution)
    weights, _ = model.predict_weighted(X_test, k=1, dist_fn=euclidean, max_dist=100)

    min_entropy = 1000
    min_entropy_idx = 0
    max_entropy = 0
    max_entropy_idx = 0

    for idx, w in enumerate(weights):
        nr_close_to_zero = (w <= 0.01).sum()
        print(idx, nr_close_to_zero)

        e = entropy(w, base=len(w))
        if e < min_entropy:
            min_entropy = e
            min_entropy_idx = idx

        if e > max_entropy:
            max_entropy = e
            max_entropy_idx = idx

    # many small: 95
    # some small: 24

    X_test_win, _ = windowing(X_test, lag=5)

    # ------------------
    idx = 29
    w = weights[idx]
    smallest_weight_idx = np.argmin(w)
    largest_weight_idx = np.argmax(w)

    x = X_test_win[idx]
    closest_roc_small = model.rocs[smallest_weight_idx][np.argmin([euclidean(r, x) for r in model.rocs[smallest_weight_idx] if r.shape[0] != 0])]
    closest_roc_large = model.rocs[largest_weight_idx][np.argmin([euclidean(r, x) for r in model.rocs[largest_weight_idx] if r.shape[0] != 0])]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.bar(np.arange(len(w)), w)
    ax1.bar(smallest_weight_idx,  w[smallest_weight_idx] , color='red')
    ax1.bar(largest_weight_idx, w[largest_weight_idx], color='green')
    ax1.set_xlabel('forecaster')
    ax1.set_ylabel('weight')

    ax2.plot(x, color='black', label='input')
    ax2.plot(closest_roc_large, color='green', label='largest weight')
    ax2.plot(closest_roc_small, color='red', label='smallest weight')
    ax2.set_xlabel('$t$')

    f.suptitle(f'weighted ensemble on {ds_name}_#{ds_index}')

    f.legend()
    f.savefig('plots/w_some_small.png')

    # ------------------

    idx = 78
    w = weights[idx]
    smallest_weight_idx = np.argmin(w)
    largest_weight_idx = np.argmax(w)

    x = X_test_win[idx]
    closest_roc_small = model.rocs[smallest_weight_idx][np.argmin([euclidean(r, x) for r in model.rocs[smallest_weight_idx] if r.shape[0] != 0])]
    closest_roc_large = model.rocs[largest_weight_idx][np.argmin([euclidean(r, x) for r in model.rocs[largest_weight_idx] if r.shape[0] != 0])]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.bar(np.arange(len(w)), w)
    ax1.bar(smallest_weight_idx,  w[smallest_weight_idx] , color='red')
    ax1.bar(largest_weight_idx, w[largest_weight_idx], color='green')
    ax1.set_xlabel('forecaster')
    ax1.set_ylabel('weight')

    ax2.plot(x, color='black', label='input')
    ax2.plot(closest_roc_large, color='green', label='largest weight')
    ax2.plot(closest_roc_small, color='red', label='smallest weight')
    ax2.set_xlabel('$t$')

    f.suptitle(f'weighted ensemble on {ds_name}_#{ds_index}')

    f.legend()
    f.savefig('plots/w_many_small.png')

    # ------------------

    # plt.figure()
    # plt.bar(np.arange(len(weights[0])), weights[min_entropy_idx])
    # plt.title('min entropy')
    # plt.savefig('plots/w_min_entropy.png')

    # ------------------

    # plt.figure()
    # plt.bar(np.arange(len(weights[0])), weights[max_entropy_idx])
    # plt.title('max entropy')
    # plt.savefig('plots/w_max_entropy.png')

if __name__ == '__main__':
    main()


