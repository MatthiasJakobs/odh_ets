import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from seedpy import fixedseed
from tsx.distances import dtw, euclidean
from tsx.datasets import windowing
from scipy.stats import entropy
from copy import deepcopy

from models import EncoderDecoder, MultiForecaster
from datasets.dataloading import load_dataset, get_all_datasets
from run_experiments import hyperparameters
from utils import smape

def main():
    data = get_all_datasets()
    rng = np.random.RandomState(958717)
    random_indices = rng.choice(np.arange(len(data)), size=15, replace=False)
    data = [data[idx] for idx in random_indices]

    runtimes = np.zeros((len(data), 3))

    for row_idx, (ds_name, ds_index) in enumerate(data):
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

        # -----------------
        model.rocs = deepcopy(rocs)
        before = time.time()
        model.predict_weighted(X_test, k=1, dist_fn=euclidean, max_dist=100)
        after = time.time()

        runtimes[row_idx][0] = after-before

        # -----------------
        drift_file = f'results/drifts/{ds_name}_{ds_index}_OEP-ROC-15_type1.npy'
        drifts = np.load(drift_file)
        model.rocs = model.restrict_rocs(deepcopy(rocs), 5)
        before = time.time()
        model.predict_weighted_drifts(X_val, X_test, drifts, k=1, dist_fn=euclidean, max_dist=100)
        after = time.time()

        runtimes[row_idx][1] = after-before

        # -----------------
        # Periodic
        periodicity = len(X_val) // 10
        drifts = np.linspace(0, len(X_val), periodicity, dtype=np.int32)
        model.rocs = deepcopy(rocs)
        before = time.time()
        model.predict_weighted_drifts(X_val, X_test, drifts, k=1, dist_fn=euclidean, max_dist=100)
        after = time.time()

        runtimes[row_idx][2] = after-before

    mean_runtime = runtimes.mean(axis=0)
    std_runtime = runtimes.std(axis=0)

    for idx, method_name in enumerate(['static', 'driftaware', 'periodic']):
        print(f'{method_name}: {mean_runtime[idx]:.2f} +- {std_runtime[idx]:.2f} [seconds]')

    print(f'estimated over {len(data)} datasets')

if __name__ == '__main__':
    main()
