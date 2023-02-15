import numpy as np
import pickle

from tsx.datasets.monash import load_monash, possible_datasets
from tsx.datasets.utils import windowing
from tsx.utils import to_random_state
from os.path import exists
from sklearn.model_selection import train_test_split

# Split all datasets into distinct sets
# One for pretraining and one for finetuning (experiments)
def split_monash_700(random_state=10):
    with open('data/monash_700.pickle', 'rb') as _f:
        monash_700 = pickle.load(_f)

    rng = np.random.RandomState(random_state)
    ds_names = list(monash_700.keys())

    ds1 = []
    ds2 = []
    
    # How many time series to pick from each dataset at maximum
    max_size = 10

    for ds_name in ds_names:
        coin = rng.binomial(1, 0.7)
        sample_size = min(monash_700[ds_name].shape[0], max_size)
        X = monash_700[ds_name]
        X_sample = X[rng.choice(len(X), size=sample_size)]
        if coin:
            ds1.append(X_sample)
        else:
            ds2.append(X_sample)

    ds1 = np.concatenate(ds1)
    ds2 = np.concatenate(ds2)
    return ds1, ds2

# Find out how many TS result if we set the desired length to `L`
def evaluate_ts_lengths(L):
    valid_datasets = []
    for ds_name in possible_datasets():
        if '_missing' not in ds_name:
            valid_datasets.append(ds_name)
    n_valid_ts = {}
    for ds_name in valid_datasets:
        data = load_monash(ds_name)['series_value']
        lengths = np.array([len(X) for X in data])
        n_valid_ts[ds_name] = (lengths >= L).sum()

    return n_valid_ts

def generate_subsets(L, out_path, random_state=None):
    rng = to_random_state(random_state)

    # Do not use datasets with missing values
    valid_datasets = []
    for ds_name in possible_datasets():
        if '_missing' not in ds_name:
            valid_datasets.append(ds_name)

    out = {}
    for ds_name in valid_datasets:
        data = load_monash(ds_name)['series_value']
        lengths = np.array([len(X) for X in data])
        candidates = np.nonzero(lengths >= L)[0]
        for c_idx in candidates:
            X = data[c_idx].to_numpy()
            # Find random start and end points
            end = rng.randint(L, len(X))
            start = end - L
            sub_X = X[start:end]

            # Skip if constant
            if np.all(sub_X[0] == sub_X):
                continue

            assert len(sub_X) == L, print(sub_X.shape)

            try:
                out[ds_name].append(sub_X)
            except KeyError:
                out[ds_name] = [sub_X]

    ds_keys = list(out.keys())
    for key in ds_keys:
        out[key] = np.vstack(out[key])

    print([f'{k} {v.shape}' for k, v in out.items()])

    with open(out_path, 'wb') as _f:
        pickle.dump(out, _f)


if __name__ == '__main__':
    lag = 5

    if not exists('data/monash_700.pickle'):
        generate_subsets(700, 'data/monash_700.pickle', random_state=1337)

    ds_pretrain, ds_exp = split_monash_700()
    print('pretrain_ds', ds_pretrain.shape, 'experiment_ds', ds_exp.shape)
    # Experiment data gets windowed and split on demand
    np.save('data/exp_data.npy', ds_exp)

    # Split pretrain dataset into train and val at the middle
    X_train, X_val = ds_pretrain[:, :500], ds_pretrain[:, 500:]
    mu_train, std_train = np.mean(X_train, axis=1), np.std(X_train, axis=1)

    # Normalize
    X_train = (X_train.T - mu_train) / std_train.T
    X_val = (X_val.T - mu_train) / std_train.T

    X_train_w = []
    y_train_w = []
    for _X in X_train:
        a, b = windowing(_X, lag)
        X_train_w.append(a)
        y_train_w.append(b)

    X_train = np.concatenate(X_train_w)
    y_train = np.concatenate(y_train_w)

    X_val_w = []
    y_val_w = []
    for _X in X_val:
        a, b = windowing(_X, lag)
        X_val_w.append(a)
        y_val_w.append(b)

    X_val = np.concatenate(X_val_w)
    y_val = np.concatenate(y_val_w)

    np.save('data/X_pt_train.npy', X_train)
    np.save('data/y_pt_train.npy', y_train)
    np.save('data/X_pt_val.npy', X_val)
    np.save('data/y_pt_val.npy', y_val)

    print('Pretraining shapes')
    print('train', X_train.shape, y_train.shape)
    print('val', X_val.shape, y_val.shape)

