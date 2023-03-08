###
# Pipeline for model training
###

import numpy as np
import torch
import pandas as pd
import argparse

from seedpy import fixedseed
from sklearn.metrics import mean_squared_error as mse
from os.path import exists
from scipy.stats import entropy
from tsx.distances import euclidean

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

# weights shape: (n, b)
# output: R
def mean_entropy(weights):
    b = weights.shape[1]
    entr = entropy(weights, base=b, axis=1)
    return np.mean(entr)

def smape(a, b):
    assert len(a.shape) <= 1
    assert len(b.shape) <= 1

    num = np.abs(a - b)
    denom = np.abs(a) + np.abs(b)

    return (num / denom).mean()


def main(override, dry_run):
    # TODO: Notes
    # No thresholding
    # euclidean

    # Global parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    lag = 5

    for ds_name, ds_index in implemented_datasets:
        print(ds_name, ds_index)
        X = load_dataset(ds_name, ds_index)

        #train_cutoff = int(len(X) * 0.5)
        #val_cutoff = train_cutoff + int(len(X) * 0.25)
        #X_train, X_val, X_test = X[0:train_cutoff], X[train_cutoff:val_cutoff], X[val_cutoff:]

        test_size = int(0.25 * len(X))
        train_size = int(0.5 * len(X))

        # TODO: Normalizing? 
        # mu, std = np.mean(X[:-test_size]), np.std(X[:-test_size])
        # X = (X - mu) / std

        X_test = X[-test_size:][:-1]
        X_val = X[train_size:-test_size]

        result_path = f'results/test_{ds_name}_#{ds_index}.csv'
        print(result_path)
        df = pd.read_csv(result_path, header=0, index_col=0)

        # Skip if experiment is already done
        if override or 'e2e_ensemble' not in df.columns or 'e2e_roc_selection' not in df.columns or 'e2e_weighted_ensemble' not in df.columns:
            savepath = f'models/{ds_name}_#{ds_index}_e2e.pth'

            hyp = hyperparameters['e2e_no_decoder']
            # Train if model is not already trained
            with fixedseed(torch, seed=hyp['model_init_seed']):
                _encdec = EncoderDecoder(hyp['n_channels']).to(device)
                e2e_no_decoder = MultiForecaster(hyp, _encdec, hyp['n_channels'][-1], lag).to(device)
            
                if not exists(savepath):
                    e2e_no_decoder.fit(X, batch_size, train_encoder=True, verbose=False)
                    torch.save(e2e_no_decoder.state_dict(), savepath)
                else:
                    e2e_no_decoder.load_state_dict(torch.load(savepath))

            e2e_no_decoder.eval()

            e2e_no_decoder.to('cpu')
            all_preds = e2e_no_decoder.predict(X, return_mean=False)
            ensemble_preds = all_preds.mean(axis=-1)

            ### Run roc selection method
            e2e_no_decoder.rocs = e2e_no_decoder.build_rocs(X_val)
            roc_preds = e2e_no_decoder.run(X_test)

            roc_distribution = np.array([len(roc) for roc in e2e_no_decoder.rocs])
            print('roc_dist', roc_distribution)
            df['e2e_ensemble'] = ensemble_preds
            df['e2e_roc_selection'] = roc_preds
            
            ### Compute weighting based on distance
            e2e_no_decoder.rocs = e2e_no_decoder.restrict_rocs(e2e_no_decoder.rocs, exact_length=5)
            _, weighted_prediction1 = e2e_no_decoder.predict_weighted(X_test, k=1, dist_fn=smape)
            df['e2e_weighted_ensemble_smape_k=1'] = weighted_prediction1
            w, weighted_prediction2 = e2e_no_decoder.predict_weighted(X_test, k=3, dist_fn=smape)
            df['e2e_weighted_ensemble_smape_k=3'] = weighted_prediction2
            w, weighted_prediction3 = e2e_no_decoder.predict_weighted(X_test, k=5, dist_fn=smape)
            df['e2e_weighted_ensemble_smape_k=5'] = weighted_prediction3
            w, weighted_prediction4 = e2e_no_decoder.predict_weighted(X_test, k=7, dist_fn=smape)
            df['e2e_weighted_ensemble_smape_k=7'] = weighted_prediction4

            ### Save the individual predictions as well
            single_names = ['e2e-fcn1', 'e2e-fcn2', 'e2e-conv1', 'e2e-conv2', 'e2e-sdt1', 'e2e-sdt2', 'e2e-lin']
            for single_name, single_pred in zip(single_names, all_preds.T):
                df[single_name] = single_pred

            ### Get a baseline with the best prediction possible
            best_possible_predictions = e2e_no_decoder.get_best_prediction(X, X_test)
            df['best_possible_selection'] = best_possible_predictions

        if override or 'deepar' not in df.columns:

            # deepar baseline
            savepath = f'models/{ds_name}_#{ds_index}_deepar'
            hyp = hyperparameters['deepar']
            deepar = DeepARWrapper(n_epochs=hyp['n_epochs'], lag=lag)
            if not exists(savepath):
                with fixedseed(np, seed=hyp['model_init_seed']):
                    deepar.fit(X)
                    deepar.save(savepath)
            else:
                deepar.load(savepath)

            preds = deepar.predict(X)

            df['deepar'] = preds
    
        if not dry_run:
            print(df.shape)
            df.to_csv(result_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--override', action='store_true', default=False)
    parser.add_argument('--dry_run', action='store_true', default=False)
    args = parser.parse_args()
    main(override=args.override, dry_run=args.dry_run)
