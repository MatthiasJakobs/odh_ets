import torch
import pandas as pd

from seedpy import fixedseed
from os import makedirs

from models import EncoderDecoder, MultiForecaster
from deepar import DeepARWrapper
from datasets.dataloading import get_all_datasets, load_dataset
from run_experiments import hyperparameters

def main():
    for ds_name, ds_index in get_all_datasets():
        print(ds_name, ds_index)
        X = load_dataset(ds_name, ds_index)

        #train_cutoff = int(len(X) * 0.5)
        #val_cutoff = train_cutoff + int(len(X) * 0.25)
        #X_train, X_val, X_test = X[0:train_cutoff], X[train_cutoff:val_cutoff], X[val_cutoff:]

        test_size = int(0.25 * len(X))
        train_size = int(0.5 * len(X))
        lag = 5

        X_train = X[:train_size]
        X_test = X[-test_size:][:-1]
        X_val = X[train_size:-test_size]

        makedirs('export/test_preds', exist_ok=True)
        makedirs('export/val_preds', exist_ok=True)
        makedirs('export/train_data', exist_ok=True)
        result_path = f'results/test_{ds_name}_#{ds_index}.csv'
        export_test_path = f'export/test_preds/test_{ds_name}_#{ds_index}.csv'
        export_val_path = f'export/val_preds/val_{ds_name}_#{ds_index}.csv'
        export_train_data_path = f'export/train_data/{ds_name}_#{ds_index}.csv'

        print(result_path)
        df_predictions = pd.read_csv(result_path, header=0, index_col=0)
        df_test = pd.DataFrame()
        df_val = pd.DataFrame()
        df_train_data = pd.DataFrame()

        df_val['y'] = X_val
        df_test['y'] = X_test
        df_train_data['train_data'] = X_train

        savepath = f'models/{ds_name}_#{ds_index}_e2e.pth'

        device = 'cuda'
        hyp = hyperparameters['e2e_no_decoder']
        with fixedseed(torch, seed=hyp['model_init_seed']):
            _encdec = EncoderDecoder(hyp['n_channels']).to(device)
            model = MultiForecaster(hyp, _encdec, hyp['n_channels'][-1], lag).to(device)
        model.load_state_dict(torch.load(savepath))
        model.eval()
        model.to('cpu')

        test_preds = model.predict(X, return_mean=False)
        val_preds = model.predict_single(X_val, return_mean=False)

        single_names = [
            's-fcn1', 
            's-fcn2', 
            's-fcn3', 
            's-fcn4', 
            'l-fcn1', 
            'l-fcn2', 
            'l-fcn3', 
            'l-fcn4', 
            'conv1', 
            'conv2', 
            'conv3', 
            'conv4', 
            'sdt1', 
            'sdt2', 
            'sdt3', 
            'lin',
            'masked-lin1',
            'masked-lin2',
            'masked-lin3',
            'masked-lin4',
            'masked-lin5',
            'poly',
            'masked-poly1',
            'masked-poly2',
            'masked-poly3',
            'masked-poly4',
        ]
        for single_name, val_single_pred, test_single_pred in zip(single_names, val_preds.T, test_preds.T):
            df_val[single_name] = val_single_pred
            df_test[single_name] = test_single_pred

        test_model_names = [
            'best_possible_selection',
            'e2e_ensemble',
            'weighted_ensemble_euclidean',
            'weighted_ensemble_dtw',
            'weighted_ensemble_smape',
            'weighted_ensemble_driftaware_euclidean',
            'weighted_ensemble_driftaware_dtw',
            'weighted_ensemble_driftaware_smape',
            'weighted_ensemble_drift_periodic_euclidean',
            'weighted_ensemble_drift_periodic_dtw',
            'weighted_ensemble_drift_periodic_smape',
            'deepar',
            'transformer',
            'nbeats',
            'OEP-ROC-15',
        ]

        for model_name in test_model_names:
            df_test[model_name] = df_predictions[model_name].values

        df_test.to_csv(export_test_path)
        df_val.to_csv(export_val_path)
        df_train_data.to_csv(export_train_data_path)

if __name__ == '__main__':
    main()
