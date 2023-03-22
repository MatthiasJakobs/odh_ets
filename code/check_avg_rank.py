import numpy as np
import pandas as pd
from os.path import exists, join
from os import makedirs

from datasets.dataloading import get_all_datasets
from critdd import Diagram
from sklearn.metrics import mean_squared_error as mse

#results_path = '../ecml_23_sax/results/main_experiments_new/'
results_path = 'results' 

def create_cd_diagram(compositors_to_check, output_name, title, treatment_mapping=None):

    used_datasets = 0
    all_datasets = get_all_datasets()
    total_datasets = len(all_datasets)

    losses = []

    for ds_name, ds_index in all_datasets:
        if ds_name == 'EOGHorizontalSignal' and ds_index == 0:
            continue
        ds_path = join(results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        if exists(ds_path):
            df = pd.read_csv(ds_path, header=0, index_col=0)
            if all([comp_name in df.columns for comp_name in compositors_to_check]):
                
                scores = []
                y = df['y'].to_numpy().squeeze()
                for comp_name in compositors_to_check:
                    pred = df[comp_name].to_numpy().squeeze()
                    scores.append(mse(y, pred, squared=False)) # RMSE

                losses.append(np.array(scores))
                used_datasets += 1

    print(f'Used {used_datasets} datasets from {total_datasets}')
    losses = np.vstack(losses)

    if treatment_mapping is not None:
        treatment_names = [treatment_mapping[name] if name in treatment_mapping.keys() else name for name in compositors_to_check]
    else:
        treatment_names = compositors_to_check

    diagram = Diagram(
        losses,
        treatment_names=treatment_names,
        maximize_outcome=False
    )

    diagram.to_file(output_name)

def main():
    print(results_path)
    
    # Create plots
    makedirs('plots', exist_ok=True)
    compositors_to_check = [
        'deepar',
        'transformer',
        'nbeats',
        'e2e_ensemble',
        #'e2e_roc_selection',
        # 'e2e_weighted_ensemble_smape_k=1',
        # 'e2e_weighted_ensemble_smape_k=3',
        # 'e2e_weighted_ensemble_smape_k=5',
        # 'e2e_weighted_ensemble_smape_k=7',
        # 'e2e_clustering_average',
        # 'e2e_clustering_weighted',
        # 'e2e-fcn1',
        # 'e2e-fcn2',
        # 'e2e-conv1',
        # 'e2e-conv2',
        # 'e2e-sdt1',
        # 'e2e-sdt2',
        # 'e2e-lin',
        'weighted_ensemble_euclidean',
        'weighted_ensemble_dtw',
        'weighted_ensemble_smape',
        'weighted_ensemble_driftaware_euclidean',
        'weighted_ensemble_driftaware_dtw',
        'weighted_ensemble_driftaware_smape',
        'weighted_ensemble_drift_periodic_euclidean',
        'weighted_ensemble_drift_periodic_dtw',
        'weighted_ensemble_drift_periodic_smape',
        'best_possible_selection',
        'OEP-ROC-15'
    ]
    treatment_mapping = {
        'OEP-ROC-15': 'OEP-ROC',
        'deepar': 'DeepAR',
        'e2e_ensemble': 'e2e-ensemble',
        'e2e_roc_selection': 'e2e-roc-selection',
        'e2e_weighted_ensemble_smape_k=1': 'e2e-weighted-smape-k=1',
        'e2e_weighted_ensemble_smape_k=3': 'e2e-weighted-smape-k=3',
        'e2e_weighted_ensemble_smape_k=5': 'e2e-weighted-smape-k=5',
        'e2e_weighted_ensemble_smape_k=7': 'e2e-weighted-smape-k=7',
        'best_possible_selection': 'best-possible-selection',
        'weighted_ensemble_dtw': 'we_dtw',
        'weighted_ensemble_smape': 'we_smape',
        'weighted_ensemble_euclidean': 'we_eucl',
        'weighted_ensemble_driftaware_dtw': 'we_drift_dtw',
        'weighted_ensemble_driftaware_smape': 'we_drift_smape',
        'weighted_ensemble_driftaware_euclidean': 'we_drift_eucl',
        'weighted_ensemble_drift_periodic_dtw': 'we_period_dtw',
        'weighted_ensemble_drift_periodic_smape': 'we_period_smape',
        'weighted_ensemble_drift_periodic_euclidean': 'we_period_eucl',
    }
    output_name = 'plots/cd-comparison.tex'
    title = 'Comparison to sota'
    create_cd_diagram(compositors_to_check, output_name, title, treatment_mapping=treatment_mapping)

if __name__ == '__main__':
    main()
