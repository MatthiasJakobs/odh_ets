import numpy as np
import pandas as pd
from os.path import exists, join
from os import makedirs

from datasets.dataloading import implemented_datasets
from critdd import Diagram
from sklearn.metrics import mean_squared_error as mse

#results_path = '../ecml_23_sax/results/main_experiments_new/'
results_path = 'results' 


def create_cd_diagram(compositors_to_check, output_name, title, treatment_mapping=None):

    used_datasets = 0
    total_datasets = len(implemented_datasets)

    losses = []

    for ds_name, ds_index in implemented_datasets:
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

    # Sort 
    avg_ranks = diagram.average_ranks
    sorted_indices = np.argsort(-avg_ranks)
    treatment_names = [treatment_names[idx] for idx in sorted_indices]
    avg_ranks = avg_ranks[sorted_indices]
    diagram.to_file(output_name, title=title)

def main():
    print(results_path)
    
    # Create plots
    makedirs('plots', exist_ok=True)
    compositors_to_check = [
        'deepar',
        'e2e_ensemble',
        'e2e_roc_selection',
        'e2e_weighted_ensemble_smape_k=1',
        'e2e_weighted_ensemble_smape_k=3',
        'e2e_weighted_ensemble_smape_k=5',
        'e2e_weighted_ensemble_smape_k=7',
        #'e2e-fcn1',
        # 'e2e-fcn2',
        # 'e2e-conv1',
        # 'e2e-conv2',
        # 'e2e-sdt1',
        # 'e2e-sdt2',
        # 'e2e-lin',
        'best_possible_selection'
    ]
    treatment_mapping = {
        'deepar': 'DeepAR',
        'e2e_ensemble': 'e2e-ensemble',
        'e2e_roc_selection': 'e2e-roc-selection',
        'e2e_weighted_ensemble_smape_k=1': 'e2e-weighted-smape-k=1',
        'e2e_weighted_ensemble_smape_k=3': 'e2e-weighted-smape-k=3',
        'e2e_weighted_ensemble_smape_k=5': 'e2e-weighted-smape-k=5',
        'e2e_weighted_ensemble_smape_k=7': 'e2e-weighted-smape-k=7',
        'best_possible_selection': 'best-possible-selection'
    }
    output_name = 'plots/cd-comparison.tex'
    title = 'Comparison to sota'
    create_cd_diagram(compositors_to_check, output_name, title, treatment_mapping=treatment_mapping)

if __name__ == '__main__':
    main()
