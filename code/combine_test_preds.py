import pandas as pd

from os.path import exists, join

from datasets.dataloading import get_all_datasets

results_path = 'results' 
external_results_path = '../redo_ecml/results' 

take_external = ['OEP-ROC-ST', 'OEP-ROC-15']

def main():

    for ds_name, ds_index in get_all_datasets():
        if ds_name == 'EOGHorizontalSignal' and ds_index == 0:
            continue
        ds_path = join(results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        external_ds_path = join(external_results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        if exists(ds_path) and exists(external_ds_path):
            df_local = pd.read_csv(ds_path, header=0, index_col=0)
            df_external = pd.read_csv(external_ds_path, header=0, index_col=0)

            for external_method in take_external:
                df_local[external_method] = df_external[external_method].values

            df_local.to_csv(ds_path)


if __name__ == '__main__':
    main()