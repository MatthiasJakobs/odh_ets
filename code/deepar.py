import mxnet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import OffsetSplitter
from gluonts.mx import DeepAREstimator, Trainer
from sklearn.metrics import mean_squared_error as mse
from seedpy import fixedseed

class DeepARWrapper:

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    # Use X[:-test_size] for training
    # X: np.ndarray
    def fit(self, X, test_size):
        df = pd.DataFrame(X, columns=['target'])
        df.index = pd.to_datetime(df.index, unit='D')

        dataset = PandasDataset(df, target="target")

        splitter = OffsetSplitter(offset=-test_size)
        training_data, _ = splitter.split(dataset)
        self.model = DeepAREstimator(prediction_length=1, freq="D", trainer=Trainer(epochs=self.n_epochs)).train(training_data)
        self.is_fitted = True

    # Use X[-test_size:] for prediction
    # X: np.ndarray
    def predict(self, X, test_size):
        if not self.is_fitted:
            raise RuntimeError('Model is not fitted')
        df = pd.DataFrame(X, columns=['target'])
        df.index = pd.to_datetime(df.index, unit='D')

        dataset = PandasDataset(df, target="target")

        splitter = OffsetSplitter(offset=-test_size)
        _, test_gen = splitter.split(dataset)

        test_data = test_gen.generate_instances(prediction_length=1, windows=test_size, distance=1)

        forecasts = list(self.model.predict(test_data.input))
        predictions = np.array([x.samples.mean() for x in forecasts]).squeeze()
        return predictions

def main():
    # Setup
    rmse = lambda a, b: mse(a, b, squared=False)

    test_size = 175
    n_epochs = 15
    X = np.load('data/exp_data.npy')[0]
    mu, std = np.mean(X[:-test_size]), np.std(X[:-test_size])
    X = (X - mu) / std

    y_test = X[-test_size:]

    # Train DeepAR
    with fixedseed(np, 0):
        model = DeepARWrapper(n_epochs=n_epochs)
        model.fit(X, test_size)
        prediction = model.predict(X, test_size)
    print(rmse(y_test, prediction))

    plt.figure()
    plt.plot(X, color='black', label='X')
    plt.plot(np.arange(test_size)+(len(X) - test_size), prediction, color='red', label='DeepAR (default params)')
    plt.tight_layout()
    plt.legend()
    plt.savefig('plots/deepar_test.png')

if __name__ == '__main__':
    main()

