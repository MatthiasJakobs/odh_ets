import mxnet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import OffsetSplitter
from gluonts.mx import DeepAREstimator, Trainer
from sklearn.metrics import mean_squared_error as mse
from seedpy import fixedseed
from pathlib import Path
from os import makedirs

class DeepARWrapper:

    def __init__(self, n_epochs, lag):
        self.n_epochs = n_epochs
        self.lag = lag
        self.is_fitted = False

    # Use X[:-test_size] for training
    # X: np.ndarray
    def fit(self, X):
        df = pd.DataFrame(X, columns=['target'])
        df.index = pd.to_datetime(df.index, unit='D')

        test_size = int(0.25 * len(X))
        dataset = PandasDataset(df, target="target")

        splitter = OffsetSplitter(offset=-test_size)
        training_data, _ = splitter.split(dataset)
        self.model = DeepAREstimator(prediction_length=1, freq="D", trainer=Trainer(epochs=self.n_epochs)).train(training_data)
        self.is_fitted = True

    # Use X[-test_size:] for prediction
    # X: np.ndarray
    def predict(self, X):
        # if not self.is_fitted:
        #     raise RuntimeError('Model is not fitted')
        df = pd.DataFrame(X, columns=['target'])
        df.index = pd.to_datetime(df.index, unit='D')

        test_size = int(0.25 * len(X))-1

        dataset = PandasDataset(df, target="target")

        splitter = OffsetSplitter(offset=-test_size)
        _, test_gen = splitter.split(dataset)

        test_data = test_gen.generate_instances(prediction_length=1, windows=test_size, distance=1)

        forecasts = list(self.model.predict(test_data.input))
        predictions = np.array([x.samples.mean() for x in forecasts]).squeeze()
        # Fair comparison: Set first `lag` values to gt
        predictions[:self.lag] = X[-test_size:-test_size+self.lag]
        return predictions

    def save(self, path):
        makedirs(path, exist_ok=True)
        self.model.serialize(Path(path))

    def load(self, path):
        from gluonts.model.predictor import Predictor
        self.model = Predictor.deserialize(Path(path))

class GluonTSWrapper:

    def __init__(self, model, n_epochs, lag):
        self.n_epochs = n_epochs
        self.lag = lag
        self.model = model
        self.is_fitted = False

    # Use X[:-test_size] for training
    # X: np.ndarray
    def fit(self, X):
        df = pd.DataFrame(X, columns=['target'])
        df.index = pd.to_datetime(df.index, unit='D')

        test_size = int(0.25 * len(X))
        dataset = PandasDataset(df, target="target")

        splitter = OffsetSplitter(offset=-test_size)
        training_data, _ = splitter.split(dataset)
        self.model = self.model(prediction_length=1, freq="D", trainer=Trainer(epochs=self.n_epochs)).train(training_data)
        self.is_fitted = True

    # Use X[-test_size:] for prediction
    # X: np.ndarray
    def predict(self, X):
        # if not self.is_fitted:
        #     raise RuntimeError('Model is not fitted')
        df = pd.DataFrame(X, columns=['target'])
        df.index = pd.to_datetime(df.index, unit='D')

        test_size = int(0.25 * len(X))-1

        dataset = PandasDataset(df, target="target")

        splitter = OffsetSplitter(offset=-test_size)
        _, test_gen = splitter.split(dataset)

        test_data = test_gen.generate_instances(prediction_length=1, windows=test_size, distance=1)

        forecasts = list(self.model.predict(test_data.input))
        predictions = np.array([x.samples.mean() for x in forecasts]).squeeze()
        # Fair comparison: Set first `lag` values to gt
        predictions[:self.lag] = X[-test_size:-test_size+self.lag]
        return predictions

    def save(self, path):
        makedirs(path, exist_ok=True)
        self.model.serialize(Path(path))

    def load(self, path):
        from gluonts.model.predictor import Predictor
        self.model = Predictor.deserialize(Path(path))

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
        model = DeepARWrapper(n_epochs=n_epochs, lag=5)
        model.fit(X)
        prediction = model.predict(X)
    print(rmse(y_test, prediction))

    plt.figure()
    plt.plot(X, color='black', label='X')
    plt.plot(np.arange(test_size)+(len(X) - test_size), prediction, color='red', label='DeepAR (default params)')
    plt.axvline(x=len(X)-test_size, color='red')
    plt.tight_layout()
    plt.legend()
    plt.savefig('plots/deepar_test.png')

if __name__ == '__main__':
    main()

