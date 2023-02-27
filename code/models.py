import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tsx.datasets.utils import windowing
from tsx.models import SoftDecisionTreeRegressor
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_snapshot = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            with torch.no_grad():
                self.best_snapshot = model.state_dict().copy()


class EncoderDecoder(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels[0], 3, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(channels[0], channels[1], 3, padding='same'),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels[1], channels[0], 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose1d(channels[0], 1, 3, padding=1)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def transform(self, x):
        return self.encoder(x).cpu().numpy()

class MultiForecaster(nn.Module):

    def __init__(self, hyperparameters, encoder_decoder, n_encoder_filters, n_encoder_lag):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.forecasters = nn.ModuleList()
        self.n_epochs = hyperparameters['n_epochs']
        self.learning_rate = hyperparameters['learning_rate']
        self.lagrange_multiplier = hyperparameters['lagrange_multiplier']
        self.combined_loss_function = self.lagrange_multiplier is not None

        # Different output forecasters

        # Fully connceted
        for hidden_size in [64, 32]:
            m = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_encoder_filters * n_encoder_lag, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 1),
            )
            self.forecasters.append(m)

        # Conv
        for filters in [64, 32]:
            m = nn.Sequential(
                nn.Conv1d(n_encoder_filters, filters, 3, padding='same'),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Flatten(),
                nn.Linear(n_encoder_lag * filters, 1),
            )
            self.forecasters.append(m)

        # SDTs
        for depths in [5, 7]:
            m = nn.Sequential(
                nn.Flatten(),
                SoftDecisionTreeRegressor(n_encoder_filters * n_encoder_lag, depth=depths),
            )
            self.forecasters.append(m)

        # Linear
        m = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_encoder_filters * n_encoder_lag, 1),
        )
        self.forecasters.append(m)

    def forward(self, x, use_decoder=True, train_encoder=True):
        if train_encoder:
            encoded = self.encoder_decoder.encode(x)
        else:
            with torch.no_grad():
                encoded = self.encoder_decoder.encode(x)
        predictions = torch.concat([m(encoded).unsqueeze(1) for m in self.forecasters], dim=1)
        
        if use_decoder:
            reconstructed = self.encoder_decoder.decode(encoded)
            return reconstructed, predictions
        
        return None, predictions

    def get_device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(self, X, return_mean=True):
        test_size = int(0.25 * len(X))
        X_test = X[-test_size:]
        x, _ = windowing(X_test, lag=5)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().unsqueeze(1).to(self.get_device())

        out = self(x, use_decoder=False)[1].squeeze()

        if return_mean and len(out.shape) >= 2:
            out = out.mean(axis=-1)
            to_prepend = x[0][0]
        else:
            to_prepend = x[0][0].unsqueeze(1).repeat(1, len(self.forecasters))

        return np.concatenate([to_prepend.cpu().numpy(), out.cpu().numpy()])

    def fit(self, X, batch_size=128, report_every=10, train_encoder=True, verbose=True, early_stopping_patience=5):

        def _compute_loss(_X, _y):

            # Reshape y to accomodate for multiple outputs
            _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, len(self.forecasters), 1))

            reconstructed, prediction = self(_X, use_decoder=self.combined_loss_function, train_encoder=train_encoder)
            pred_loss = prediction_loss_fn(prediction, _y)
            if self.combined_loss_function:
                rec_loss = reconstruction_loss_fn(reconstructed, _X)
                # Combine the normal loss (pred_loss) with a multiplied version of the reconstruction error
                L = pred_loss + self.lagrange_multiplier * rec_loss
            else:
                L = pred_loss
            
            return L

        device = self.get_device()

        # Take first 75% for training
        train_size = int(0.5 * len(X))
        val_size = int(0.25 * len(X))
        test_size = int(0.25 * len(X))

        X_train = X[:train_size]
        X_val = X[train_size:-test_size]
        X_test = X[-test_size:]

        # Do windoing on train and val
        X_train_w, y_train_w = windowing(X_train, lag=5)
        X_val_w, y_val_w = windowing(X_val, lag=5)

        # Test to fit on train+val and only evaluate on test
        ds_train = TensorDataset(torch.from_numpy(X_train_w).float().unsqueeze(1).to(device), torch.from_numpy(y_train_w).float().to(device))
        ds_val = TensorDataset(torch.from_numpy(X_val_w).float().unsqueeze(1).to(device), torch.from_numpy(y_val_w).float().to(device))
        # dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        # dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
        # TODO: Use val for hyperparameter tuning etc. For now, lets train on all data up until test
        dl_train = DataLoader(ds_train + ds_val, batch_size=batch_size, shuffle=False)
        dl_val = DataLoader(ds_train + ds_val, batch_size=batch_size, shuffle=False)

        log = []

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        reconstruction_loss_fn = nn.MSELoss()
        prediction_loss_fn = nn.MSELoss()

        ES = EarlyStopping(patience=early_stopping_patience)

        for epoch in range(self.n_epochs):
            if ES.early_stop:
                print(f'Stopping after {epoch} epochs')
                self.load_state_dict(ES.best_snapshot)
                break
                    
            self.train()
            train_epoch_loss = []
            val_epoch_loss = []

            for _X, _y in dl_train:
                optimizer.zero_grad()

                L = _compute_loss(_X, _y)
                L.backward()
                optimizer.step()

                train_epoch_loss.append(L.detach())

            if epoch % report_every == 0:
                with torch.no_grad():
                    for _X, _y in dl_val:
                        self.eval()

                        L = _compute_loss(_X, _y)
                        val_epoch_loss.append(L)

                    train_epoch_loss = torch.stack(train_epoch_loss).mean().cpu()
                    val_epoch_loss = torch.stack(val_epoch_loss).mean().cpu()
                    log.append([epoch, train_epoch_loss, val_epoch_loss])

                    ES(val_epoch_loss, self)

                    if verbose:
                        print(f'{epoch} {train_epoch_loss:.5f} | {val_epoch_loss:.5f}')

        log = np.vstack(log)
        return log


    # Get the best possible prediction
    # def get_best_loss(self, X, y, loss):
    #     if isinstance(X, np.ndarray):
    #         X = torch.from_numpy(X).float().unsqueeze(1)

    #     with torch.no_grad():
    #         self.eval()
    #         preds = self(X, use_decoder=False).cpu().numpy()
    
    #     # Choose best prediction at each step
    #     final_prediction = []
    #     for t in range(len(X)):
    #         preds_t = preds[t]
    #         best_l = 1e10
    #         best_model_idx = 0
    #         for idx, pred in enumerate(preds_t):
    #             l = loss(y[t].reshape(1), pred)
    #             if l < best_l:
    #                 best_l = l
    #                 best_model_idx = idx

    #         final_prediction.append(preds_t[best_model_idx])

    #     return loss(np.concatenate(final_prediction), y)
