import numpy as np
import torch
import torch.nn as nn
import pandas as pd

class EncoderDecoder(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels[0], 3, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(channels[0], channels[1], 3, padding='same'),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels[1], channels[0], 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
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

    def __init__(self, encoder_decoder, n_encoder_filters, n_encoder_lag):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.forecasters = nn.ModuleList()

        # Different output forecasters

        # Fully connceted
        for hidden_size in [64, 32]:
            m = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_encoder_filters * n_encoder_lag, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, 1),
            )
            self.forecasters.append(m)

        # Conv
        for filters in [64, 32]:
            m = nn.Sequential(
                nn.Conv1d(n_encoder_filters, filters, 3, padding='same'),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Flatten(),
                nn.Linear(n_encoder_lag * filters, 1),
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
        
        return predictions

    @torch.no_grad()
    def predict(self, x, return_mean=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().unsqueeze(1).to(next(self.parameters()).device)
        out = self(x, use_decoder=False).squeeze()
        if return_mean and len(out.shape) >= 2:
            out = out.mean(axis=-1)
        return np.concatenate([x[0][0].cpu().numpy(), out.cpu().numpy()])

    def fit(self, dl_train, dl_val, hyperparameters, verbose=True):
        n_epochs = hyperparameters['n_epochs']
        combined_loss_function = hyperparameters['combined_loss_function']
        learning_rate = hyperparameters['learning_rate']
        lagrange_multiplier = hyperparameters['lagrange_multiplier']
        report_every = hyperparameters['report_every']

        log = []

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        reconstruction_loss_fn = nn.MSELoss()
        prediction_loss_fn = nn.MSELoss()

        for epoch in range(n_epochs):
            self.train()
            train_epoch_loss = []
            val_epoch_loss = []

            for _X, _y in dl_train:
                optimizer.zero_grad()

                if combined_loss_function:
                    reconstructed, prediction = self(_X, use_decoder=True)

                    # Reshape y to accomodate for multiple outputs
                    _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))

                    rec_loss = reconstruction_loss_fn(reconstructed, _X)
                    pred_loss = prediction_loss_fn(prediction, _y)

                    # Combine the normal loss (pred_loss) with a multiplied version of the reconstruction error
                    L = pred_loss + lagrange_multiplier * rec_loss
                else:
                    prediction = self(_X, use_decoder=False)

                    # Reshape y to accomodate for multiple outputs
                    _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))

                    pred_loss = prediction_loss_fn(prediction, _y)
                    L = pred_loss

                L.backward()
                optimizer.step()

                train_epoch_loss.append(L.detach())

            if epoch % report_every == 0:
                with torch.no_grad():
                    for _X, _y in dl_val:
                        self.eval()

                        if combined_loss_function:
                            reconstructed, prediction = self(_X, use_decoder=True)

                            # Reshape y to accomodate for multiple outputs
                            _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))

                            rec_loss = reconstruction_loss_fn(reconstructed, _X)
                            pred_loss = prediction_loss_fn(prediction, _y)

                            # Combine the normal loss (pred_loss) with a multiplied version of the reconstruction error
                            L = pred_loss + lagrange_multiplier * rec_loss
                        else:
                            prediction = self(_X, use_decoder=False)

                            # Reshape y to accomodate for multiple outputs
                            _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))

                            pred_loss = prediction_loss_fn(prediction, _y)
                            L = pred_loss

                        val_epoch_loss.append(L)

                train_epoch_loss = torch.stack(train_epoch_loss).mean().cpu()
                val_epoch_loss = torch.stack(val_epoch_loss).mean().cpu()
                log.append([train_epoch_loss, val_epoch_loss])
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
