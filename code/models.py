from dataclasses import replace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tsx.datasets.utils import windowing
from tsx.models import SoftDecisionTreeRegressor
from tsx.metrics import mse
from tsx.distances import dtw, euclidean
from scipy.special import softmax
from tsx.utils import to_random_state
from seedpy import fixedseed

from roc_tools import find_closest_rocs, cluster_rocs, select_topm, drift_detected, get_topm
from utils import smape

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

class PolynomialRegression(nn.Module):

    def __init__(self, input_size, degree, mask=None):
        super().__init__()
        self.input_size = input_size
        self.degree = degree

        if mask is not None:
            self.mask = torch.Tensor(mask)
        else:
            self.mask = torch.ones(input_size)

        # How many values per datapoint
        self.linear = nn.Linear(degree * input_size, 1)

    def forward(self, X):
        self.mask = self.mask.to(X.device)
        return self.linear(self.polynomial_features(self.mask[None, :] * X))

    def polynomial_features(self, X):
        return torch.hstack([X ** i for i in range(1, self.degree + 1)])

class MaskedLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, mask=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if mask is not None:
            self.mask = torch.Tensor(mask)
        else:
            self.mask = torch.ones(in_features)

    def forward(self, X):
        self.mask = self.mask.to(X.device)
        X = self.mask[None, :] * X
        return super().forward(X)


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
        for hidden_size in [64, 32, 16, 8]:
            m = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_encoder_filters * n_encoder_lag, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
            self.forecasters.append(m)

        # Fully connceted 2
        for hidden_size in [64, 32, 16, 8]:
            m = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_encoder_filters * n_encoder_lag, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
            self.forecasters.append(m)

        # Conv
        for filters in [64, 32, 16, 8]:
            m = nn.Sequential(
                nn.Conv1d(n_encoder_filters, filters, 3, padding='same'),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(n_encoder_lag * filters, 1),
            )
            self.forecasters.append(m)

        # SDTs
        for depths in [2, 4, 8]:
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

        with fixedseed(np, seed=7273491):
            masks = np.random.binomial(1, 0.5, size=(5, n_encoder_filters * n_encoder_lag)).tolist()
        for i in range(5):
            self.forecasters.append(
                nn.Sequential(
                    nn.Flatten(),
                    MaskedLinear(n_encoder_filters * n_encoder_lag, 1, mask=masks[i])
                )
            )

        # Polynomial Regression 
        with fixedseed(np, seed=81745612):
            masks = np.random.binomial(1, 0.5, size=(5, n_encoder_filters * n_encoder_lag)).tolist()
        self.forecasters.append(
            nn.Sequential(
                nn.Flatten(),
                PolynomialRegression(n_encoder_filters * n_encoder_lag, degree=3, mask=None),
            )
        )

        for i in range(4):
            self.forecasters.append(
                nn.Sequential(
                    nn.Flatten(),
                    PolynomialRegression(n_encoder_filters * n_encoder_lag, degree=3, mask=masks[i]),
                )
            )

    def forward(self, x, use_decoder=True, train_encoder=True, return_feats=False):
        if train_encoder:
            encoded = self.encoder_decoder.encode(x)
        else:
            with torch.no_grad():
                encoded = self.encoder_decoder.encode(x)

        predictions = torch.concat([m(encoded).reshape(-1, 1, 1) for m in self.forecasters], dim=1)
        
        if use_decoder:
            reconstructed = self.encoder_decoder.decode(encoded)
            return reconstructed, predictions
        
        if return_feats:
            return encoded, predictions
        else:
            return None, predictions

    def get_device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(self, X, return_mean=True):
        test_size = int(0.25 * len(X))
        X_test = X[-test_size:][:-1]
        return self.predict_single(X_test, return_mean=return_mean)

    @torch.no_grad()
    def predict_single(self, X, return_mean=True):
        x, _ = windowing(X, lag=5)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().unsqueeze(1).to(self.get_device())

        out = self(x, use_decoder=False, train_encoder=False)[1].squeeze()

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

    def _get_grad(self, X, output):
        return torch.autograd.grad(outputs=output, inputs=X, grad_outputs=torch.ones_like(output))[0]

    # Batched Gradcam for speedup
    def _gradcam(self, feats, preds):
        gradients = self._get_grad(feats, preds)
        feats = feats.detach()

        w = torch.mean(gradients, axis=-1)

        cam = torch.zeros_like(feats[:, 0])
        for k in range(feats.shape[1]):
            cam += w[:, k, None] * feats[:, k]

        cam = torch.nn.functional.relu(cam).squeeze().numpy()
        return cam

    def restrict_rocs(self, rocs, exact_length):
        new_rocs = [[] for _ in range(len(self.forecasters))]
        for f_idx, forecaster_rocs in enumerate(rocs):
            for r in forecaster_rocs:
                r = r.squeeze()
                if len(r) == exact_length:
                    new_rocs[f_idx].append(r)

        return new_rocs

    def build_rocs(self, X_val, only_best=False):

        # For a 2d array, split every row at zeros and add them to a list if their size is longer than 2
        def split_array_at_zeros(X, activations):
            cam = [[] for _ in range(len(X))]
            for row_idx, row in enumerate(activations):
                zero_indices = np.where(row == 0)[0]
                last_index = 0

                # Add entire datapoint if no zeros
                if len(zero_indices) == 0:
                    cam[row_idx].append(X[row_idx].squeeze())
                    continue


                # Only add subsequences if they are longer than 2
                for zi in zero_indices:
                    sub = row[last_index:zi]
                    if len(sub) > 2:
                        cam[row_idx].append(X[row_idx, last_index:zi].squeeze())

                    last_index = zi+1

            return cam

        n_forecasters = len(self.forecasters)
        x, y = windowing(X_val, lag=5, use_torch=True)

        x = x.unsqueeze(1)
        all_cams = [ [] for _ in range(n_forecasters)]
        losses = np.zeros((n_forecasters, len(x)))

        feats = self.encoder_decoder.encode(x)
        for f_idx, fc in enumerate(self.forecasters):
            pred = fc(feats)
            l = (pred - y)**2
            cam = self._gradcam(feats, l)
            losses[f_idx] = l.detach().numpy().squeeze()

            split_rocs = split_array_at_zeros(x, cam)
            all_cams[f_idx].extend(split_rocs)

        # Compact rocs
        final_rocs = [ [] for _ in range(n_forecasters)]
        rocs = [[] for _ in range(n_forecasters)]
        if only_best:
            # Find best forecaster for each datapoint
            losses = losses.T
            best_forecasters = []
            for datapoint_idx in range(len(x)):
                best_forecaster_idx = np.argmin(losses[datapoint_idx])
                best_forecasters.append(best_forecaster_idx)
                rocs[best_forecaster_idx].append(all_cams[best_forecaster_idx][datapoint_idx])

        else:
            # Take all
            rocs = all_cams

        for f_idx, forecaster_rocs in enumerate(rocs):
            for datapoint_rocs in forecaster_rocs:
                final_rocs[f_idx].extend(datapoint_rocs)

        return final_rocs

    def run(self, X_test, dist_fn=dtw):
        X, Y = windowing(X_test, lag=5, use_torch=True)
        prediction = []
        for x, _ in zip(X, Y):
            # Find closest RoC
            closest_forecaster = 0
            smallest_distance = float('inf')
            for f_idx in range(len(self.forecasters)):
                for r in self.rocs[f_idx]:
                    if r.shape[0] == 0:
                        continue
                    d = dist_fn(r, x)
                    if d < smallest_distance:
                        closest_forecaster = f_idx
                        smallest_distance = d

            with torch.no_grad():
                feats = self.encoder_decoder.encode(x.unsqueeze(0).unsqueeze(0))
                pred = self.forecasters[closest_forecaster](feats)
                pred = pred.numpy().squeeze()
                prediction.append(pred)

        return np.hstack([X_test[:5], np.stack(prediction)])

    def predict_weighted_drifts(self, X_val, X_test, drifts, max_dist=0.999, k=1, dist_fn=None, lag=5):
        X, _ = windowing(X_test, lag=lag, use_torch=True)
        prediction = []
        weights = np.zeros((X.shape[0], len(self.forecasters)))

        val_start = 0
        val_stop = len(X_val) + lag
        X_complete = np.concatenate([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        for idx, x in enumerate(X):

            val_start += 1
            val_stop += 1
            current_val = X_complete[val_start:val_stop]

            # Recreate rocs if necessary
            if idx in drifts:
                val_start = val_stop - len(X_val) - lag
                current_val = X_complete[val_start:val_stop]
                new_rocs = self.build_rocs(current_val)
                for i in range(len(self.forecasters)):
                    self.rocs[i].extend(new_rocs[i])
                if dist_fn == euclidean:
                    self.rocs = self.restrict_rocs(self.rocs, lag)

            # Find closest RoC for each model
            dist_vec = np.zeros((len(self.forecasters)))
            for f_idx in range(len(self.forecasters)):
                distances = [dist_fn(r, x) for r in self.rocs[f_idx] if r.shape[0] != 0]
                if len(distances) == 0:
                    dist_vec[f_idx] = max_dist
                else:
                    dist_vec[f_idx] = np.sort(distances)[:k].mean()

            # Compute weighting of ensemble
            w = dist_vec
            weights[idx] = w.squeeze()

            with torch.no_grad():
                feats = self.encoder_decoder.encode(x.unsqueeze(0).unsqueeze(0))

                if dist_fn == smape:
                    tmp =[(1-w[f_idx]) * self.forecasters[f_idx](feats) for f_idx in range(len(self.forecasters))] 
                    pred = sum(tmp) / (1-w).sum()
                else:
                    w = softmax(-w)
                    pred = sum([w[idx] * self.forecasters[idx](feats) for idx in range(len(self.forecasters))])

                prediction.append(pred.item())

        return weights, np.hstack([X_test[:5], np.array(prediction)])

    def predict_weighted(self, X_test, max_dist=0.999, k=1, dist_fn=None):
        X, _ = windowing(X_test, lag=5, use_torch=True)
        prediction = []
        weights = np.zeros((X.shape[0], len(self.forecasters)))

        for idx, x in enumerate(X):
            # Find closest RoC for each model
            dist_vec = np.zeros((len(self.forecasters)))
            for f_idx in range(len(self.forecasters)):
                distances = [dist_fn(r, x) for r in self.rocs[f_idx] if r.shape[0] != 0]
                if len(distances) == 0:
                    dist_vec[f_idx] = max_dist
                else:
                    dist_vec[f_idx] = np.sort(distances)[:k].mean()

            # Compute weighting of ensemble
            w = dist_vec

            with torch.no_grad():
                feats = self.encoder_decoder.encode(x.unsqueeze(0).unsqueeze(0))

                if dist_fn == smape:
                    tmp =[(1-w[f_idx]) * self.forecasters[f_idx](feats) for f_idx in range(len(self.forecasters))] 
                    pred = sum(tmp) / (1-w).sum()
                    w = (1-w) / (1-w).sum()
                else:
                    w = softmax(-w)
                    pred = sum([w[idx] * self.forecasters[idx](feats) for idx in range(len(self.forecasters))])

                weights[idx] = w.squeeze()
                prediction.append(pred.item())

        return weights, np.hstack([X_test[:5], np.array(prediction)])

    def get_best_prediction(self, X, X_test):
        all_preds = self.predict(X, return_mean=False)
        losses = (all_preds - X_test[:, None])**2
        best_possible_predictions = []
        for l_idx, l in enumerate(losses):
            best_idx = np.argmin(l)
            best_possible_predictions.append(all_preds[l_idx, best_idx].squeeze())
        best_possible_predictions = np.array(best_possible_predictions)

        # Set to gt here for fairer comparison
        best_possible_predictions[:5] = X_test[:5]

        return best_possible_predictions

    def recluster_and_reselect(self, x, nr_clusters_ensemble=3, skip_clustering=False, skip_topm=False, nr_select=None, dist_fn=euclidean):
        # Find closest time series in each models RoC to x
        models, rocs = find_closest_rocs(x, self.rocs, dist_fn=dist_fn)

        # Cluster all RoCs into nr_clusters_ensemble clusters
        if not skip_clustering:
            models, rocs = cluster_rocs(models, rocs, nr_clusters_ensemble, dist_fn=dist_fn)

        # Calculate upper and lower bound of delta (radius of circle)
        numpy_regions = [r.numpy() for r in rocs]
        lower_bound = 0.5 * np.sqrt(np.sum((numpy_regions - np.mean(numpy_regions, axis=0))**2) / nr_clusters_ensemble)
        upper_bound = np.sqrt(np.sum((numpy_regions - x.numpy())**2) / nr_clusters_ensemble)
        #assert lower_bound <= upper_bound, "Lower bound bigger than upper bound"
        if not lower_bound <= upper_bound:
            return [], []

        # Select topm models according to upper bound
        if not skip_topm:
            selected_models, selected_rocs = select_topm(models, rocs, x, upper_bound, dist_fn=dist_fn)
            return selected_models, selected_rocs

        # Select fixed number
        if nr_select is not None:
            selected_indices = np.argsort([dist_fn(x.squeeze(), r.squeeze()) for r in rocs])[:self.nr_select]
            selected_models = np.array(models)[selected_indices]
            selected_rocs = [rocs[idx] for idx in selected_indices]
            return selected_models, selected_rocs
        return models, rocs 

    def predict_clustered(self, X_val, X_test, lag=5, weighting=False, random_state=None, dist_fn=euclidean):

        rng = to_random_state(random_state)

        @torch.no_grad()
        def ensemble_predict(x, ensemble, w):
            feats = self.encoder_decoder.encode(x.unsqueeze(0).unsqueeze(0))

            if weighting:
                models = [self.forecasters[idx] for idx in ensemble]
                softmax_weights = softmax([-w[idx] for idx in ensemble])
                pred = sum([softmax_weights[idx] * models[idx](feats) for idx in range(len(ensemble))])
                return pred.item()
            else:
                return torch.cat([self.forecasters[f_idx](feats) for f_idx in ensemble]).mean().item()

        def calc_weights(x, dist_fn):
            if dist_fn == euclidean:
                max_value = 100
            elif dist_fn  == dtw:
                max_value = 100
            elif dist_fn == smape:
                max_value = 0.999

            dist_vec = np.zeros((len(self.forecasters)))
            for f_idx in range(len(self.forecasters)):
                distances = [dist_fn(r, x) for r in self.rocs[f_idx] if r.shape[0] != 0]
                if len(distances) == 0:
                    dist_vec[f_idx] = max_value
                else:
                    dist_vec[f_idx] = np.min(distances)

            return dist_vec

        # hyperparameters
        nr_clusters_ensemble = 10

        predictions = []
        mean_residuals = []
        means = []
        ensemble_residuals = []
        topm_buffer = []

        val_start = 0
        val_stop = len(X_val) + lag
        X_complete = torch.cat([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        # Remove all RoCs that have not length lag
        if dist_fn == euclidean:
            self.rocs = self.restrict_rocs(self.rocs, lag)

        means = [torch.mean(current_val).numpy()]

        # First iteration 
        x = X_test[:lag]
        topm, _ = self.recluster_and_reselect(x, nr_clusters_ensemble, dist_fn=dist_fn)

        # Compute min distance to x from the all models
        _, closest_rocs = find_closest_rocs(x, self.rocs, dist_fn=dist_fn)
        mu_d = np.min([dist_fn(r, x) for r in closest_rocs if r.shape[0] != 0])

        # topm_buffer contains the models used for prediction
        topm_buffer = get_topm(topm_buffer, topm, rng, len(self.forecasters))

        # Compute weighting and ensemble predict
        w = calc_weights(x, dist_fn)
        predictions.append(ensemble_predict(x, topm_buffer, w))

        # Subsequent iterations
        for target_idx in range(lag+1, len(X_test)):
            f_test = (target_idx-lag)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 
            val_start += 1
            val_stop += 1
            
            current_val = X_complete[val_start:val_stop]
            means.append(torch.mean(current_val).numpy())
            mean_residuals.append(means[-1]-means[-2])

            _, closest_rocs = find_closest_rocs(x, self.rocs, dist_fn=dist_fn)
            ensemble_residuals.append(mu_d - np.min([dist_fn(r, x) for r in closest_rocs if r.shape[0] != 0]))

            # Drift detection
            drift_type_one = drift_detected(mean_residuals, len(current_val), R=1.5)
            drift_type_two = drift_detected(ensemble_residuals, len(current_val), R=100)

            if drift_type_one:
                val_start = val_stop - len(X_val) - lag
                current_val = X_complete[val_start:val_stop]
                mean_residuals = []
                means = [torch.mean(current_val).numpy()]
                self.rocs = self.build_rocs(current_val)
                if dist_fn == euclidean:
                    self.rocs = self.restrict_rocs(self.rocs, lag)

            if drift_type_one or drift_type_two:
                topm, _ = self.recluster_and_reselect(x, target_idx)
                if drift_type_two:
                    _, closest_rocs = find_closest_rocs(x, self.rocs)
                    mu_d = np.min([dist_fn(r, x) for r in closest_rocs if r.shape[0] != 0])
                    ensemble_residuals = []
                topm_buffer = get_topm(topm_buffer, topm, rng, len(self.forecasters))

            w = calc_weights(x, dist_fn)
            predictions.append(ensemble_predict(x, topm_buffer, w))

        return np.concatenate([X_test[:lag].numpy(), np.array(predictions)])
