import numpy as np
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels[0], 3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(channels[0], channels[1], 3, padding='same'),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels[1], channels[0], 3, padding=1),
            nn.ReLU(),
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

class MultiForecaster(nn.Module):

    def __init__(self, encoder_decoder, n_encoder_filters, n_encoder_lag, train_encoder=True):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.forecasters = nn.ModuleList()
        self.train_encoder = train_encoder

        # Different output forecasters

        # Fully connceted
        for hidden_size in [64, 32]:
            m = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_encoder_filters * n_encoder_lag, hidden_size),
                #nn.Dropout(0.8),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                #nn.Dropout(0.8),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
            self.forecasters.append(m)

        # Conv
        for filters in [64, 32]:
            m = nn.Sequential(
                nn.Conv1d(n_encoder_filters, filters, 3, padding='same'),
                nn.ReLU(),
                nn.Conv1d(filters, filters // 2, 3, padding='same'),
                nn.ReLU(),
                nn.Flatten(),
                #nn.Dropout(0.8),
                nn.Linear(n_encoder_lag * filters // 2, 1),
            )
            self.forecasters.append(m)

    def forward(self, x, use_decoder=True):
        if self.train_encoder:
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
    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().unsqueeze(1)
        return self(x, use_decoder=False).squeeze().mean(axis=-1).cpu().numpy()


    # Get the best possible prediction
    def get_best_loss(self, X, y, loss):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().unsqueeze(1)

        with torch.no_grad():
            self.eval()
            preds = self(X, use_decoder=False).cpu().numpy()
    
        # Choose best prediction at each step
        final_prediction = []
        for t in range(len(X)):
            preds_t = preds[t]
            best_l = 1e10
            best_model_idx = 0
            for idx, pred in enumerate(preds_t):
                l = loss(y[t].reshape(1), pred)
                if l < best_l:
                    best_l = l
                    best_model_idx = idx

            final_prediction.append(preds_t[best_model_idx])

        return loss(np.concatenate(final_prediction), y)
