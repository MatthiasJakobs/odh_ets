import numpy as np
import torch
import torch.nn as nn

from os.path import exists
from torch.utils.data import TensorDataset, DataLoader
from tsx.datasets import load_monash, windowing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from seedpy import fixedseed

from models import EncoderDecoder, MultiForecaster

### Global hyperparameters
normalize = True
rng = np.random.RandomState(1471826)
batch_size = 64
lag = 10
###

def train_finetuned(model, dl_train, dl_val):
    ### Hyperparameters
    learning_rate = 2e-5
    epochs = 30
    ###

    pred_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        train_epoch_l1 = []
        train_epoch_l2 = []
        train_epoch_total = []
        val_epoch_l1 = []
        val_epoch_l2 = []
        val_epoch_total = []

        for _X, _y in dl_train:
            optimizer.zero_grad()
            prediction = model(_X, use_decoder=False)
            _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))
            
            loss1 = 0
            loss2 = pred_loss(prediction, _y)
            loss = loss2

            loss.backward()
            optimizer.step()

            train_epoch_l1.append(torch.zeros(1))
            train_epoch_l2.append(torch.zeros(1))
            train_epoch_total.append(loss.detach())

        with torch.no_grad():
            for _X, _y in dl_val:
                model.eval()
                prediction = model(_X, use_decoder=False)
                _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))
                
                loss1 = 0
                loss2 = pred_loss(prediction, _y)
                loss = loss2

                val_epoch_l1.append(torch.zeros(1))
                val_epoch_l2.append(torch.zeros(1))
                val_epoch_total.append(loss)

        train_epoch_l1 = torch.stack(train_epoch_l1).mean()
        train_epoch_l2 = torch.stack(train_epoch_l2).mean()
        train_epoch_total = torch.stack(train_epoch_total).mean()
        val_epoch_l1 = torch.stack(val_epoch_l1).mean()
        val_epoch_l2 = torch.stack(val_epoch_l2).mean()
        val_epoch_total = torch.stack(val_epoch_total).mean()

        print(f'{epoch} \033[1m {train_epoch_total:.5f} \033[0m | \033[1m {val_epoch_total:.5f} \033[0m')
    
    torch.save(model.state_dict(), 'results/model_finetuned.pth')

    return model


def train_end_to_end(model, dl_train, dl_val):
    ### Hyperparameters
    learning_rate = 2e-3
    epochs = 50
    lagrange_pred_loss = 50
    use_decoder = False
    ###

    rec_loss = nn.MSELoss()
    pred_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        train_epoch_l1 = []
        train_epoch_l2 = []
        train_epoch_total = []
        val_epoch_l1 = []
        val_epoch_l2 = []
        val_epoch_total = []

        for _X, _y in dl_train:
            optimizer.zero_grad()
            if use_decoder:
                reconstructed, prediction = model(_X)
                _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))
                
                loss1 = rec_loss(reconstructed, _X)
                loss2 = pred_loss(prediction, _y)
                loss = loss1 + lagrange_pred_loss * loss2

                loss.backward()
                optimizer.step()

                train_epoch_l1.append(loss1.detach())
                train_epoch_l2.append(loss2.detach())
                train_epoch_total.append(loss.detach())
            else:
                prediction = model(_X, use_decoder=False)
                _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))
                
                loss1 = 0
                loss2 = pred_loss(prediction, _y)
                loss = loss2

                loss.backward()
                optimizer.step()

                train_epoch_l1.append(torch.zeros(1))
                train_epoch_l2.append(torch.zeros(1))
                train_epoch_total.append(loss.detach())

        with torch.no_grad():
            for _X, _y in dl_val:
                model.eval()
                if use_decoder:
                    reconstructed, prediction = model(_X)
                    _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))
                    
                    loss1 = rec_loss(reconstructed, _X)
                    loss2 = pred_loss(prediction, _y)
                    loss = loss1 + lagrange_pred_loss * loss2

                    val_epoch_l1.append(loss1)
                    val_epoch_l2.append(loss2)
                    val_epoch_total.append(loss)
                else:
                    prediction = model(_X, use_decoder=False)
                    _y = _y.reshape(-1, 1).unsqueeze(1).repeat((1, prediction.shape[1], 1))
                    
                    loss1 = 0
                    loss2 = pred_loss(prediction, _y)
                    loss = loss2

                    val_epoch_l1.append(torch.zeros(1))
                    val_epoch_l2.append(torch.zeros(1))
                    val_epoch_total.append(loss)

        train_epoch_l1 = torch.stack(train_epoch_l1).mean()
        train_epoch_l2 = torch.stack(train_epoch_l2).mean()
        train_epoch_total = torch.stack(train_epoch_total).mean()
        val_epoch_l1 = torch.stack(val_epoch_l1).mean()
        val_epoch_l2 = torch.stack(val_epoch_l2).mean()
        val_epoch_total = torch.stack(val_epoch_total).mean()

        print(f'{epoch} {train_epoch_l1:.5f} {train_epoch_l2:.5f} \033[1m {train_epoch_total:.5f} \033[0m | {val_epoch_l1:.5f} {val_epoch_l2:.5f} \033[1m {val_epoch_total:.5f} \033[0m')
    
    torch.save(model.state_dict(), 'results/model_e2e.pth')

    return model

def train_encoder_decoder(model, dl_train, dl_val):
    ### Hyperparameters
    n_channels = (32, 64)
    learning_rate = 2e-5
    epochs = 50
    ###

    rec_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        train_epoch_loss = []
        val_epoch_loss = []

        for _X, _ in dl_train:
            optimizer.zero_grad()
            reconstructed = model(_X)
            
            loss = rec_loss(reconstructed, _X)

            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.detach())

        # Eval
        with torch.no_grad():
            for _X, _ in dl_val:
                model.eval()
                reconstructed = model(_X)
                loss = rec_loss(reconstructed, _X)
                val_epoch_loss.append(loss)

        train_epoch_loss = torch.stack(train_epoch_loss).mean()
        val_epoch_loss = torch.stack(val_epoch_loss).mean()

        print(f'{epoch} {train_epoch_loss:.5f} | {val_epoch_loss:.5f}')
    
    torch.save(model.state_dict(), 'results/model_encdec.pth')

    return model

def main():
    X = load_monash('m4_daily')['series_value']
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    # Split (and maybe normalize) data into windows
    for x in X:
        _x = x.to_numpy()
        if normalize:
            mu, std = np.mean(_x), np.std(_x)
            _x = (_x - mu) / std
        _x, _y = windowing(_x, lag)

        if rng.binomial(1, 0.8):
            X_train.append(_x)
            y_train.append(_y)
        else:
            X_val.append(_x)
            y_val.append(_y)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Shrink amount of data (for testing)
    percentage = 0.01
    n_train = int(len(X_train) * percentage)
    n_val = int(len(X_val) * percentage)
    train_indices = rng.choice(X_train.shape[0], size=n_train, replace=False)
    val_indices = rng.choice(X_val.shape[0], size=n_val, replace=False)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    X_val = X_val[val_indices]
    y_val = y_val[val_indices]

    # Dataloader and other 
    ds_train = TensorDataset(torch.from_numpy(X_train).float().unsqueeze(1), torch.from_numpy(y_train).float())
    ds_val = TensorDataset(torch.from_numpy(X_val).float().unsqueeze(1), torch.from_numpy(y_val).float())
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # Train End2End
    n_channels = (32, 64)
    with fixedseed(torch, seed=1111111):
        model_e2e = MultiForecaster(EncoderDecoder(n_channels), n_channels[-1], lag)

    if exists('results/model_e2e.pth'):
        model_e2e.load_state_dict(torch.load('results/model_e2e.pth'))
    else:
        model_e2e = train_end_to_end(model_e2e, dl_train, dl_val)


    # Train encoder/decoder first, then finetune
    n_channels = (32, 64)
    with fixedseed(torch, seed=1111111):
        model_encdec = EncoderDecoder(n_channels)
    if exists('results/model_encdec.pth'):
        model_encdec.load_state_dict(torch.load('results/model_encdec.pth'))
    else:
        model_encdec = train_encoder_decoder(model_encdec, dl_train, dl_val)

    with fixedseed(torch, seed=1111111):
        model_finetuned = MultiForecaster(model_encdec, n_channels[-1], lag, train_encoder=False)
    if exists('results/model_finetuned.pth'):
        model_finetuned.load_state_dict(torch.load('results/model_finetuned.pth'))
    else:
        model_finetuned = train_finetuned(model_finetuned, dl_train, dl_val)

    loss = lambda a, b: mse(a, b, squared=False)

    # Get best possible loss (according to RMSE)
    model_train_loss = model_e2e.get_best_loss(X_train, y_train, loss)
    model_val_loss = model_e2e.get_best_loss(X_val, y_val, loss)
    print(f'Model e2e best: train rmse {model_train_loss}, val rmse {model_val_loss}')
    train_preds = model_e2e.predict(X_train)
    val_preds = model_e2e.predict(X_val)
    print(f'Model e2e ensemble: train rmse {loss(train_preds, y_train)}, val rmse {loss(val_preds, y_val)}')

    model_train_loss = model_finetuned.get_best_loss(X_train, y_train, loss)
    model_val_loss = model_finetuned.get_best_loss(X_val, y_val, loss)
    print(f'Model finetuned best: train rmse {model_train_loss}, val rmse {model_val_loss}')
    train_preds = model_finetuned.predict(X_train)
    val_preds = model_finetuned.predict(X_val)
    print(f'Model finetuned ensemble: train rmse {loss(train_preds, y_train)}, val rmse {loss(val_preds, y_val)}')

    # Comparison with RandomForest
    rf = RandomForestRegressor(n_estimators=128, max_depth=7, random_state=rng)
    rf.fit(X_train, y_train)
    train_preds = rf.predict(X_train)
    val_preds = rf.predict(X_val)

    print(f'RF Baseline: train rmse {loss(y_train, train_preds):.5f}, val rmse {loss(y_val, val_preds):.5f}')


if __name__ == '__main__':
    main()