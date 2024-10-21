from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class PricePredictor:
    def __init__(self, lookback=60, prediction_horizon=1, epochs=100, batch_size=32, learning_rate=0.001, hidden_dim=100, num_layers=2, val_split=0.2, optimizer='adam', loss_fn='mse'):
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.val_split = val_split
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def prepare_data(self, data):
        close_data = data[['close']].values
        scaled_data = self.scaler.fit_transform(close_data)

        X = []
        y = []
        for i in range(self.lookback, len(scaled_data) - self.prediction_horizon + 1):
            X.append(scaled_data[i - self.lookback:i])
            y.append(scaled_data[i + self.prediction_horizon - 1, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = LSTMModel(input_dim=input_shape[2], hidden_dim=self.hidden_dim, num_layers=self.num_layers, output_dim=1)
        return model.to(self.device)

    def train(self, data):
        X, y = self.prepare_data(data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, shuffle=False)

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self.build_model(X_train.shape)

        # Select optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Select loss function
        if self.loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif self.loss_fn == 'mae':
            criterion = nn.L1Loss()
        elif self.loss_fn == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            self.model.train()
            # Train on batches
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                # Evaluate on validation set
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val)

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss.item())

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss.item():.4f}')

        return history

    def predict(self, data):
        close_data = data[['close']].values
        scaled_data = self.scaler.transform(close_data)

        X = []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
        X = np.array(X)

        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)

        return self.scaler.inverse_transform(predictions.cpu().numpy())

    def evaluate(self, data):
        X, y = self.prepare_data(data)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)

        mse = nn.MSELoss()(y_pred.squeeze(), y)
        return mse.item()
