import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import configparser
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

def load_data(file_path):
    """
    Loads data from a file.

    Parameters:
    file_path (str): The path to the data file. Supported formats are .xlsx, .csv, and .h5.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
    ValueError: If the file format is not supported.
    """
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path, header=[0, 1])
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path, header=[0, 1])
    elif file_path.endswith('.h5'):
        data = pd.read_hdf(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx, .csv, or .h5")
    return data

def preprocess_data(data, input_columns, output_columns, test_size=0.2, random_state=42, augmentation_noise=0.0):
    """
    Preprocesses the data by splitting into training and test sets, scaling, and optionally adding noise.

    Parameters:
    data (pd.DataFrame): The input data.
    input_columns (list): List of input column names.
    output_columns (list): List of output column names.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    augmentation_noise (float): Standard deviation of Gaussian noise to add to the training data.

    Returns:
    tuple: Scaled and split data (X_train, X_test, y_train, y_test, scaler_X, scaler_y).
    """
    X = data.loc[:, input_columns].values
    y = data.loc[:, output_columns].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    if augmentation_noise > 0:
        noise = augmentation_noise * np.random.randn(*X_train.shape)
        X_train += noise

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def create_tensors(X_train, X_test, y_train, y_test):
    """
    Converts numpy arrays to PyTorch tensors.

    Parameters:
    X_train (np.ndarray): Training input data.
    X_test (np.ndarray): Test input data.
    y_train (np.ndarray): Training output data.
    y_test (np.ndarray): Test output data.

    Returns:
    tuple: PyTorch tensors (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor).
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

class NeuralNetwork(nn.Module):
    """
    A neural network model with configurable hidden layers, activation functions, and dropout.

    Parameters:
    input_size (int): Number of input features.
    output_size (int): Number of output features.
    hidden_layer_sizes (list): List of sizes for hidden layers.
    activation_function (str): Activation function to use ('ReLU', 'Sigmoid', 'Tanh').
    dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation_function, dropout_rate):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size

        # Add hidden layers dynamically
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))  # Linear transformation
            layers.append(nn.BatchNorm1d(size))       # Batch normalization
            if activation_function == 'ReLU':         # Activation function
                layers.append(nn.ReLU())
            elif activation_function == 'Sigmoid':
                layers.append(nn.Sigmoid())
            elif activation_function == 'Tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))   # Dropout for reducing overfitting
            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        return self.network(x)


def train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, num_epochs, use_scheduler,
                scheduler_step_size, scheduler_gamma, early_stopping, patience, scheduler_type):
    """
    Trains the neural network model.

    Parameters:
    model (nn.Module): The neural network model.
    criterion (nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    X_train_tensor (torch.Tensor): Training input data.
    y_train_tensor (torch.Tensor): Training output data.
    num_epochs (int): Number of training epochs.
    use_scheduler (bool): Whether to use a learning rate scheduler.
    scheduler_step_size (int): Step size for the scheduler.
    scheduler_gamma (float): Gamma value for the scheduler.
    early_stopping (bool): Whether to use early stopping.
    patience (int): Number of epochs to wait for improvement before stopping.
    scheduler_type (str): Type of scheduler to use ('StepLR', 'CosineAnnealingLR').

    Returns:
    None
    """
    if scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_scheduler:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        if early_stopping:
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

def evaluate_model(model, criterion, X_test_tensor, y_test_tensor, scaler_y, output_columns, metrics):
    """
    Evaluates the neural network model on the test data.

    Parameters:
    model (nn.Module): The neural network model.
    criterion (nn.Module): Loss function.
    X_test_tensor (torch.Tensor): Test input data.
    y_test_tensor (torch.Tensor): Test output data.
    scaler_y (StandardScaler): Scaler for the output data.
    output_columns (list): List of output column names.
    metrics (list): List of metrics to evaluate ('MSE', 'MAE', 'R2').

    Returns:
    None
    """
    model.eval()
    y_test_pred = model(X_test_tensor)
    test_loss = criterion(y_test_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

    y_test_pred = scaler_y.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler_y.inverse_transform(y_test_tensor.numpy())

    for i, col in enumerate(output_columns):
        print(f'{col}:')
        print(f'  Tatsächlich: {y_test[0, i]:.2f}, Vorhergesagt: {y_test_pred[0, i]:.2f}')

    if 'MSE' in metrics:
        mse = mean_squared_error(y_test, y_test_pred)
        print(f'MSE: {mse:.4f}')
    if 'MAE' in metrics:
        mae = mean_absolute_error(y_test, y_test_pred)
        print(f'MAE: {mae:.4f}')
    if 'R2' in metrics:
        r2 = r2_score(y_test, y_test_pred)
        print(f'R2: {r2:.4f}')

        # Plot the predictions if enabled in the config
        plot_predictions(y_test, y_test_pred, output_columns, config)
def plot_predictions(y_test, y_pred, output_columns, config):
    """
    Plots the actual vs. predicted values for each output variable.

    Parameters:
    y_test (np.ndarray): The actual test values (ground truth).
    y_pred (np.ndarray): The predicted values from the model.
    output_columns (list): List of output column names.
    config (ConfigParser): ConfigParser object to read the configuration.

    Returns:
    None
    """
    # Check if plotting is enabled in the config
    if not config.getboolean('Visualization', 'EnablePlot', fallback=False):
        print("Plotting is disabled in the configuration.")
        return

    # Create subplots for each output column
    num_outputs = len(output_columns)
    plt.figure(figsize=(12, 6))

    for i, col in enumerate(output_columns):
        plt.subplot(1, num_outputs, i + 1)
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, label='Vorhersage')
        plt.plot([y_test[:, i].min(), y_test[:, i].max()],
                 [y_test[:, i].min(), y_test[:, i].max()], 'r--', label='Ideal')
        plt.title(f'{col}\nTatsächlich vs. Vorhergesagt')
        plt.xlabel('Tatsächlich')
        plt.ylabel('Vorhergesagt')
        plt.legend()

    plt.tight_layout()
    plt.show()


def run_neural_network(file_path, input_columns, output_columns, mode=None):
    """
    Runs the neural network training and evaluation pipeline.

    Parameters:
    file_path (str): Path to the data file.
    input_columns (list): List of input column names.
    output_columns (list): List of output column names.
    mode (str, optional): Mode for configuration selection. Defaults to None.

    Returns:
    None
    """
    if mode == None:
        conf = "NeuralNetworkDefault"
    elif mode == "OedometerTest":
        conf = "NeuralNetworkOedometerTest"

    num_epochs = config.getint(conf, 'NumEpochs')
    learning_rate = config.getfloat(conf, 'LearningRate')
    batch_size = config.getint(conf, 'BatchSize')
    use_scheduler = config.getboolean(conf, 'UseScheduler')
    scheduler_step_size = config.getint(conf, 'SchedulerStepSize')
    scheduler_gamma = config.getfloat(conf, 'SchedulerGamma')
    hidden_layer_sizes = [int(size) for size in config.get(conf, 'HiddenLayerSizes').split(',')]
    activation_function = config.get(conf, 'ActivationFunction')
    dropout_rate = config.getfloat(conf, 'DropoutRate')
    early_stopping = config.getboolean(conf, 'EarlyStopping')
    patience = config.getint(conf, 'Patience')
    augmentation_noise = config.getfloat(conf, 'AugmentationNoise')
    metrics = config.get(conf, 'Metrics').split(',')
    scheduler_type = config.get(conf, 'SchedulerType')

    data = load_data(file_path)
    if data is None:
        raise ValueError("Data could not be loaded. Please check the file path and format.")

    print("Data loaded successfully:", data.head())  # Debugging line

    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(data, input_columns, output_columns,
                                                                           augmentation_noise=augmentation_noise)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = create_tensors(X_train, X_test, y_train, y_test)

    input_size = X_train_tensor.shape[1]
    output_size = y_train_tensor.shape[1]
    model = NeuralNetwork(input_size, output_size, hidden_layer_sizes, activation_function, dropout_rate)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, num_epochs, use_scheduler,
                scheduler_step_size, scheduler_gamma, early_stopping, patience, scheduler_type)
    evaluate_model(model, criterion, X_test_tensor, y_test_tensor, scaler_y, output_columns, metrics)