import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data(file_path):
    """
    Load data from a file.

    Parameters:
    file_path (str): The path to the file to load. Supported formats are .xlsx, .csv, and .h5.

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

def preprocess_data(data, input_columns, output_columns, test_size=0.2, random_state=42):
    """
    Preprocess the data by splitting into training and test sets and scaling.

    Parameters:
    data (pd.DataFrame): The data to preprocess.
    input_columns (list): List of input column names.
    output_columns (list): List of output column names.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Scaled training and test data, and the scalers used for X and y.
    """
    # Select the columns based on the second level of the multi-level columns
    X = data.loc[:, input_columns].values
    y = data.loc[:, output_columns].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def create_tensors(X_train, X_test, y_train, y_test):
    """
    Convert numpy arrays to PyTorch tensors.

    Parameters:
    X_train (np.ndarray): Training input data.
    X_test (np.ndarray): Test input data.
    y_train (np.ndarray): Training output data.
    y_test (np.ndarray): Test output data.

    Returns:
    tuple: PyTorch tensors for training and test data.
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with two hidden layers.

    Parameters:
    input_size (int): The number of input features.
    output_size (int): The number of output features.
    """
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, num_epochs=100):
    """
    Train the neural network model.

    Parameters:
    model (nn.Module): The neural network model.
    criterion (nn.Module): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    X_train_tensor (torch.Tensor): Training input data.
    y_train_tensor (torch.Tensor): Training output data.
    num_epochs (int): Number of epochs to train the model.
    """
    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, criterion, X_test_tensor, y_test_tensor, scaler_y, output_columns):
    """
    Evaluate the neural network model on the test data.

    Parameters:
    model (nn.Module): The neural network model.
    criterion (nn.Module): The loss function.
    X_test_tensor (torch.Tensor): Test input data.
    y_test_tensor (torch.Tensor): Test output data.
    scaler_y (StandardScaler): Scaler used for the output data.
    output_columns (list): List of output column names.
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
def run_neural_network(file_path, input_columns, output_columns, num_epochs=100):
    """
    Run the complete neural network pipeline: load data, preprocess, train, and evaluate.

    Parameters:
    file_path (str): The path to the data file.
    input_columns (list): List of input column names.
    output_columns (list): List of output column names.
    num_epochs (int): Number of epochs to train the model.
    """
    data = load_data(file_path)

    # Debug: Print the columns of the DataFrame
    print("DataFrame columns:", data.columns)

    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(data, input_columns, output_columns)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = create_tensors(X_train, X_test, y_train, y_test)

    input_size = X_train_tensor.shape[1]
    output_size = y_train_tensor.shape[1]
    model = NeuralNetwork(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, num_epochs)
    evaluate_model(model, criterion, X_test_tensor, y_test_tensor, scaler_y, output_columns)