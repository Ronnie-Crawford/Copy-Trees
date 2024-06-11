import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# Define major variables
TARGET_LENGTH = 1000    # The target length to which all sequences will be resized
BATCH_SIZE = 16         # Number of samples per batch for training
LEARNING_RATE = 0.001   # Learning rate for the optimizer
NUM_EPOCHS = 25         # Number of training epochs
HIDDEN_SIZE = 64        # Number of hidden units in the LSTM
NUM_LAYERS = 2          # Number of layers in the LSTM

# Define paths
TRAIN_CANCEROUS_DIR = 'path_to_train_cancerous'         # Directory containing training data for cancerous samples
TRAIN_NON_CANCEROUS_DIR = 'path_to_train_non_cancerous' # Directory containing training data for non-cancerous samples
TEST_CANCEROUS_DIR = 'path_to_test_cancerous'           # Directory containing test data for cancerous samples
TEST_NON_CANCEROUS_DIR = 'path_to_test_non_cancerous'   # Directory containing test data for non-cancerous samples

# Check for available device
if torch.cuda.is_available():
    
    DEVICE = torch.device('cuda')
    print("Using GPU")
    
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    
    DEVICE = torch.device('mps')
    print("Using MPS")
    
else:
    
    DEVICE = torch.device('cpu')
    print("Using CPU")


# Define the dataset class
class GenomicDataset(Dataset):

    def __init__(self, data_dirs):
        
        """
        Initialize the dataset by loading the data from directories.

        :param data_dirs: Dictionary with directory paths and their corresponding labels.
        """
        
        self.samples = []
        self.labels = []

        for label, data_dir in data_dirs.items():
            for file_name in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file_name)
                data = self.load_data(file_path)
                self.samples.append(data)
                self.labels.append(label)

    def load_data(self, file_path):
        
        """
        Load the genomic data from a file.

        :param file_path: Path to the data file.
        :return: Encoded sequence data.
        """
        
        data = pd.read_csv(file_path, sep=' ', header=None, skiprows=6, names=['chr', 'pos', 'copy_number'])
        sequences = []

        for chromosome in data['chr'].unique():
            
            chrom_data = data[data['chr'] == chromosome]
            sequences.extend(self.process_chromosome_data(chrom_data))

        return np.array(sequences)

    def process_chromosome_data(self, chrom_data):
        
        """
        Process the data for a single chromosome.

        :param chrom_data: DataFrame containing the chromosome data.
        :return: List of encoded sequence data for the chromosome.
        """
        
        positions = chrom_data['pos'].values
        copy_numbers = chrom_data['copy_number'].values
        copy_diff = np.diff(copy_numbers, prepend=0)  # Calculate the differences in copy number
        sequences = []

        for i in range(len(positions)):
            
            sequences.append([chrom_data['chr'].values[i], positions[i], copy_diff[i], copy_numbers[i]])

        return sequences

    def __len__(self):
        
        """
        Return the number of samples in the dataset.
        """
        
        return len(self.samples)

    def __getitem__(self, idx):
        
        """
        Get a sample and its label from the dataset.

        :param idx: Index of the sample to retrieve.
        :return: Tuple of (sample, label).
        """
        
        sample = torch.tensor(self.samples[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label

class GenomicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        
        super(GenomicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        """
        Forward pass of the LSTM model.

        :param x: Input tensor of shape (batch_size, sequence_length, input_size).
        :return: Output tensor with the predicted probabilities.
        """
        
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(DEVICE)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS):
    
    """
    Train the model.

    :param model: The LSTM model.
    :param dataloaders: Dictionary containing training and validation dataloaders.
    :param criterion: Loss function.
    :param optimizer: Optimization algorithm.
    :param num_epochs: Number of training epochs.
    """
    
    model.to(DEVICE)
    
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloaders['train']:
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloaders["train"])}')


def evaluate_model(model, dataloader):
    
    """
    Evaluate the model on the test dataset and calculate accuracy.

    :param model: The trained LSTM model.
    :param dataloader: DataLoader for the test dataset.
    :return: Accuracy of the model on the test dataset.
    """
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    
    accuracy = correct / total
    return accuracy


def load_datasets():
    
    """
    Load the training and testing datasets.

    :return: Tuple containing the training and testing dataloaders.
    """
    
    train_dirs = {1: TRAIN_CANCEROUS_DIR, 0: TRAIN_NON_CANCEROUS_DIR}
    test_dirs = {1: TEST_CANCEROUS_DIR, 0: TEST_NON_CANCEROUS_DIR}
    train_dataset = GenomicDataset(train_dirs)
    test_dataset = GenomicDataset(test_dirs)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return {'train': train_dataloader, 'test': test_dataloader}


def initialize_model(input_size):
    
    """
    Initialize the LSTM model, criterion, and optimizer.

    :param input_size: Number of features in the input data.
    :return: Tuple containing the model, criterion, and optimizer.
    """
    
    model = GenomicLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, criterion, optimizer


def main():
    
    """
    Main function to execute the script.
    """
    
    dataloaders = load_datasets()
    input_size = 4  # Features: chromosome, position, copy number difference, absolute copy number
    model, criterion, optimizer = initialize_model(input_size)

    # Train the model
    train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS)

    # Save the model
    torch.save(model.state_dict(), 'genomic_lstm.pth')
    
    # Evaluate the model
    accuracy = evaluate_model(model, dataloaders['test'])
    print(f'Accuracy on test dataset: {accuracy:.2f}')


# Execute the main function
if __name__ == '__main__':
    main()
