# Copy Tree Classifier
# Author: Ronnie Crawford
# Created: 2024-05-31
# Purpose: Read trees generated from copy number of regions in the genome (reference Jack's programme),
# and use to classify cancerous and non-cancerous.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define major variables
TARGET_LENGTH = 1000    # The target length to which all sequences will be resized
BATCH_SIZE = 16         # Number of samples per batch for training
LEARNING_RATE = 0.001   # Learning rate for the optimiser
NUM_EPOCHS = 25         # Number of training epochs
KERNEL_SIZE = 3         # Size of the kernel for convolutional layers
STRIDE = 1              # Stride size for convolutional layers
PADDING = 1             # Padding size for convolutional layers
POOL_KERNEL_SIZE = 2    # Size of the kernel for pooling layers
POOL_STRIDE = 2         # Stride size for pooling layers

# Define model architecture variables
NUM_CHANNELS = 16       # Number of channels for all convolutional layers
FC1_NODES = 128         # Number of nodes in the first fully connected layer
NUM_CONV_LAYERS = 2     # Number of convolutional layers

# Define paths
TRAIN_CANCEROUS_DIR = 'path_to_train_cancerous'         # Directory containing training data for cancerous samples
TRAIN_NON_CANCEROUS_DIR = 'path_to_train_non_cancerous' # Directory containing training data for non-cancerous samples
TEST_CANCEROUS_DIR = 'path_to_test_cancerous'           # Directory containing test data for cancerous samples
TEST_NON_CANCEROUS_DIR = 'path_to_test_non_cancerous'   # Directory containing test data for non-cancerous samples

# Check for available device
if torch.cuda.is_available(): DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): DEVICE = torch.device('mps')
else: DEVICE = torch.device('cpu')

def main():
    
    # Load the datasets
    train_dirs = {1: TRAIN_CANCEROUS_DIR, 0: TRAIN_NON_CANCEROUS_DIR}
    test_dirs = {1: TEST_CANCEROUS_DIR, 0: TEST_NON_CANCEROUS_DIR}
    train_dataset = GenomicDataset(train_dirs, TARGET_LENGTH)
    test_dataset = GenomicDataset(test_dirs, TARGET_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataloaders = {'train': train_dataloader, 'test': test_dataloader}

    # Initialize the model, loss function and optimiser
    model = GenomicCNN()
    loss_function = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, dataloaders, loss_function, optimiser, NUM_EPOCHS)

    # Save the model
    torch.save(model.state_dict(), 'genomic_cnn.pth')
    
    # Evaluate the model
    accuracy = evaluate_model(model, test_dataloader)
    print(f'Accuracy on test dataset: {accuracy:.2f}')

# Define the dataset class
class GenomicDataset(Dataset):

    def __init__(self, data_dirs, target_length=TARGET_LENGTH):
        
        """
        Initialize the dataset by loading the data from directories.

        :arg data_dirs: Dictionary with directory paths and their corresponding labels.
        :arg target_length: The target length to which all sequences will be resized.
        """
        
        self.target_length = target_length
        self.samples = []
        self.labels = []

        for label, data_dir in data_dirs.items():
            for file_name in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file_name)
                data = self.load_data(file_path)
                self.samples.extend(data)
                self.labels.extend([label] * len(data))

    def load_data(self, file_path):
        
        """
        Load the genomic data from a file.

        :arg file_path: Path to the data file.
        :return: List of sequences.
        """
        
        data = pd.read_csv(file_path, sep=' ', header=None, skiprows=6, names=['chr', 'pos', 'copy_number'])
        sequences = []
        current_chromosome = data.iloc[0, 0]
        current_sample = []
        current_pos = 0
        
        for i in range(len(data)):
            start_pos = current_pos
            end_pos = data.iloc[i, 1]
            copy_number = data.iloc[i, 2]
            
            # Fill in the copy number for all positions between start_pos and end_pos
            current_sample.extend([copy_number] * (end_pos - start_pos))
            current_pos = end_pos
        
        sequences.append(self.resize_sequence(current_sample))
        return sequences

    def resize_sequence(self, sequence):
        
        """
        Resize the sequence to the target length using linear interpolation.

        :param sequence: The original sequence.
        :return: Resized sequence.
        """
        
        sequence = np.array(sequence)
        original_length = len(sequence)
        if original_length == self.target_length:
            return sequence
        resized_sequence = np.interp(
            np.linspace(0, original_length - 1, self.target_length),
            np.arange(original_length),
            sequence
        )
        return resized_sequence

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

class GenomicCNN(nn.Module):

    def __init__(self):
        
        super(GenomicCNN, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=NUM_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.pool = nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE, padding=0)
        self.conv2 = nn.Conv1d(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(NUM_CHANNELS, FC1_NODES)  # Adjust input size to match output from global_pool
        self.fc2 = nn.Linear(FC1_NODES, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # Define the forward pass
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(-1, NUM_CHANNELS)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Training function
def train_model(model, dataloaders, loss_function, optimiser, num_epochs=NUM_EPOCHS):
    
    """
    Train the model.

    :param model: The CNN model.
    :param dataloaders: Dictionary containing training and validation dataloaders.
    :param loss_function: Method to work out loss.
    :param optimiser: Optimization algorithm.
    :param num_epochs: Number of training epochs.
    """
    
    model.to(DEVICE)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            labels = labels.unsqueeze(1)  # Adjust labels dimension
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloaders["train"])}')


# Inference function
def evaluate_model(model, dataloader):
    
    """
    Evaluate the model on the test dataset and calculate accuracy.

    :param model: The trained CNN model.
    :param dataloader: DataLoader for the test dataset.
    :return: Accuracy of the model on the test dataset.
    """
    
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    
    accuracy = correct / total
    return accuracy

# Execute the main function
if __name__ == '__main__':
    main()
