# Copy Tree Classifier
# Author: Ronnie Crawford
# Created: 2024-06-10
# Purpose: Read trees generated from copy number of regions in the genome (reference Jack's programme),
# and use to classify cancerous and non-cancerous.

import os

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

TREE_DIRECTORIES_PATH = 

def main():

    regions_df = read_directory(TREE_DIRECTORIES_PATH)

def read_directory(directory_path : str) -> input_trees_df : pd.DataFrame:
    
    """
    Reads all CSV files from the directory, processes them, and returns a concatenated DataFrame.
    
    Args:
    directory_path (str): Path to the directory containing the CSV files.
    
    Returns:
    input_trees_df (pd.DataFrame): Concatenated DataFrame of all processed files.
    """
    
    tree_files = [subdirectory_path.direct_to_tree_file() for subdirectory_path in os.listdir(directory_path)]
    
    with ThreadPoolExecutor() as executor:
        
        # Process files in parallel
        results = list(executor.map(process_file, tree_files))
    
    # Concatenate all processed DataFrames
    regions_df = pd.concat(results, ignore_index=True)
    
    return regions_df

def direct_to_tree_file(directory_path : str) -> tree_file_path : str:
    
    return directory_path

def read_file(file_path : str) -> tree_df : pd.DataFrame:
    
    """
    Reads a CSV file, processes it, and returns a DataFrame with additional columns.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    tree_df (pd.DataFrame): Processed DataFrame with regions (segments).
    """
    
    assert file_path.endswith(".tree")
    
    # Read the CSV file
    new_tree_df = pd.read_csv(file_path, sep='\t', header=11, names=['chromosome', 'start_position', 'copy_number'])
    new_tree_df['sample'] = os.path.basename(file_path)
    
    # Process each chromosome separately
    chromosome_segments = []
    for _, chromosome_df in new_tree_df.groupby('chromosome'):
        
        chromosome_df['end_position'] = chromosome_df['start_position'].shift(-1)
        chromosome_df = chromosome_df[:-1]  # Remove the last row as it has NaN in 'end_position'
        chromosome_df['region_length'] = chromosome_df['end_position'] - chromosome_df['start_position']
        chromosome_segments.append(chromosome_df)
    
    # Concatenate all chromosome segments
    tree_df = pd.concat(chromosome_segments, ignore_index=True)
    
    return tree_df



