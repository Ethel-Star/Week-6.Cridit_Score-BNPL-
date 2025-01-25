import pandas as pd
import os

def load_data(csv_file_path):
    """
    Loads the dataset from a CSV file.
    
    Args:
        csv_file_path (str): The path to the CSV file.
        
    Returns:
        pandas.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"The file {csv_file_path} was not found.")
    
    # Load the data into a pandas DataFrame
    data = pd.read_csv(csv_file_path)
    
    # Print basic information about the data
    print("Data Loaded Successfully!")
    print(f"Shape of the dataset: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Display the first few rows of the data
    print("\nFirst few rows of the data:")
    #data.head() # Display first 5 rows of the dataset
    
    return data
