"""
Data Processing Module for Impulse Control Predictor
Handles dataset loading, exploration, and initial data cleaning
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Handles all data loading and preprocessing operations
    """
    
    def __init__(self, file_path: str):
        """
        Initialize DataProcessor with dataset path
        
        Args:
            file_path (str): Path to the CSV dataset
        """
        self.file_path = file_path
        self.df = None
        self.original_df = None
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading dataset from: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        self.original_df = self.df.copy()
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        return self.df
    
    def display_summary_statistics(self) -> dict:
        """
        Display comprehensive summary statistics
        
        Returns:
            dict: Summary statistics dictionary
        """
        print("\n" + "="*80)
        print("DATASET SUMMARY STATISTICS")
        print("="*80)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"  - Rows: {self.df.shape[0]}")
        print(f"  - Columns: {self.df.shape[1]}")
        
        print("\n" + "-"*80)
        print("DATA TYPES:")
        print("-"*80)
        print(self.df.dtypes)
        
        print("\n" + "-"*80)
        print("FIRST FEW ROWS:")
        print("-"*80)
        print(self.df.head())
        
        print("\n" + "-"*80)
        print("STATISTICAL SUMMARY:")
        print("-"*80)
        print(self.df.describe())
        
        return {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.to_dict(),
            'describe': self.df.describe().to_dict()
        }
    
    def check_missing_values(self) -> pd.DataFrame:
        """
        Check and report missing values in the dataset
        
        Returns:
            pd.DataFrame: DataFrame with missing value statistics
        """
        print("\n" + "="*80)
        print("MISSING VALUES ANALYSIS")
        print("="*80)
        
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        if len(missing_data) == 0:
            print("✓ No missing values found in the dataset!")
        else:
            print(missing_data.to_string(index=False))
        
        return missing_data
    
    def handle_missing_values(self, strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing values using specified strategy
        
        Args:
            strategy (str): 'smart' (auto-detection), 'mean', 'median', 'drop', 'forward_fill'
        
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)
        
        if strategy == 'smart':
            print("Using smart strategy (auto-detection based on column type)...")
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if self.df[col].dtype in ['float64', 'int64']:
                        # Use median for numerical columns
                        median_val = self.df[col].median()
                        self.df[col].fillna(median_val, inplace=True)
                        print(f"  ✓ {col}: Filled {self.df[col].isnull().sum()} nulls with median ({median_val:.2f})")
                    else:
                        # Use mode for categorical columns
                        mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                        self.df[col].fillna(mode_val, inplace=True)
                        print(f"  ✓ {col}: Filled with mode ({mode_val})")
        
        elif strategy == 'mean':
            print("Filling numerical columns with mean...")
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numerical_cols] = self.df[numerical_cols].fillna(self.df[numerical_cols].mean())
        
        elif strategy == 'median':
            print("Filling numerical columns with median...")
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numerical_cols] = self.df[numerical_cols].fillna(self.df[numerical_cols].median())
        
        elif strategy == 'drop':
            print("Dropping rows with missing values...")
            self.df = self.df.dropna()
        
        print(f"\n✓ Missing values handled successfully!")
        print(f"  Remaining missing values: {self.df.isnull().sum().sum()}")
        
        return self.df
    
    def get_data_type_info(self) -> dict:
        """
        Get information about data types and columns
        
        Returns:
            dict: Information about numerical and categorical columns
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print("\n" + "="*80)
        print("COLUMN INFORMATION")
        print("="*80)
        print(f"\nNumerical Columns ({len(numerical_cols)}):")
        print(f"  {numerical_cols}")
        
        print(f"\nCategorical Columns ({len(categorical_cols)}):")
        print(f"  {categorical_cols}")
        
        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols,
            'total': len(numerical_cols) + len(categorical_cols)
        }
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned dataset
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        return self.df.copy()
    
    def get_original_data(self) -> pd.DataFrame:
        """
        Get the original dataset before any processing
        
        Returns:
            pd.DataFrame: Original dataset
        """
        return self.original_df.copy()


def create_sample_dataset(file_path: str, n_samples: int = 1000) -> None:
    """
    Create a sample dataset for testing (in case the actual dataset doesn't exist)
    
    Args:
        file_path (str): Path to save the sample dataset
        n_samples (int): Number of samples to generate
    """
    print(f"Generating sample dataset with {n_samples} records...")
    
    np.random.seed(42)
    
    data = {
        'customer_id': np.arange(1, n_samples + 1),
        'total_time_spent': np.random.uniform(30, 600, n_samples),  # seconds
        'number_of_clicks': np.random.randint(1, 50, n_samples),
        'add_to_cart_time': np.random.uniform(0.5, 30, n_samples),  # minutes
        'discount_percentage': np.random.uniform(0, 70, n_samples),
        'purchase_flag': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'purchase_time': np.random.randint(0, 24, n_samples),  # hour of day
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.5, 0.35, 0.15]),
        'previous_purchases': np.random.randint(0, 30, n_samples),
        'customer_age': np.random.randint(18, 70, n_samples),
        'product_category': np.random.choice(['Electronics', 'Fashion', 'Home', 'Sports', 'Beauty'], n_samples),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"✓ Sample dataset created at: {file_path}")
