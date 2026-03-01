"""
Feature Engineering Module for Impulse Control Predictor
Handles creation of impulse purchase indicators and features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class ImpulseFeatureEngineer:
    """
    Engineer features for impulse purchase prediction
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer with dataset
        
        Args:
            df (pd.DataFrame): Input dataset
        """
        self.df = df.copy()
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        
    def create_session_speed(self) -> pd.DataFrame:
        """
        Create session_speed = total_time_spent / number_of_clicks
        Lower speed indicates quick browsing (impulse buying indicator)
        
        Returns:
            pd.DataFrame: Dataset with session_speed feature
        """
        print("\n" + "-"*80)
        print("FEATURE 1: Session Speed")
        print("-"*80)
        
        # Handle division by zero
        self.df['session_speed'] = np.where(
            self.df['number_of_clicks'] > 0,
            self.df['total_time_spent'] / self.df['number_of_clicks'],
            0
        )
        
        print(f"✓ session_speed created")
        print(f"  Mean: {self.df['session_speed'].mean():.2f} seconds/click")
        print(f"  Min: {self.df['session_speed'].min():.2f} | Max: {self.df['session_speed'].max():.2f}")
        
        self.feature_columns.append('session_speed')
        return self.df
    
    def create_urgency_score(self, threshold_minutes: float = 2) -> pd.DataFrame:
        """
        Create urgency_score = 1 if add_to_cart_time < threshold_minutes else 0
        Quick add-to-cart action indicates impulse behavior
        
        Args:
            threshold_minutes (float): Threshold for quick action (default: 2 minutes)
        
        Returns:
            pd.DataFrame: Dataset with urgency_score feature
        """
        print("\n" + "-"*80)
        print(f"FEATURE 2: Urgency Score (threshold: {threshold_minutes} min)")
        print("-"*80)
        
        self.df['urgency_score'] = (self.df['add_to_cart_time'] < threshold_minutes).astype(int)
        
        urgency_count = self.df['urgency_score'].sum()
        print(f"✓ urgency_score created")
        print(f"  High urgency purchases: {urgency_count} ({urgency_count/len(self.df)*100:.1f}%)")
        
        self.feature_columns.append('urgency_score')
        return self.df
    
    def create_discount_sensitivity(self) -> pd.DataFrame:
        """
        Create discount_sensitivity = discount_percentage where purchase_flag == 1
        High discount sensitivity indicates impulse behavior
        
        Returns:
            pd.DataFrame: Dataset with discount_sensitivity feature
        """
        print("\n" + "-"*80)
        print("FEATURE 3: Discount Sensitivity")
        print("-"*80)
        
        self.df['discount_sensitivity'] = np.where(
            self.df['purchase_flag'] == 1,
            self.df['discount_percentage'],
            0
        )
        
        avg_discount = self.df[self.df['purchase_flag'] == 1]['discount_percentage'].mean()
        print(f"✓ discount_sensitivity created")
        print(f"  Average discount for purchasers: {avg_discount:.2f}%")
        print(f"  Mean discount sensitivity: {self.df['discount_sensitivity'].mean():.2f}%")
        
        self.feature_columns.append('discount_sensitivity')
        return self.df
    
    def create_night_purchase_flag(self, start_hour: int = 22, end_hour: int = 5) -> pd.DataFrame:
        """
        Create night_purchase_flag = 1 if purchase_time between start_hour and end_hour
        Night purchases often indicate less planned/impulse buying
        
        Args:
            start_hour (int): Start hour for night (default: 22)
            end_hour (int): End hour for night (default: 5)
        
        Returns:
            pd.DataFrame: Dataset with night_purchase_flag feature
        """
        print("\n" + "-"*80)
        print(f"FEATURE 4: Night Purchase Flag (between {start_hour}h and {end_hour}h)")
        print("-"*80)
        
        self.df['night_purchase_flag'] = np.where(
            (self.df['purchase_time'] >= start_hour) | (self.df['purchase_time'] < end_hour),
            1,
            0
        )
        
        night_count = self.df['night_purchase_flag'].sum()
        print(f"✓ night_purchase_flag created")
        print(f"  Night purchases: {night_count} ({night_count/len(self.df)*100:.1f}%)")
        
        self.feature_columns.append('night_purchase_flag')
        return self.df
    
    def create_mobile_user_flag(self) -> pd.DataFrame:
        """
        Create mobile_user_flag = 1 if device_type == "Mobile"
        Mobile users tend to make more impulse purchases
        
        Returns:
            pd.DataFrame: Dataset with mobile_user_flag feature
        """
        print("\n" + "-"*80)
        print("FEATURE 5: Mobile User Flag")
        print("-"*80)
        
        self.df['mobile_user_flag'] = (self.df['device_type'] == 'Mobile').astype(int)
        
        mobile_count = self.df['mobile_user_flag'].sum()
        print(f"✓ mobile_user_flag created")
        print(f"  Mobile users: {mobile_count} ({mobile_count/len(self.df)*100:.1f}%)")
        
        self.feature_columns.append('mobile_user_flag')
        return self.df
    
    def normalize_features(self, features_to_normalize: list) -> pd.DataFrame:
        """
        Normalize specified features to 0-1 range using MinMaxScaler
        
        Args:
            features_to_normalize (list): List of feature names to normalize
        
        Returns:
            pd.DataFrame: Dataset with normalized features
        """
        print("\n" + "-"*80)
        print("NORMALIZING FEATURES")
        print("-"*80)
        
        for feature in features_to_normalize:
            if feature in self.df.columns:
                # Reshape for sklearn
                feature_array = self.df[feature].values.reshape(-1, 1)
                self.df[f'{feature}_normalized'] = self.scaler.fit_transform(feature_array)
                print(f"✓ {feature} normalized to [0, 1]")
        
        return self.df
    
    def create_impulse_control_index(self,
                                     session_speed_weight: float = 0.3,
                                     discount_weight: float = 0.3,
                                     urgency_weight: float = 0.2,
                                     night_weight: float = 0.2) -> pd.DataFrame:
        """
        Create Impulse Control Index (ICI) - composite score
        
        ICI = w1*normalized_session_speed + w2*normalized_discount + 
              w3*urgency_score + w4*night_purchase_flag
        
        Args:
            session_speed_weight (float): Weight for session speed (default: 0.3)
            discount_weight (float): Weight for discount (default: 0.3)
            urgency_weight (float): Weight for urgency (default: 0.2)
            night_weight (float): Weight for night purchase (default: 0.2)
        
        Returns:
            pd.DataFrame: Dataset with Impulse Control Index
        """
        print("\n" + "="*80)
        print("CREATE IMPULSE CONTROL INDEX (ICI)")
        print("="*80)
        
        # Ensure features are normalized
        if 'session_speed_normalized' not in self.df.columns:
            self.normalize_features(['session_speed', 'discount_sensitivity'])
        
        # Calculate ICI
        self.df['Impulse_Control_Index'] = (
            session_speed_weight * self.df['session_speed_normalized'] +
            discount_weight * self.df['discount_sensitivity_normalized'] +
            urgency_weight * self.df['urgency_score'] +
            night_weight * self.df['night_purchase_flag']
        )
        
        # Normalize ICI to 0-1 range
        ici_min = self.df['Impulse_Control_Index'].min()
        ici_max = self.df['Impulse_Control_Index'].max()
        
        if ici_max > ici_min:
            self.df['Impulse_Control_Index'] = (
                (self.df['Impulse_Control_Index'] - ici_min) / (ici_max - ici_min)
            )
        else:
            self.df['Impulse_Control_Index'] = 0
        
        print(f"\n✓ Impulse Control Index (ICI) created")
        print(f"  Formula: {session_speed_weight}*session_speed + " +
              f"{discount_weight}*discount + {urgency_weight}*urgency + {night_weight}*night")
        print(f"\n  Statistics:")
        print(f"    Mean ICI: {self.df['Impulse_Control_Index'].mean():.4f}")
        print(f"    Median ICI: {self.df['Impulse_Control_Index'].median():.4f}")
        print(f"    Std Dev: {self.df['Impulse_Control_Index'].std():.4f}")
        print(f"    Min: {self.df['Impulse_Control_Index'].min():.4f}")
        print(f"    Max: {self.df['Impulse_Control_Index'].max():.4f}")
        
        return self.df
    
    def create_impulse_purchase_label(self, ici_threshold: float = 0.6) -> pd.DataFrame:
        """
        Create target variable: Impulse_Purchase = 1 if ICI > threshold, else 0
        
        Args:
            ici_threshold (float): Threshold for impulse purchase classification (default: 0.6)
        
        Returns:
            pd.DataFrame: Dataset with Impulse_Purchase target variable
        """
        print("\n" + "="*80)
        print(f"CREATE TARGET VARIABLE (ICI Threshold: {ici_threshold})")
        print("="*80)
        
        self.df['Impulse_Purchase'] = (self.df['Impulse_Control_Index'] > ici_threshold).astype(int)
        
        impulse_count = self.df['Impulse_Purchase'].sum()
        non_impulse_count = len(self.df) - impulse_count
        
        print(f"\n✓ Impulse_Purchase target variable created")
        print(f"  Impulse Purchases (1): {impulse_count} ({impulse_count/len(self.df)*100:.1f}%)")
        print(f"  Non-Impulse Purchases (0): {non_impulse_count} ({non_impulse_count/len(self.df)*100:.1f}%)")
        print(f"  Class Balance Ratio: {impulse_count/max(non_impulse_count, 1):.2f}")
        
        return self.df
    
    def get_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix and target variable
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        target = self.df['Impulse_Purchase']
        
        # Select engineered features and relevant original features
        feature_cols = [
            'session_speed', 'urgency_score', 'discount_sensitivity',
            'night_purchase_flag', 'mobile_user_flag',
            'Impulse_Control_Index', 'previous_purchases', 'customer_age'
        ]
        
        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        features = self.df[feature_cols]
        
        print(f"\n✓ Feature matrix created with {len(feature_cols)} features")
        
        return features, target
    
    def get_engineered_data(self) -> pd.DataFrame:
        """
        Get the complete engineered dataset
        
        Returns:
            pd.DataFrame: Dataset with all engineered features
        """
        return self.df.copy()
    
    def print_feature_summary(self) -> None:
        """
        Print summary of all engineered features
        """
        print("\n" + "="*80)
        print("ENGINEERED FEATURES SUMMARY")
        print("="*80)
        
        engineered_features = [
            'session_speed', 'urgency_score', 'discount_sensitivity',
            'night_purchase_flag', 'mobile_user_flag', 'Impulse_Control_Index',
            'Impulse_Purchase'
        ]
        
        for feature in engineered_features:
            if feature in self.df.columns:
                print(f"\n{feature}:")
                print(f"  Type: {self.df[feature].dtype}")
                print(f"  Mean: {self.df[feature].mean():.4f}")
                print(f"  Std: {self.df[feature].std():.4f}")
                print(f"  Min: {self.df[feature].min():.4f}")
                print(f"  Max: {self.df[feature].max():.4f}")
