"""
Model Explainability Module for Impulse Control Predictor
SHAP-based model interpretation and feature importance explanation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Explain model predictions using SHAP and other methods
    """
    
    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.shap_values = None
        self.explainer = None
        
    def initialize_shap(self) -> 'ModelExplainer':
        """
        Initialize SHAP explainer
        Only works with XGBoost models or other tree-based models
        
        Returns:
            Self for method chaining
        """
        print("\n" + "="*80)
        print("INITIALIZING SHAP EXPLAINER")
        print("="*80)
        
        try:
            import shap
            
            # For tree-based models
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(self.X_test)
                
                print("✓ SHAP TreeExplainer initialized successfully!")
                print(f"  Model type: {type(self.model).__name__}")
                print(f"  Features: {len(self.X_test.columns)}")
                print(f"  Samples: {len(self.X_test)}")
                
            else:
                print("✗ Model type not supported for SHAP TreeExplainer")
                self.explainer = None
                
        except ImportError:
            print("✗ SHAP library not installed. Install with: pip install shap")
            self.explainer = None
        except Exception as e:
            print(f"✗ Error initializing SHAP: {str(e)}")
            self.explainer = None
        
        return self
    
    def plot_summary_plot(self, plot_type='bar', max_display=10) -> None:
        """
        Plot SHAP summary plot
        
        Args:
            plot_type (str): Type of plot ('bar', 'beeswarm')
            max_display (int): Maximum features to display
        """
        if self.shap_values is None:
            print("SHAP values not initialized. Call initialize_shap() first.")
            return
        
        print("\n" + "="*80)
        print(f"SHAP SUMMARY PLOT ({plot_type.upper()})")
        print("="*80)
        
        try:
            import shap
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            shap.summary_plot(self.shap_values, self.X_test, plot_type=plot_type,
                            max_display=max_display, show=False)
            
            plt.title(f'SHAP {plot_type.capitalize()} Summary Plot - Top {max_display} Features',
                     fontweight='bold', fontsize=12, pad=20)
            plt.tight_layout()
            plt.show()
            
            print(f"✓ {plot_type.capitalize()} summary plot displayed!")
            
        except Exception as e:
            print(f"✗ Error creating summary plot: {str(e)}")
    
    def plot_waterfall(self, sample_idx: int = 0) -> None:
        """
        Plot SHAP waterfall plot for a specific sample
        
        Args:
            sample_idx (int): Index of sample to explain
        """
        if self.shap_values is None:
            print("SHAP values not initialized. Call initialize_shap() first.")
            return
        
        print("\n" + "="*80)
        print(f"SHAP WATERFALL PLOT (Sample #{sample_idx})")
        print("="*80)
        
        try:
            import shap
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            shap.plots.waterfall(shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=self.X_test.iloc[sample_idx],
                feature_names=self.X_test.columns.tolist()
            ), show=False)
            
            plt.title(f'SHAP Waterfall Plot - Sample #{sample_idx}',
                     fontweight='bold', fontsize=12, pad=20)
            plt.tight_layout()
            plt.show()
            
            print(f"✓ Waterfall plot displayed for sample #{sample_idx}!")
            
        except Exception as e:
            print(f"✗ Error creating waterfall plot: {str(e)}")
    
    def plot_force_plot(self, sample_idx: int = 0) -> None:
        """
        Display SHAP force plot
        
        Args:
            sample_idx (int): Index of sample to explain
        """
        if self.shap_values is None:
            print("SHAP values not initialized. Call initialize_shap() first.")
            return
        
        print("\n" + "="*80)
        print(f"SHAP FORCE PLOT (Sample #{sample_idx})")
        print("="*80)
        
        try:
            import shap
            
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_idx],
                self.X_test.iloc[sample_idx],
                feature_names=self.X_test.columns.tolist(),
                matplotlib=True
            )
            
            plt.title(f'SHAP Force Plot - Sample #{sample_idx}',
                     fontweight='bold', fontsize=12, pad=20)
            plt.tight_layout()
            plt.show()
            
            print(f"✓ Force plot displayed for sample #{sample_idx}!")
            
        except Exception as e:
            print(f"✗ Error creating force plot: {str(e)}")
    
    def plot_dependence_plots(self, features=None, max_features=5) -> None:
        """
        Plot SHAP dependence plots for specified features
        
        Args:
            features (list): List of feature names
            max_features (int): Maximum features to plot
        """
        if self.shap_values is None:
            print("SHAP values not initialized. Call initialize_shap() first.")
            return
        
        print("\n" + "="*80)
        print("SHAP DEPENDENCE PLOTS")
        print("="*80)
        
        try:
            import shap
            
            if features is None:
                # Use top features by mean absolute SHAP value
                feature_importance = np.abs(self.shap_values).mean(axis=0)
                top_indices = np.argsort(-feature_importance)[:max_features]
                features = self.X_test.columns[top_indices].tolist()
            
            n_features = min(len(features), 4)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()
            
            fig.suptitle('SHAP Dependence Plots - Feature Impact on Model Output',
                        fontweight='bold', fontsize=14)
            
            for idx, feature in enumerate(features[:4]):
                try:
                    shap.dependence_plot(
                        feature,
                        self.shap_values,
                        self.X_test,
                        feature_names=self.X_test.columns.tolist(),
                        ax=axes[idx],
                        show=False
                    )
                    axes[idx].set_title(f'Impact of {feature}', fontweight='bold')
                except Exception as e:
                    print(f"Warning: Could not plot {feature}: {str(e)}")
            
            # Hide unused subplots
            for idx in range(len(features), 4):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            print(f"✓ Dependence plots displayed for {len(features)} features!")
            
        except Exception as e:
            print(f"✗ Error creating dependence plots: {str(e)}")
    
    def get_feature_importance_explanation(self, top_n=10) -> pd.DataFrame:
        """
        Get feature importance explanation from SHAP
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.shap_values is None:
            print("SHAP values not initialized. Call initialize_shap() first.")
            return None
        
        print("\n" + "="*80)
        print("SHAP FEATURE IMPORTANCE SUMMARY")
        print("="*80)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Mean_SHAP_Value': mean_abs_shap,
            'Importance_Rank': np.argsort(-mean_abs_shap) + 1
        }).sort_values('Mean_SHAP_Value', ascending=False)
        
        print("\n" + importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def explain_prediction(self, sample_idx: int = 0) -> None:
        """
        Provide detailed explanation for a single prediction
        
        Args:
            sample_idx (int): Index of sample to explain
        """
        print("\n" + "="*80)
        print(f"DETAILED PREDICTION EXPLANATION - Sample #{sample_idx}")
        print("="*80)
        
        sample = self.X_test.iloc[sample_idx]
        prediction = self.model.predict(self.X_test.iloc[[sample_idx]])[0]
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(self.X_test.iloc[[sample_idx]])[0]
            print(f"\nPrediction: {'IMPULSE PURCHASE' if prediction == 1 else 'NO IMPULSE PURCHASE'}")
            print(f"Probability (Class 0): {proba[0]:.4f}")
            print(f"Probability (Class 1): {proba[1]:.4f}")
        else:
            print(f"Prediction: {prediction}")
        
        print(f"\nFeature Values for Sample #{sample_idx}:")
        print("-"*80)
        for feature, value in sample.items():
            print(f"  {feature:.<40} {value:.4f}")
        
        if self.shap_values is not None:
            print(f"\nTop 5 Contributing Features (SHAP Values):")
            print("-"*80)
        
            shap_row = self.shap_values[sample_idx]
            top_indices = np.argsort(-np.abs(shap_row))[:5]
            
            for rank, idx in enumerate(top_indices, 1):
                feature = self.X_test.columns[idx]
                shap_value = shap_row[idx]
                feature_value = sample[feature]
                
                direction = "↑ increases" if shap_value > 0 else "↓ decreases"
                print(f"  {rank}. {feature:.<35} SHAP={shap_value:>10.4f} {direction}")
                print(f"     Feature value: {feature_value:.4f}")


def print_explainability_summary() -> None:
    """
    Print summary of explainability analysis
    """
    print("\n" + "="*80)
    print("MODEL EXPLAINABILITY SUMMARY")
    print("="*80)
    
    print("""
SHAP (SHapley Additive exPlanations) Analysis:
-----------------------------------------------
✓ SHAP Values: Explain each feature's contribution to individual predictions
✓ Summary Plot (Bar): Overall feature importance across all predictions
✓ Summary Plot (Beeswarm): Distribution of SHAP values for each feature
✓ Waterfall Plot: How each feature affects a specific prediction
✓ Force Plot: Visual representation of prediction composition
✓ Dependence Plot: Feature interaction with model output
✓ Feature Importance: Ranked importance by mean absolute SHAP value

Key Insights:
- Features with high |SHAP values| have strong impact on predictions
- SHAP values integrate feature importance with game theory
- Individual sample explanations help understand model behavior
- Helps identify bias and model fairness issues
""")
