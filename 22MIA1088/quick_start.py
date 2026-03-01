"""
QUICK START GUIDE - Impulse Control Predictor
===============================================

This file contains examples of how to use the project modules.
Run these examples to understand the workflow.
"""

# ============================================================================
# EXAMPLE 1: Data Processing
# ============================================================================

def example_data_processing():
    """Example: Load and process data"""
    from data_processing import DataProcessor, create_sample_dataset
    import os
    
    print("\n" + "="*80)
    print("EXAMPLE 1: DATA PROCESSING")
    print("="*80)
    
    # Create sample dataset if needed
    if not os.path.exists('data/ecommerce_customer_churn_dataset.csv'):
        print("\nCreating sample dataset...")
        create_sample_dataset('data/ecommerce_customer_churn_dataset.csv', n_samples=1000)
    
    # Load dataset
    processor = DataProcessor('data/ecommerce_customer_churn_dataset.csv')
    df = processor.load_dataset()
    
    # Display statistics
    processor.display_summary_statistics()
    
    # Check missing values
    processor.check_missing_values()
    
    # Handle missing values
    df_clean = processor.handle_missing_values()
    
    return df_clean


# ============================================================================
# EXAMPLE 2: Feature Engineering
# ============================================================================

def example_feature_engineering(df):
    """Example: Engineer features"""
    from feature_engineering import ImpulseFeatureEngineer
    
    print("\n" + "="*80)
    print("EXAMPLE 2: FEATURE ENGINEERING")
    print("="*80)
    
    # Create feature engineer
    engineer = ImpulseFeatureEngineer(df)
    
    # Create all features
    engineer.create_session_speed()
    engineer.create_urgency_score()
    engineer.create_discount_sensitivity()
    engineer.create_night_purchase_flag()
    engineer.create_mobile_user_flag()
    
    # Create ICI and target variable
    engineer.normalize_features(['session_speed', 'discount_sensitivity'])
    engineer.create_impulse_control_index()
    engineer.create_impulse_purchase_label()
    
    # Print summary
    engineer.print_feature_summary()
    
    # Get feature matrix
    X, y = engineer.get_feature_matrix()
    
    return engineer.get_engineered_data(), X, y


# ============================================================================
# EXAMPLE 3: Model Training
# ============================================================================

def example_model_training(X, y):
    """Example: Train models"""
    from sklearn.model_selection import train_test_split
    from train_model import ImpulseModelTrainer
    
    print("\n" + "="*80)
    print("EXAMPLE 3: MODEL TRAINING")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create trainer
    trainer = ImpulseModelTrainer(X, y)
    
    # Train all models
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_xgboost()
    
    # Visualize comparison
    trainer.visualize_model_comparison()
    trainer.visualize_xgboost_analysis()
    trainer.visualize_roc_detailed()
    trainer.visualize_feature_importance_xgb()
    
    return trainer


# ============================================================================
# EXAMPLE 4: Model Evaluation
# ============================================================================

def example_model_evaluation(trainer):
    """Example: Evaluate models"""
    from evaluation import print_evaluation_summary
    
    print("\n" + "="*80)
    print("EXAMPLE 4: MODEL EVALUATION")
    print("="*80)
    
    # Print summary
    print_evaluation_summary(trainer)
    
    # Get model summary
    summary = trainer.get_model_summary()
    print("\n" + summary.to_string(index=False))


# ============================================================================
# EXAMPLE 5: SHAP Explainability
# ============================================================================

def example_shap_explainability(trainer):
    """Example: Explain model with SHAP"""
    from explainability import ModelExplainer, print_explainability_summary
    import pandas as pd
    
    print("\n" + "="*80)
    print("EXAMPLE 5: SHAP EXPLAINABILITY")
    print("="*80)
    
    # Get test data
    X_test = trainer.X_test
    
    # Create explainer
    explainer = ModelExplainer(
        trainer.models['XGBoost'],
        trainer.X_train,
        X_test
    )
    
    # Initialize SHAP
    explainer.initialize_shap()
    
    # Create visualizations
    explainer.plot_summary_plot(plot_type='bar')
    explainer.plot_summary_plot(plot_type='beeswarm')
    
    # Get feature importance
    importance = explainer.get_feature_importance_explanation(top_n=10)
    
    # Print summary
    print_explainability_summary()


# ============================================================================
# EXAMPLE 6: Save and Load Model
# ============================================================================

def example_save_load_model(trainer):
    """Example: Save and load trained model"""
    import pickle
    import os
    
    print("\n" + "="*80)
    print("EXAMPLE 6: SAVE AND LOAD MODEL")
    print("="*80)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/xgb_impulse_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(trainer.models['XGBoost'], f)
    print(f"✓ Model saved to: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print(f"✓ Model loaded successfully!")
    
    # Make prediction
    prediction = loaded_model.predict(trainer.X_test.iloc[:5])
    probability = loaded_model.predict_proba(trainer.X_test.iloc[:5])
    
    print(f"\nSample Predictions:")
    print(f"  Predictions: {prediction}")
    print(f"  Probabilities:\n{probability}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPULSE CONTROL PREDICTOR - QUICK START GUIDE")
    print("="*80)
    
    try:
        # Run examples
        df = example_data_processing()
        df_engineered, X, y = example_feature_engineering(df)
        trainer = example_model_training(X, y)
        example_model_evaluation(trainer)
        
        # SHAP (optional - may take time)
        # example_shap_explainability(trainer)
        
        example_save_load_model(trainer)
        
        print("\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nMake sure:")
        print("  1. All required packages are installed: pip install -r requirements.txt")
        print("  2. Dataset exists at: data/ecommerce_customer_churn_dataset.csv")
        print("  3. All modules are in the same directory")
