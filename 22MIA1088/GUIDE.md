# 📚 Impulse Control Predictor - Detailed Guide

## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Project Workflow](#project-workflow)
3. [Using Google Colab](#using-google-colab)
4. [Using Local Environment](#using-local-environment)
5. [Module Documentation](#module-documentation)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- 2 GB free disk space

### Quick Installation

#### Option 1: Using requirements.txt (Recommended)
```bash
# Navigate to project directory
cd 22MIA1088

# Install all dependencies
pip install -r requirements.txt
```

#### Option 2: Individual Installations
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

#### Verify Installation
```python
import pandas as pd
import xgboost as xgb
import shap
print("✓ All packages installed successfully!")
```

---

## Project Workflow

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING & EXPLORATION                               │
│    - Load CSV dataset                                        │
│    - Display statistics                                      │
│    - Check missing values                                    │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA PREPROCESSING                                        │
│    - Handle missing values                                   │
│    - Standardize formats                                     │
│    - Clean data                                              │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING                                       │
│    - Create domain-specific features                         │
│    - Normalize features                                      │
│    - Calculate Impulse Control Index                         │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CREATE TARGET VARIABLE                                    │
│    - ICI > 0.6 → Impulse Purchase (1)                        │
│    - ICI ≤ 0.6 → No Impulse (0)                              │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. TRAIN-TEST SPLIT                                          │
│    - 80% training, 20% testing                               │
│    - Stratified split for balance                            │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. MODEL TRAINING                                            │
│    - Logistic Regression                                     │
│    - Random Forest                                           │
│    - XGBoost (Primary)                                       │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. MODEL EVALUATION                                          │
│    - Calculate metrics (Accuracy, Precision, etc.)           │
│    - Create confusion matrices                               │
│    - Plot ROC curves                                         │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. FEATURE IMPORTANCE                                        │
│    - Extract XGBoost importances                             │
│    - Visualize top features                                  │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. SHAP EXPLAINABILITY                                       │
│    - Create SHAP summary plots                               │
│    - Generate force plots                                    │
│    - Show dependence plots                                   │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│ 10. SAVE MODEL                                               │
│     - Pickle XGBoost model                                   │
│     - Save metadata                                          │
│     - Ready for production                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Using Google Colab

### Recommended Method

```
1. Go to Google Colab: https://colab.research.google.com/

2. Click "File" → "Open Notebook" → "Upload" tab

3. Select: Impulse_Control_Predictor.ipynb

4. Install packages (first cell runs automatically)

5. Run cells sequentially (Shift + Enter)

6. All visualizations display inline
```

### Uploading Your Dataset to Colab

```python
# In first cell of Colab notebook
from google.colab import files
uploaded = files.upload()  # Select your CSV file
```

### Downloading Results from Colab

```python
# Download trained model
from google.colab import files
files.download('models/xgb_impulse_model.pkl')
files.download('models/model_metadata.json')
```

---

## Using Local Environment

### 1. Running the Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Opens browser at http://localhost:8888

# Open Impulse_Control_Predictor.ipynb

# Run cells sequentially
```

### 2. Running Individual Modules

```python
# Example: Using data processing module
from data_processing import DataProcessor, create_sample_dataset

# Create sample data
create_sample_dataset('data/sample.csv', n_samples=1000)

# Load and process
processor = DataProcessor('data/sample.csv')
df = processor.load_dataset()
processor.display_summary_statistics()
processor.check_missing_values()
df_clean = processor.handle_missing_values()
```

### 3. Using quick_start.py

```bash
# Run all examples
python quick_start.py

# This will:
# - Load sample data
# - Create features
# - Train models
# - Evaluate models
# - Save trained model
```

---

## Module Documentation

### 1. data_processing.py

**Purpose:** Load and prepare data

**Key Classes:**
- `DataProcessor` - Main data handling class

**Key Methods:**
```python
processor = DataProcessor('path/to/data.csv')

# Load data
df = processor.load_dataset()

# Explore data
processor.display_summary_statistics()
processor.check_missing_values()

# Clean data
df_clean = processor.handle_missing_values(strategy='smart')

# Get processed data
df_cleaned = processor.get_cleaned_data()
```

### 2. feature_engineering.py

**Purpose:** Create impulse purchase indicators

**Key Classes:**
- `ImpulseFeatureEngineer` - Feature creation

**Key Methods:**
```python
engineer = ImpulseFeatureEngineer(df)

# Create individual features
engineer.create_session_speed()
engineer.create_urgency_score()
engineer.create_discount_sensitivity()
engineer.create_night_purchase_flag()
engineer.create_mobile_user_flag()

# Create Impulse Control Index
engineer.normalize_features(['session_speed', 'discount_sensitivity'])
engineer.create_impulse_control_index(
    session_speed_weight=0.3,
    discount_weight=0.3,
    urgency_weight=0.2,
    night_weight=0.2
)

# Create target variable
engineer.create_impulse_purchase_label(ici_threshold=0.6)

# Get results
X, y = engineer.get_feature_matrix()
df_engineered = engineer.get_engineered_data()
```

### 3. train_model.py

**Purpose:** Train ML models with extensive visualizations

**Key Classes:**
- `ImpulseModelTrainer` - Model training

**Key Methods:**
```python
trainer = ImpulseModelTrainer(X, y, test_size=0.2)

# Train models
trainer.train_logistic_regression()
trainer.train_random_forest()
trainer.train_xgboost()

# Visualize results
trainer.visualize_model_comparison()      # 4-panel comparison
trainer.visualize_xgboost_analysis()      # Detailed XGBoost analysis
trainer.visualize_roc_detailed()          # ROC curves
trainer.visualize_feature_importance_xgb()# Feature importance

# Get results
best_name, best_model = trainer.get_best_model()
summary = trainer.get_model_summary()
```

### 4. evaluation.py

**Purpose:** Evaluate model performance comprehensively

**Key Classes:**
- `ModelEvaluator` - Metrics calculation
- `MetricsVisualizer` - Visualization

**Key Methods:**
```python
evaluator = ModelEvaluator(y_true, y_pred, y_proba)

# Calculate metrics
metrics = evaluator.calculate_metrics()
evaluator.print_metrics_summary(model_name='XGBoost')
evaluator.print_classification_report()

# Get detailed info
cm_info = evaluator.get_confusion_matrix_info()
metrics_df = evaluator.get_metrics_dataframe()
class_metrics = evaluator.get_class_wise_metrics()

# Visualizations
MetricsVisualizer.plot_confusion_matrix(cm)
MetricsVisualizer.plot_metrics_comparison({...})
MetricsVisualizer.plot_precision_recall_curve(y_true, y_proba)
```

### 5. explainability.py

**Purpose:** Explain predictions using SHAP

**Key Classes:**
- `ModelExplainer` - SHAP analysis

**Key Methods:**
```python
explainer = ModelExplainer(model, X_train, X_test)

# Initialize SHAP
explainer.initialize_shap()

# Create visualizations
explainer.plot_summary_plot(plot_type='bar')     # Bar plot
explainer.plot_summary_plot(plot_type='beeswarm')# Beeswarm
explainer.plot_waterfall(sample_idx=0)           # Waterfall
explainer.plot_force_plot(sample_idx=0)          # Force plot
explainer.plot_dependence_plots(max_features=5)  # Dependence

# Get insights
importance = explainer.get_feature_importance_explanation(top_n=10)
explainer.explain_prediction(sample_idx=0)
```

---

## Understanding Results

### Key Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 0.0-1.0 | Proportion of correct predictions |
| **Precision** | 0.0-1.0 | Accuracy of positive predictions |
| **Recall** | 0.0-1.0 | Coverage of actual positives |
| **F1 Score** | 0.0-1.0 | Harmonic mean of precision & recall |
| **ROC-AUC** | 0.5-1.0 | Performance at all thresholds (0.5=random, 1.0=perfect) |

### Interpreting Visualizations

#### 1. Confusion Matrix
```
                 Predicted
              No Impulse | Impulse
         ┌──────────────┬─────────┐
         │    TN        │   FP    │
Actual   ├──────────────┼─────────┤
         │    FN        │   TP    │
         └──────────────┴─────────┘

TN: Correctly predicted "No Impulse"
FP: Incorrectly predicted "Impulse" (FALSE POSITIVE)
FN: Incorrectly predicted "No Impulse" (FALSE NEGATIVE)
TP: Correctly predicted "Impulse"
```

#### 2. ROC Curve
- **Higher curve** = Better performance
- **Top-left corner** = Ideal classifier
- **Diagonal line** = Random classifier (AUC=0.5)
- **AUC > 0.8** = Good discriminative ability

#### 3. SHAP Summary Plot (Bar)
- **Longer bars** = More important features
- **Ordered from top to bottom** = Importance ranking
- **Use for understanding overall model behavior**

#### 4. SHAP Summary Plot (Beeswarm)
- **Red points (high values)** = Push prediction up
- **Blue points (low values)** = Push prediction down
- **Width of distribution** = Variability of feature impact
- **Use for understanding feature interactions**

---

## Troubleshooting

### Issue 1: "Module not found" Error

**Problem:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
pip install xgboost
pip install -r requirements.txt  # Install all at once
```

### Issue 2: Out of Memory Error

**Problem:** Notebook crashes with memory error

**Solution:**
```python
# Reduce dataset size
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data

# Use fewer trees in models
trainer = ImpulseModelTrainer(X, y)
# Models automatically use reasonable defaults
```

### Issue 3: SHAP Plots Not Displaying

**Problem:** SHAP visualizations don't appear

**Solution:**
```python
import matplotlib.pyplot as plt
plt.show()  # Force display

# Or use matplotlib notebook
%matplotlib notebook  # In Jupyter
```

### Issue 4: Dataset Not Found

**Problem:**
```
FileNotFoundError: data/ecommerce_customer_churn_dataset.csv
```

**Solution:**
```python
# Option 1: Create sample dataset
from data_processing import create_sample_dataset
create_sample_dataset('data/sample.csv', n_samples=1000)

# Option 2: Provide your own dataset
df = pd.read_csv('path/to/your/data.csv')
```

### Issue 5: Model Training Takes Too Long

**Problem:** XGBoost training is slow

**Solution:**
```python
# Reduce n_estimators
xgb_model = xgb.XGBClassifier(
    n_estimators=50,  # Reduced from 100
    max_depth=3,      # Reduced from 5
    learning_rate=0.1
)

# Or use early stopping
```

### Issue 6: Poor Model Performance

**Problem:** Accuracy is too low

**Solutions:**
1. Check data quality
2. Engineer better features
3. Adjust ICI threshold (try 0.5 or 0.7)
4. Balance classes (if imbalanced)
5. Get more training data

---

## Tips & Best Practices

### 1. Data Quality
- ✓ Ensure sufficient data (min 500 samples)
- ✓ Remove duplicates and outliers
- ✓ Check for data leakage
- ✓ Validate feature engineering logic

### 2. Model Training
- ✓ Use stratified split for balanced classes
- ✓ Always use validation/test set
- ✓ Monitor training curves
- ✓ Save trained models regularly

### 3. Evaluation
- ✓ Use multiple metrics, not just accuracy
- ✓ Check precision-recall tradeoff
- ✓ Validate on unseen test data
- ✓ Consider business impact

### 4. Deployment
- ✓ Version control your models
- ✓ Document assumptions
- ✓ Set up monitoring
- ✓ Plan for retraining

---

## FAQ

**Q: Can I use my own dataset?**
A: Yes! Replace the CSV path in DataProcessor with your file path. Make sure your CSV has the required columns: `total_time_spent`, `number_of_clicks`, `add_to_cart_time`, `discount_percentage`, `purchase_flag`, `purchase_time`, `device_type`, etc.

**Q: How do I improve model performance?**
A: 1) Get more data, 2) Engineer better features, 3) Tune hyperparameters, 4) Try ensemble methods, 5) Handle class imbalance.

**Q: Can I use different model thresholds?**
A: Yes! Change `ici_threshold` in `create_impulse_purchase_label()` method. Higher threshold = stricter impulse purchase definition.

**Q: How do I make predictions on new data?**
A: Load the saved model and call `predict()` or `predict_proba()` on your new features (same feature names and order).

---

## Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**Last Updated:** March 2026  
**Status:** Production Ready ✓  
**For Support:** Refer to README.md or code comments

