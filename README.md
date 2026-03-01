 Impulse Control Predictor for Online Shopping

A complete Machine Learning project to predict whether a customer will make an impulsive purchase based on behavioral and transactional features.

 📋 Project Overview

Goal: Predict impulsive purchases in e-commerce using machine learning

Target Variable: Impulse_Purchase (0 or 1)

Dataset:** E-commerce customer data with behavioral metrics

 🎯 Key Features

 Engineered Features:
- session_speed: Average time per click (seconds/click)
- urgency_score: Quick add-to-cart action (< 2 minutes)
- discount_sensitivity: Response to discount offers
- night_purchase_flag: Purchases between 10 PM and 5 AM
- mobile_user_flag: Mobile device usage indicator

 Target Variable Calculation:

Impulse Control Index (ICI) = 0.3*normalized_session_speed
                            + 0.3*normalized_discount
                            + 0.2*urgency_score
                            + 0.2*night_purchase_flag

Impulse_Purchase = 1 if ICI > 0.6 else 0


📁 Project Structure

22MIA1088/
│
├── Impulse_Control_Predictor.ipynb
│   └── Main Jupyter Notebook (End-to-End ML Pipeline)
│
├── streamlit_dashboard.py
│   └── Interactive Streamlit Dashboard for Predictions & Visualization
│
├── data/
│   └── ecommerce_customer_churn_dataset.csv
│       └── Raw E-commerce Dataset
│
├── models/
│   ├── xgb_impulse_model.pkl
│   │   └── Trained XGBoost Model
│   │
│   ├── scaler.pkl
│   │   └── MinMaxScaler for Feature Scaling
│   │
│   ├── feature_columns.pkl
│   │   └── List of Model Feature Columns
│   │
│   └── model_metadata.json
│       └── Model Performance & Configuration Details
│
├── data_processing.py
│   └── Data Loading, Cleaning & Preprocessing
│
├── feature_engineering.py
│   └── Impulse-Based Feature Creation
│
├── train_model.py
│   └── Model Training & Comparison (LR, RF, XGB)
│
├── evaluation.py
│   └── Performance Metrics & Visualization
│
├── explainability.py
│   └── SHAP-Based Model Interpretation
│
└── README.md
    └── Project Documentation


 🚀 Quick Start

 Option 1: Google Colab (Recommended)
1. Open the Jupyter notebook: `Impulse_Control_Predictor.ipynb`
2. Upload to Google Colab
3. Run cells sequentially
4. All visualizations will display automatically

 Option 2: Local Jupyter Notebook

 Install required packages
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn

 Run the notebook
jupyter notebook Impulse_Control_Predictor.ipynb


 Option 3: Interactive Streamlit Dashboard ✨

 Install Streamlit (if not already installed)
pip install streamlit

 Launch dashboard
 run streamlit_dashboard.py

✨ Dashboard URL:http://localhost:8501  
✨ Features:
- 5 interactive tabs (Home, Dashboard, Predictions, Analysis, Settings)
- 6 Plotly streamlitvisualizations
- CSV upload for batch predictions
- Real-time model predictions
- Model analysis and insights

 📊 Models Trained

 1. Logistic Regression
- **Purpose:** Baseline linear model
- **Interpretability:** High
- **Speed:** Very fast

 2. Random Forest
- **Purpose:** Ensemble baseline
- **Interpretability:** Medium
- **Speed:** Medium

 3. XGBoost (PRIMARY MODEL) ⭐
- Purpose: Gradient boosting with advanced features
- Interpretability: Medium (with SHAP)
- Speed:Fast
- Best for: Production deployment

 📈 Evaluation Metrics


Accuracy:  Overall correctness of predictions
Precision: True positives / All positive predictions
Recall:    True positives / All actual positives
F1 Score:  Harmonic mean of precision and recall
ROC-AUC:   Area under the ROC curve (0.5 = random, 1.0 = perfect)


 🔍 Model Explainability

### SHAP Analysis
- Summary Plot (Bar): Feature importance ranking
- Summary Plot (Beeswarm): SHAP value distributions
- Force Plot: Individual prediction decomposition
- Waterfall Plot: Feature contributions to prediction
- Dependence Plot: Feature interactions


📊 Expected Performance

Based on typical e-commerce data:



Accuracy : 75-85% 
Precision : 70-80% 
Recall : 75-85% 
F1 Score : 72-82% 
ROC-AUC :80-90% 



 🎓 Learning Outcomes

This project demonstrates:
- ✓ Complete ML pipeline development
- ✓ Feature engineering best practices
- ✓ Multiple model training and comparison
- ✓ Comprehensive model evaluation
- ✓ Model explainability with SHAP
- ✓ Production-ready code organization
- ✓ Extended visualizations and analysis

📚 Modules Overview

 data_processing.py
Handles data loading, exploration, and cleaning:
- `DataProcessor.load_dataset()` - Load CSV file
- `DataProcessor.check_missing_values()` - Missing value analysis
- `DataProcessor.handle_missing_values()` - Smart imputation
- `DataProcessor.display_summary_statistics()` - Data overview

 feature_engineering.py
Creates impulse purchase indicators:
- `ImpulseFeatureEngineer.create_session_speed()`
- `ImpulseFeatureEngineer.create_urgency_score()`
- `ImpulseFeatureEngineer.create_discount_sensitivity()`
- `ImpulseFeatureEngineer.create_night_purchase_flag()`
- `ImpulseFeatureEngineer.create_mobile_user_flag()`
- `ImpulseFeatureEngineer.create_impulse_control_index()`

 train_model.py
Train and compare ML models:
- `ImpulseModelTrainer.train_logistic_regression()`
- `ImpulseModelTrainer.train_random_forest()`
- `ImpulseModelTrainer.train_xgboost()`
- `ImpulseModelTrainer.visualize_model_comparison()`
- `ImpulseModelTrainer.visualize_xgboost_analysis()`

 evaluation.py
Comprehensive model evaluation:
- `ModelEvaluator.calculate_metrics()`
- `ModelEvaluator.print_classification_report()`
- `MetricsVisualizer.plot_confusion_matrix()`
- `MetricsVisualizer.plot_precision_recall_curve()`

explainability.py
SHAP-based model interpretation:
- `ModelExplainer.initialize_shap()`
- `ModelExplainer.plot_summary_plot()`
- `ModelExplainer.plot_waterfall()`
- `ModelExplainer.plot_dependence_plots()`

🎨 Visualizations Generated

1. **ICI Distribution** - Target variable distribution
2. **Model Accuracy Comparison** - Bar chart of accuracies
3. **ROC Curves** - Performance comparison at different thresholds
4. **Confusion Matrices** - True/False positives/negatives
5. **Metrics Heatmap** - All metrics for all models
6. **Feature Importance** - Top features from XGBoost
7. **SHAP Summary Plots** - Bar and beeswarm plots
8. **SHAP Force Plots** - Individual prediction explanations
9. **SHAP Dependence Plots** - Feature-target interactions

🔐 Production Considerations

Before deploying:
1. ✓ Monitor model performance on new data
2. ✓ Set up data validation pipeline
3. ✓ Implement logging and monitoring
4. ✓ Plan for model retraining
5. ✓ Create prediction API endpoint
6. ✓ Implement A/B testing
7. ✓ Document assumptions and limitations

📝 Notes

- **Data Quality:** Model performance depends on data quality
- **Class Balance:** Handle imbalanced classes if necessary
- **Feature Scaling:** Features are normalized in the pipeline
- **Reproducibility:** Random state is fixed at 42 for consistency

🤝 Contributing

To improve the model:
1. Collect more training data
2. Engineer new features based on domain knowledge
3. Tune hyperparameters
4. Try different models
5. Implement ensemble methods

📞 Support

For issues or questions:
- Check the Jupyter notebook for detailed comments
- Review individual module documentation
- Examine SHAP plots for model insights

📄 License

This project is provided as-is for educational purposes.

---

**Version:** 1.0  
**Last Updated:** March 2026  
**Status:** Production Ready ✓
