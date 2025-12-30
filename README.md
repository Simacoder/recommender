# Farm to Feed: Produce Recommendation ML Pipeline

A machine learning solution to predict customer purchase behavior and optimize fresh produce recommendations for Farm to Feed's marketplace platform.

## 🎯 Project Overview

Farm to Feed connects farmers, businesses, and consumers with odd-looking fruits and vegetables that would otherwise go to waste. This project builds a recommender system that:

- **Predicts purchase likelihood** for each customer-product pair in the next 7 and 14 days
- **Forecasts purchase quantities** to help with inventory management
- **Reduces food waste** by optimizing supply chain efficiency
- **Increases farmer income** by improving sales predictability
- **Expands market access** for smallholder farmers

## 📊 Dataset

### Data Structure
The dataset contains weekly transaction records with the following key columns:

**Identifiers:**
- `ID`: Unique customer–product–week identifier
- `customer_id`: Anonymized unique customer ID
- `product_unit_variant_id`: Specific product SKU (grade + unit combination)
- `week_start`: Monday start date of the week
- `product_id`: Higher-level product identifier
- `product_grade_variant_id`: Grade–unit configuration ID

**Behavioral Features:**
- `qty_this_week`: Total quantity purchased in the current week
- `num_orders_week`: Number of separate orders during the week
- `spend_this_week`: Total monetary spend in the week
- `purchased_this_week`: Binary purchase indicator
- `selling_price`: Price per unit during the week

**Product Information:**
- `grade_name`: Product grade (e.g., premium, standard, grade-B)
- `unit_name`: Selling unit (e.g., kg, bunch, pack)

**Customer Information:**
- `customer_category`: Segment (retailer, hotel, food service, etc.)
- `customer_status`: Account status (active, inactive)
- `customer_created_at`: Account registration timestamp

**Targets (7 and 14-day horizons):**
- `Target_purchase_next_1w`: Binary indicator of purchase in next 7 days
- `Target_qty_next_1w`: Total quantity purchased in next 7 days
- `Target_purchase_next_2w`: Binary indicator of purchase in next 14 days
- `Target_qty_next_2w`: Total quantity purchased in next 14 days

### Dataset Statistics
- **Time Period**: Weekly aggregated transaction data
- **Unique Customers**: Thousands of customers across different segments
- **Unique Products**: Multiple produce items with varying grades and units
- **Class Balance**: Imbalanced (more non-purchases than purchases)
- **Temporal Coverage**: Multiple weeks of historical data

## 🛠️ Technical Stack

```
Python 3.8+
pandas         - Data manipulation and analysis
numpy          - Numerical computations
scikit-learn   - Machine learning models
matplotlib     - Data visualization
seaborn        - Statistical visualization
```

## 📁 Project Structure

```
farm-to-feed-ml/
├── farm_to_feed_pipeline.py      # Main Python module with recommender class
├── farm_to_feed_notebook.ipynb   # Jupyter notebook with full pipeline
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── train.csv                      # Training dataset
├── test.csv                       # Test dataset (optional)
└── submission.csv                # Output predictions (generated)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/farm-to-feed-ml.git
cd farm-to-feed-ml

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline

**Option A: Using Python Script**

```python
from farm_to_feed_pipeline import FarmToFeedRecommender
import pandas as pd

# Load data
df_train = pd.read_csv('train.csv')

# Initialize recommender
recommender = FarmToFeedRecommender()

# Prepare data
df_train = recommender.load_and_prepare_data(df_train)

# Engineer features
features_df = recommender.engineer_features(df_train)
print(f"Created {len(features_df)} customer-product pairs")

# Create targets
targets_df = recommender.create_targets(df_train, features_df)

# Train models
recommender.train_models(features_df, targets_df)

# Make predictions
predictions = recommender.predict(features_df)

# Export submission
recommender.export_submission(predictions, 'submission.csv')
```

**Option B: Using Jupyter Notebook**

```bash
jupyter notebook farm_to_feed_notebook.ipynb
```

Then run cells sequentially from top to bottom.

### 3. Output

The pipeline generates `submission.csv` with the following format:

```csv
ID,Target_purchase_next_1w,Target_qty_next_1w,Target_purchase_next_2w,Target_qty_next_2w
12345_67890,0.75,3.5,0.82,5.2
12345_67891,0.15,0.8,0.25,1.2
...
```

Where:
- **ID**: customer_id_product_unit_variant_id
- **Target_purchase_next_1w**: Probability (0-1) of purchase in next 7 days
- **Target_qty_next_1w**: Predicted quantity for next 7 days
- **Target_purchase_next_2w**: Probability (0-1) of purchase in next 14 days
- **Target_qty_next_2w**: Predicted quantity for next 14 days

## 🧠 Feature Engineering

The pipeline automatically engineers 25+ features grouped into categories:

### Purchase History Features
- `purchase_weeks`: Number of weeks with at least one purchase
- `purchase_rate`: Proportion of weeks with purchases
- `weeks_since_last_purchase`: Recency metric

### Quantity Patterns
- `mean_qty`: Average quantity per purchase
- `std_qty`: Variability in purchase amounts
- `qty_cv`: Coefficient of variation (consistency metric)
- `max_qty`, `min_qty`: Purchase range

### Temporal Dynamics
- `recent_qty`: Average quantity in last 4 weeks
- `qty_trend`: Trend comparing recent vs. historical behavior
- `recent_purchase_rate`: Purchase frequency in recent weeks

### Order Patterns
- `mean_orders`: Average orders per week
- `max_orders`: Maximum orders in a week

### Monetary Features
- `total_spend`: Total historical spend
- `mean_spend`: Average spend per week
- `avg_price`: Average product price
- `price_volatility`: Price variation

### Customer Features
- `customer_lifetime_days`: Days since account creation
- `customer_category`: Encoded customer segment
- `customer_status`: Encoded account status
- `grade_name`: Encoded product grade
- `unit_name`: Encoded selling unit

## 🤖 Model Architecture

### Dual-Horizon Approach

The solution trains separate models for each prediction horizon:

```
├── 1-Week (7 days)
│   ├── Classifier: Predicts purchase probability
│   └── Regressor: Predicts quantity for purchasers
│
└── 2-Week (14 days)
    ├── Classifier: Predicts purchase probability
    └── Regressor: Predicts quantity for purchasers
```

### Model Details

**Classification (Purchase Probability)**
- Algorithm: Gradient Boosting Classifier
- Task: Binary classification (will purchase or not)
- Metric Optimized: Area Under the ROC Curve (AUC)
- Hyperparameters:
  - `n_estimators`: 150 boosting stages
  - `learning_rate`: 0.05 (conservative learning)
  - `max_depth`: 6 (tree depth)
  - `subsample`: 0.8 (stochastic boosting)

**Regression (Purchase Quantity)**
- Algorithm: Gradient Boosting Regressor
- Task: Continuous value prediction
- Metric Optimized: Mean Absolute Error (MAE)
- Training: Only on customers who actually purchased
- Constraints: Predictions clipped to non-negative values
- Hyperparameters: Same as classifier

### Why This Approach?

1. **Decoupled Tasks**: Treating purchase prediction and quantity prediction separately improves accuracy for each task
2. **Conditional Regression**: Quantity models only trained on positive examples (realistic quantities)
3. **Time-Specific Learning**: Different patterns emerge at 1-week vs 2-week horizons
4. **Imbalanced Data Handling**: Stratified train-test splits handle class imbalance

## 📈 Performance Metrics

The competition uses a **weighted multi-metric evaluation**:

```
Final Score = (AUC × 0.50) + (1 - MAE × 0.50)
```

Where:
- **AUC (50% weight)**: Measures classification quality for purchase prediction
  - Range: 0 to 1 (higher is better)
  - Interpretation: Probability that the model ranks a random positive example higher than a random negative example
  
- **MAE (50% weight)**: Measures quantity prediction accuracy
  - Range: 0 to ∞ (lower is better)
  - Interpretation: Average absolute difference between predicted and actual quantities

### Baseline Performance (Expected)
- **1-Week AUC**: 0.65-0.75
- **1-Week MAE**: 1.5-2.5 units
- **2-Week AUC**: 0.62-0.72
- **2-Week MAE**: 2.0-3.0 units

## 🔧 Hyperparameter Tuning

To improve performance, adjust these parameters in `train_models()`:

```python
# Gradient Boosting parameters
clf = GradientBoostingClassifier(
    n_estimators=150,      # More trees = more complex model
    learning_rate=0.05,    # Slower learning = more stable
    max_depth=6,           # Tree complexity
    subsample=0.8,         # Stochastic sampling
    min_samples_split=10,  # Minimum samples to split node
    random_state=42
)
```

### Tuning Tips
- **Increase `n_estimators`** if underfitting (high training error)
- **Decrease `learning_rate`** for smoother convergence but slower training
- **Increase `max_depth`** for more complex patterns (risk: overfitting)
- **Decrease `subsample`** for more regularization
- **Use `GridSearchCV`** for systematic hyperparameter search

## 🎯 Improvement Strategies

### 1. Feature Engineering
```python
# Add seasonal features
df['week_of_year'] = df['week_start'].dt.isocalendar().week
df['month'] = df['week_start'].dt.month

# Add interaction features
features['qty_price_interaction'] = features['mean_qty'] * features['avg_price']

# Add competitive features
features['product_popularity'] = df.groupby('product_id').size()
```

### 2. Model Ensembling
```python
from sklearn.ensemble import VotingClassifier

# Combine multiple models
voting_clf = VotingClassifier(
    estimators=[('gb', clf_gb), ('rf', clf_rf), ('xgb', clf_xgb)],
    voting='soft'
)
```

### 3. Class Imbalance Handling
```python
# Use class weights
clf = GradientBoostingClassifier(
    class_weight='balanced'  # or compute custom weights
)
```

### 4. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Keep only top K most important features
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)
```

## 📊 Visualization & Analysis

### Generate Feature Importance Plot
```python
import matplotlib.pyplot as plt

feature_importance = recommender.purchase_models['1w'].feature_importances_
top_features = sorted(zip(recommender.feature_cols, feature_importance), 
                      key=lambda x: x[1], reverse=True)[:15]

names, importance = zip(*top_features)
plt.barh(names, importance)
plt.xlabel('Importance')
plt.title('Top 15 Features - 1-Week Purchase Prediction')
plt.show()
```

### Prediction Distribution
```python
import matplotlib.pyplot as plt

plt.hist(predictions['Target_purchase_next_1w'], bins=50, edgecolor='black')
plt.xlabel('Purchase Probability')
plt.ylabel('Count')
plt.title('Distribution of Predicted Purchase Probabilities')
plt.show()
```

## ⚠️ Common Issues & Solutions

### Issue: Low AUC Score
**Possible Causes:**
- Features not informative enough
- Class imbalance not handled
- Model underfitting

**Solutions:**
- Engineer more domain-specific features
- Use `stratify` parameter in train-test split
- Increase `n_estimators` or `max_depth`

### Issue: High MAE for Quantity
**Possible Causes:**
- Training only on positive examples may be too limited
- Outliers in quantity distribution

**Solutions:**
- Include low-quantity purchases in training
- Apply log transformation to quantity
- Use robust regression (Huber regressor)

### Issue: Memory/Speed Issues
**Solutions:**
```python
# Use a subset of data
df_train = df_train.sample(frac=0.8, random_state=42)

# Reduce model complexity
n_estimators = 100  # instead of 150
max_depth = 4       # instead of 6
```

## 📝 Submission Guide

1. **Generate predictions** using the pipeline
2. **Verify output format**:
   - Correct column names
   - ID format: customer_id_product_unit_variant_id
   - Probabilities between 0 and 1
   - Quantities are non-negative
3. **Upload to competition platform** (e.g., Zindi, Kaggle)
4. **Monitor leaderboard** performance
5. **Iterate** based on feedback

## 🔄 Workflow

```
Data Loading
    ↓
Exploratory Analysis
    ↓
Feature Engineering
    ↓
Target Creation
    ↓
Train-Test Split
    ↓
Model Training
    ├── 1-Week Classifier
    ├── 1-Week Regressor
    ├── 2-Week Classifier
    └── 2-Week Regressor
    ↓
Prediction Generation
    ↓
Submission Export
```

## 📚 References

### Key Concepts
- **Time Series Forecasting**: Predicting future from historical patterns
- **Imbalanced Classification**: Handling unequal class distributions
- **Feature Engineering**: Domain knowledge for feature creation
- **Gradient Boosting**: Ensemble learning with sequential tree building

### Useful Libraries
- [scikit-learn Documentation](https://scikit-learn.org)
- [pandas User Guide](https://pandas.pydata.org/docs)
- [XGBoost](https://xgboost.readthedocs.io) (alternative to Gradient Boosting)
- [LightGBM](https://lightgbm.readthedocs.io) (fast gradient boosting)

## 🤝 Contributing

To improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request

### Ideas for Contribution
- Add SHAP feature importance analysis
- Implement cross-validation framework
- Create automated hyperparameter tuning
- Add data quality checks and profiling
- Build web API for predictions

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ✉️ Contact & Support

- **Issues**: Report bugs via GitHub issues
- **Discussions**: Start discussions for ideas and questions
- **Email**: simacoder@hotmail.com

---

**Made with ❤️ for sustainable agriculture and food security**

*Last Updated: 2024*
*Competition: Farm to Feed x Digital Africa ML Challenge*

# AUTHORS
- Simanga Mchunu
- Sinenhlahla Nsele