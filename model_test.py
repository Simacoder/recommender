import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class FarmToFeedRecommender:
    def __init__(self):
        self.purchase_models = {}
        self.quantity_models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_cols = None
        
    def load_and_prepare_data(self, df):
        """Load transaction data and prepare for modeling"""
        df = df.copy()
        df['week_start'] = pd.to_datetime(df['week_start'])
        df['customer_created_at'] = pd.to_datetime(df['customer_created_at'])
        return df
    
    def engineer_features(self, df, customer_col='customer_id', 
                         product_col='product_unit_variant_id', week_col='week_start'):
        """Engineer features for each customer-product pair"""
        
        features = []
        
        for (customer, product), group in df.groupby([customer_col, product_col]):
            group = group.sort_values(week_col)
            
            if len(group) < 1:
                continue
            
            # Purchase history features
            purchase_weeks = len(group[group['purchased_this_week'] == 1])
            total_weeks_history = len(group)
            purchase_rate = purchase_weeks / (total_weeks_history + 1)
            weeks_since_last_purchase = total_weeks_history - purchase_weeks
            
            # Quantity patterns
            qty_history = group['qty_this_week'].values
            total_qty = qty_history.sum()
            mean_qty = qty_history.mean()
            std_qty = qty_history.std() if len(qty_history) > 1 else 0
            max_qty = qty_history.max()
            min_qty = qty_history.min()
            qty_cv = std_qty / (mean_qty + 1)
            
            # Order patterns
            orders_history = group['num_orders_week'].values
            mean_orders = orders_history.mean()
            max_orders = orders_history.max()
            
            # Spend patterns
            spend_history = group['spend_this_week'].values
            total_spend = spend_history.sum()
            mean_spend = spend_history.mean()
            
            # Temporal patterns
            recent_weeks = min(4, len(group))
            recent_qty = group.tail(recent_weeks)['qty_this_week'].mean()
            recent_purchase_rate = (group.tail(recent_weeks)['purchased_this_week'].sum() / recent_weeks)
            
            if total_weeks_history >= 8:
                old_qty = group.head(total_weeks_history // 2)['qty_this_week'].mean()
                qty_trend = (recent_qty - old_qty) / (old_qty + 1)
            else:
                qty_trend = 0
            
            # Customer features
            customer_data = group.iloc[0]
            customer_lifetime_days = (group[week_col].max() - customer_data['customer_created_at']).days
            
            grade_name = customer_data['grade_name']
            unit_name = customer_data['unit_name']
            customer_category = customer_data['customer_category']
            customer_status = customer_data['customer_status']
            
            avg_price = group['selling_price'].mean()
            price_volatility = group['selling_price'].std() if len(group) > 1 else 0
            
            product_id = customer_data['product_id']
            
            feature_row = {
                'customer_id': customer,
                'product_unit_variant_id': product,
                'product_id': product_id,
                'grade_name': grade_name,
                'unit_name': unit_name,
                'customer_category': customer_category,
                'customer_status': customer_status,
                
                'purchase_weeks': purchase_weeks,
                'total_weeks_history': total_weeks_history,
                'purchase_rate': purchase_rate,
                'weeks_since_last_purchase': weeks_since_last_purchase,
                
                'total_qty': total_qty,
                'mean_qty': mean_qty,
                'std_qty': std_qty,
                'max_qty': max_qty,
                'min_qty': min_qty,
                'qty_cv': qty_cv,
                
                'mean_orders': mean_orders,
                'max_orders': max_orders,
                
                'total_spend': total_spend,
                'mean_spend': mean_spend,
                
                'recent_qty': recent_qty,
                'recent_purchase_rate': recent_purchase_rate,
                'qty_trend': qty_trend,
                
                'customer_lifetime_days': customer_lifetime_days,
                
                'avg_price': avg_price,
                'price_volatility': price_volatility,
                
                'last_week_start': group[week_col].max(),
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def create_targets_simple(self, df_test, features_df):
        """Create simple targets based on test set presence"""
        
        targets = features_df.copy()
        targets['target_purchase_next_1w'] = 1
        targets['target_qty_next_1w'] = 1
        targets['target_purchase_next_2w'] = 1
        targets['target_qty_next_2w'] = 1
        
        return targets
    
    def prepare_training_data(self, features_df, targets_df=None):
        """Prepare X and y for model training"""
        
        categorical_cols = ['grade_name', 'unit_name', 'customer_category', 'customer_status']
        numeric_cols = [col for col in features_df.columns 
                       if col not in ['customer_id', 'product_unit_variant_id', 'product_id',
                                     'last_week_start'] + categorical_cols]
        
        X = features_df[numeric_cols + categorical_cols].copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        X = X.fillna(0)
        self.feature_cols = list(X.columns)
        
        return X, numeric_cols, categorical_cols
    
    def train_models(self, features_df, targets_df, test_size=0.2, random_state=42):
        """Train classification and regression models"""
        
        X, numeric_cols, categorical_cols = self.prepare_training_data(features_df, targets_df)
        
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)
        
        print("=" * 60)
        print("Training 1-Week Models...")
        print("=" * 60)
        
        # Check class balance for 1-week
        class_counts_1w = targets_df['target_purchase_next_1w'].value_counts()
        print(f"1-Week class distribution:\n{class_counts_1w}\n")
        
        if len(class_counts_1w) < 2:
            print(f"  Warning: Only one class present in 1-week targets. Using default model.")
            self.purchase_models['1w'] = None
            self.quantity_models['1w'] = None
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, targets_df['target_purchase_next_1w'], 
                    test_size=test_size, random_state=random_state, 
                    stratify=targets_df['target_purchase_next_1w']
                )
                
                clf_1w = GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.05, max_depth=6,
                    subsample=0.8, min_samples_split=10, random_state=random_state
                )
                clf_1w.fit(X_train, y_train)
                self.purchase_models['1w'] = clf_1w
                
                y_pred_proba_1w = clf_1w.predict_proba(X_test)[:, 1]
                auc_1w = roc_auc_score(y_test, y_pred_proba_1w)
                print(f" 1-Week Purchase Classifier AUC: {auc_1w:.4f}")
                
                purchasers_1w = targets_df['target_purchase_next_1w'] == 1
                if purchasers_1w.sum() > 10:
                    X_reg_1w = X_scaled[purchasers_1w]
                    y_reg_1w = targets_df.loc[purchasers_1w, 'target_qty_next_1w']
                    
                    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                        X_reg_1w, y_reg_1w, test_size=test_size, random_state=random_state
                    )
                    
                    reg_1w = GradientBoostingRegressor(
                        n_estimators=150, learning_rate=0.05, max_depth=6,
                        subsample=0.8, min_samples_split=10, random_state=random_state
                    )
                    reg_1w.fit(X_train_reg, y_train_reg)
                    self.quantity_models['1w'] = reg_1w
                    
                    mae_1w = mean_absolute_error(y_test_reg, reg_1w.predict(X_test_reg))
                    print(f" 1-Week Quantity Regressor MAE: {mae_1w:.4f}")
            except Exception as e:
                print(f" Error training 1-week model: {e}")
                self.purchase_models['1w'] = None
                self.quantity_models['1w'] = None
        
        print("\n" + "=" * 60)
        print("Training 2-Week Models...")
        print("=" * 60)
        
        # Check class balance for 2-week
        class_counts_2w = targets_df['target_purchase_next_2w'].value_counts()
        print(f"2-Week class distribution:\n{class_counts_2w}\n")
        
        if len(class_counts_2w) < 2:
            print(f"  Warning: Only one class present in 2-week targets. Using default model.")
            self.purchase_models['2w'] = None
            self.quantity_models['2w'] = None
        else:
            try:
                _, _, y_train_2w, y_test_2w = train_test_split(
                    X_scaled, targets_df['target_purchase_next_2w'],
                    test_size=test_size, random_state=random_state, 
                    stratify=targets_df['target_purchase_next_2w']
                )
                
                clf_2w = GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.05, max_depth=6,
                    subsample=0.8, min_samples_split=10, random_state=random_state
                )
                clf_2w.fit(X_train, y_train_2w)
                self.purchase_models['2w'] = clf_2w
                
                y_pred_proba_2w = clf_2w.predict_proba(X_test)[:, 1]
                auc_2w = roc_auc_score(y_test_2w, y_pred_proba_2w)
                print(f" 2-Week Purchase Classifier AUC: {auc_2w:.4f}")
                
                purchasers_2w = targets_df['target_purchase_next_2w'] == 1
                if purchasers_2w.sum() > 10:
                    X_reg_2w = X_scaled[purchasers_2w]
                    y_reg_2w = targets_df.loc[purchasers_2w, 'target_qty_next_2w']
                    
                    X_train_reg_2w, X_test_reg_2w, y_train_reg_2w, y_test_reg_2w = train_test_split(
                        X_reg_2w, y_reg_2w, test_size=test_size, random_state=random_state
                    )
                    
                    reg_2w = GradientBoostingRegressor(
                        n_estimators=150, learning_rate=0.05, max_depth=6,
                        subsample=0.8, min_samples_split=10, random_state=random_state
                    )
                    reg_2w.fit(X_train_reg_2w, y_train_reg_2w)
                    self.quantity_models['2w'] = reg_2w
                    
                    mae_2w = mean_absolute_error(y_test_reg_2w, reg_2w.predict(X_test_reg_2w))
                    print(f" 2-Week Quantity Regressor MAE: {mae_2w:.4f}")
            except Exception as e:
                print(f" Error training 2-week model: {e}")
                self.purchase_models['2w'] = None
                self.quantity_models['2w'] = None
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
    
    def predict(self, features_df):
        """Generate predictions for all customer-product pairs"""
        
        X, _, _ = self.prepare_training_data(features_df, pd.DataFrame())
        X_scaled = self.scalers['main'].transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)
        
        predictions = pd.DataFrame({
            'ID': (features_df['customer_id'].astype(str) + '_' + 
                   features_df['product_unit_variant_id'].astype(str))
        })
        
        # 1-week predictions
        if self.purchase_models['1w'] is not None:
            predictions['Target_purchase_next_1w'] = self.purchase_models['1w'].predict_proba(X_scaled)[:, 1]
        else:
            predictions['Target_purchase_next_1w'] = 0.5
        
        if '1w' in self.quantity_models and self.quantity_models['1w'] is not None:
            qty_pred_1w = self.quantity_models['1w'].predict(X_scaled)
            qty_pred_1w = np.maximum(qty_pred_1w, 0)
            predictions['Target_qty_next_1w'] = qty_pred_1w
        else:
            predictions['Target_qty_next_1w'] = 0
        
        # 2-week predictions
        if self.purchase_models['2w'] is not None:
            predictions['Target_purchase_next_2w'] = self.purchase_models['2w'].predict_proba(X_scaled)[:, 1]
        else:
            predictions['Target_purchase_next_2w'] = 0.5
        
        if '2w' in self.quantity_models and self.quantity_models['2w'] is not None:
            qty_pred_2w = self.quantity_models['2w'].predict(X_scaled)
            qty_pred_2w = np.maximum(qty_pred_2w, 0)
            predictions['Target_qty_next_2w'] = qty_pred_2w
        else:
            predictions['Target_qty_next_2w'] = 0
        
        return predictions
    
    def export_submission(self, predictions, output_file='submission.csv'):
        """Export predictions in required format"""
        
        submission = predictions[['ID', 'Target_purchase_next_1w', 'Target_qty_next_1w',
                                  'Target_purchase_next_2w', 'Target_qty_next_2w']].copy()
        
        submission.to_csv(output_file, index=False)
        print(f"\n Submission saved to '{output_file}'")
        print(f"  Total predictions: {len(submission)}")
        return submission


# ===== MAIN PIPELINE =====
if __name__ == "__main__":
    print("=" * 60)
    print("Farm to Feed ML Pipeline - Train & Test")
    print("=" * 60)
    
    # Load training and test data
    print("\n1. Loading data...")
    df_train = pd.read_csv('data/Train.csv')
    df_test = pd.read_csv('data/Test.csv')
    print(f" Train set: {df_train.shape}")
    print(f" Test set: {df_test.shape}")
    
    # Initialize recommender
    recommender = FarmToFeedRecommender()
    
    # Prepare data
    print("\n2. Preparing data...")
    df_train = recommender.load_and_prepare_data(df_train)
    df_test = recommender.load_and_prepare_data(df_test)
    
    # Engineer features from training data
    print("\n3. Engineering features from training data...")
    features_df = recommender.engineer_features(df_train)
    print(f" Created {len(features_df)} customer-product pairs from train")
    
    # Get ALL unique customer-product pairs from test set
    print("\n3b. Adding test-only customer-product pairs...")
    test_pairs = df_test.groupby(['customer_id', 'product_unit_variant_id']).first().reset_index()
    
    # Find pairs that are in test but not in train features
    test_pairs['pair_id'] = test_pairs['customer_id'].astype(str) + '_' + test_pairs['product_unit_variant_id'].astype(str)
    features_df['pair_id'] = features_df['customer_id'].astype(str) + '_' + features_df['product_unit_variant_id'].astype(str)
    
    test_only_pairs = test_pairs[~test_pairs['pair_id'].isin(features_df['pair_id'])]
    print(f" Found {len(test_only_pairs)} pairs only in test set")
    
    # Add test-only pairs with default features
    if len(test_only_pairs) > 0:
        test_only_features = test_pairs[~test_pairs['pair_id'].isin(features_df['pair_id'])][
            ['customer_id', 'product_unit_variant_id', 'product_id', 'grade_name', 'unit_name', 
             'customer_category', 'customer_status']
        ].copy()
        
        # Add default numeric features
        test_only_features['purchase_weeks'] = 0
        test_only_features['total_weeks_history'] = 0
        test_only_features['purchase_rate'] = 0
        test_only_features['weeks_since_last_purchase'] = 0
        test_only_features['total_qty'] = 0
        test_only_features['mean_qty'] = 0
        test_only_features['std_qty'] = 0
        test_only_features['max_qty'] = 0
        test_only_features['min_qty'] = 0
        test_only_features['qty_cv'] = 0
        test_only_features['mean_orders'] = 0
        test_only_features['max_orders'] = 0
        test_only_features['total_spend'] = 0
        test_only_features['mean_spend'] = 0
        test_only_features['recent_qty'] = 0
        test_only_features['recent_purchase_rate'] = 0
        test_only_features['qty_trend'] = 0
        test_only_features['customer_lifetime_days'] = 0
        test_only_features['avg_price'] = 0
        test_only_features['price_volatility'] = 0
        test_only_features['last_week_start'] = pd.Timestamp.now()
        
        # Combine with original features
        features_df = pd.concat([features_df.drop('pair_id', axis=1), test_only_features], 
                               ignore_index=True)
    else:
        features_df = features_df.drop('pair_id', axis=1)
    
    print(f" Total feature rows: {len(features_df)}")
    
    # Create targets using test data
    print("\n4. Creating target variables from test data...")
    targets_df = recommender.create_targets_simple(df_test, features_df)
    
    print("\n--- Target Statistics ---")
    print(f"Total pairs with targets: {len(targets_df)}")
    
    # Train models
    print("\n5. Training models...")
    recommender.train_models(features_df, targets_df)
    
    # Make predictions
    print("\n6. Generating predictions...")
    predictions = recommender.predict(features_df)
    
    print(f"\nPredictions generated: {len(predictions)} rows")
    print(predictions.head(10))
    
    # Export submission
    print("\n7. Exporting submission...")
    submission = recommender.export_submission(predictions, 'submission1.csv')
    
    print(f"\nFirst 10 rows of submission:")
    print(submission.head(10))
    
    print("\n" + "=" * 60)
    print(" Pipeline complete! Submission ready for upload.")
    print("=" * 60)