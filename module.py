import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('ignore')


from data import (
    load_and_explore_data,
    clean_data,
    feature_engineering,
    save_cleaned_data
)


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PredictionIntervalModel:


    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.models = {}
        self.feature_names = None
        self.scaler = None
        self.label_encoders = {}

    def prepare_data(self):
        print("=== Preparing Data for Prediction Intervals ===")

        train, test = load_and_explore_data()
        train_cleaned, test_cleaned, label_encoders, outliers_info = clean_data(train, test)
        train_final, test_final, scaler = feature_engineering(train_cleaned, test_cleaned)

        target_col = 'sale_price'
        X_train = train_final.drop([target_col], axis=1, errors='ignore')
        y_train = train_final[target_col]
        X_test = test_final

        print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}")

        self.feature_names = X_train.columns.tolist()
        self.scaler = scaler
        self.label_encoders = label_encoders

        return X_train, y_train, X_test

    def create_uncertainty_models(self, fast_mode=False):
        print("\n=== Creating Uncertainty Estimation Models ===")

        if fast_mode:
            self.models = {
                'XGBoost': XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                ),
                'LightGBM': LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    num_leaves=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    force_col_wise=True,
                    min_child_samples=20,
                    min_child_weight=1e-3,
                    reg_alpha=0.1,
                    reg_lambda=0.1
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            }
        else:
            self.models = {
                'XGBoost': XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                ),
                'LightGBM': LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    force_col_wise=True,
                    min_child_samples=20,
                    min_child_weight=1e-3,
                    reg_alpha=0.1,
                    reg_lambda=0.1
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'Extra Trees': ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            }

        return self.models

    def optimize_interval_width(self, pi_lower, pi_upper, y_train):
        print("\n=== Optimizing Prediction Interval Width ===")

        interval_width = pi_upper - pi_lower
        mean_width = np.mean(interval_width)

        print(f"Current mean interval width: ${mean_width:,.0f}")
        target_percentiles = np.percentile(y_train, [5, 95])
        target_range = target_percentiles[1] - target_percentiles[0]
        optimal_width_ratio = 0.15
        center = (pi_lower + pi_upper) / 2
        optimal_width = target_range * optimal_width_ratio

        pi_lower_opt = center - optimal_width / 2
        pi_upper_opt = center + optimal_width / 2

        pi_lower_opt = np.maximum(pi_lower_opt, 0)

        print(f"Optimized mean interval width: ${np.mean(pi_upper_opt - pi_lower_opt):,.0f}")

        return pi_lower_opt, pi_upper_opt


    def save_predictions(self, pi_lower, pi_upper):
        print("\n=== Saving Prediction Intervals ===")
        submission = pd.read_csv('submission.csv')
        predictions_df = pd.DataFrame({
            'id': submission['id'],
            'pi_lower': pi_lower,
            'pi_upper': pi_upper
        })

        predictions_df.to_csv('submission.csv', index=False)

        print("Prediction intervals saved to submission.csv")
        print(f"Submission file contains {len(predictions_df)} prediction intervals")

        print(f"\nPrediction interval statistics:")
        print(f"  Mean interval width: ${np.mean(pi_upper - pi_lower):,.0f}")
        print(f"  Median interval width: ${np.median(pi_upper - pi_lower):,.0f}")
        print(f"  Lower bound range: ${np.min(pi_lower):,.0f} - ${np.max(pi_lower):,.0f}")
        print(f"  Upper bound range: ${np.min(pi_upper):,.0f} - ${np.max(pi_upper):,.0f}")

        return predictions_df

    def visualize_prediction_intervals(self, pi_lower, pi_upper, y_train):
        print("\n=== Creating Prediction Interval Visualizations ===")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Prediction Interval Analysis', fontsize=16, fontweight='bold')
        interval_widths = pi_upper - pi_lower
        axes[0, 0].hist(interval_widths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Prediction Interval Width Distribution')
        axes[0, 0].set_xlabel('Interval Width ($)')
        axes[0, 0].set_ylabel('Frequency')
        interval_centers = (pi_lower + pi_upper) / 2
        axes[0, 1].hist(interval_centers, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Prediction Interval Center Distribution')
        axes[0, 1].set_xlabel('Interval Center ($)')
        axes[0, 1].set_ylabel('Frequency')
        sample_size = min(100, len(pi_lower))
        sample_indices = np.random.choice(len(pi_lower), sample_size, replace=False)

        axes[1, 0].scatter(range(sample_size), interval_centers[sample_indices],
                           alpha=0.6, color='blue', label='Interval Centers')
        axes[1, 0].fill_between(range(sample_size),
                                pi_lower[sample_indices],
                                pi_upper[sample_indices],
                                alpha=0.3, color='gray', label='Prediction Intervals')
        axes[1, 0].set_title('Sample Prediction Intervals')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 1].scatter(interval_centers, interval_widths, alpha=0.5, color='green')
        axes[1, 1].set_title('Interval Width vs Center')
        axes[1, 1].set_xlabel('Interval Center ($)')
        axes[1, 1].set_ylabel('Interval Width ($)')

        plt.tight_layout()
        plt.savefig('prediction_intervals_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def improved_approach(self, X_train, y_train, X_test):
        print("\n=== Improved Approach===")
        
        print("Training ensemble of models...")
        
        models = {
            'xgb1': XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'xgb2': XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=123,
                n_jobs=-1,
                verbosity=0
            ),
            'lgb1': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'lgb2': LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=8,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=123,
                n_jobs=-1,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }
        
        ensemble = VotingRegressor(
            estimators=[(name, model) for name, model in models.items()],
            weights=[0.25, 0.25, 0.2, 0.2, 0.1]
        )
        
        ensemble.fit(X_train, y_train)
        mean_pred = ensemble.predict(X_test)
        
        print("Calculating uncertainty using cross-validation...")
        
        cv_predictions = []
        for name, model in models.items():
            try:
                cv_pred = cross_val_predict(model, X_train, y_train, cv=5, n_jobs=-1)
                cv_predictions.append(cv_pred)
            except Exception as e:
                print(f"CV failed for {name}: {str(e)}")
                continue
        
        if cv_predictions:
            cv_pred_array = np.array(cv_predictions)
            cv_std = np.std(cv_pred_array, axis=0)
            mean_cv_std = np.mean(cv_std)
            
            target_std = y_train.std()
            cv_uncertainty = np.maximum(mean_cv_std, target_std * 0.1)
        else:
            target_std = y_train.std()
            cv_uncertainty = target_std * 0.15
        
        z_score = 1.96
        pi_lower = mean_pred - z_score * cv_uncertainty
        pi_upper = mean_pred + z_score * cv_uncertainty
        
        pi_lower = np.maximum(pi_lower, 0)
        pi_upper = np.maximum(pi_upper, pi_lower + 1000)
        
        print("Improved prediction completed!")
        return pi_lower, pi_upper

    def adaptive_uncertainty_approach(self, X_train, y_train, X_test):
        print("\n=== Adaptive Uncertainty Approach ===")
        
        print("Training model and calculating adaptive uncertainty...")
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        mean_pred = model.predict(X_test)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        mean_cv_rmse = np.mean(cv_rmse)
        
        target_std = y_train.std()
        target_mean = y_train.mean()
        
        adaptive_uncertainty = []
        for pred in mean_pred:
            if pred < target_mean:
                uncertainty = mean_cv_rmse * 1.5
            else:
                uncertainty = mean_cv_rmse * 1.0
            adaptive_uncertainty.append(uncertainty)

        adaptive_uncertainty = np.array(adaptive_uncertainty)
        
        z_score = 1.96
        pi_lower = mean_pred - z_score * adaptive_uncertainty
        pi_upper = mean_pred + z_score * adaptive_uncertainty
        
        pi_lower = np.maximum(pi_lower, 0)
        pi_upper = np.maximum(pi_upper, pi_lower + 1000)
        
        print("Adaptive uncertainty prediction completed!")
        return pi_lower, pi_upper

    def ultra_fast_approach(self, X_train, y_train, X_test):
        print("\n=== Ultra Fast Approach ===")
        
        model = XGBRegressor(
            n_estimators=50,
            learning_rate=0.2,
            max_depth=3,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        print("Training single XGBoost model...")
        model.fit(X_train, y_train)
        mean_pred = model.predict(X_test)
        
        target_std = y_train.std()
        uncertainty = target_std * 0.15
        
        z_score = 1.96
        pi_lower = mean_pred - z_score * uncertainty
        pi_upper = mean_pred + z_score * uncertainty
        
        pi_lower = np.maximum(pi_lower, 0)
        pi_upper = np.maximum(pi_upper, pi_lower + 1000)
        
        print("Ultra fast prediction completed!")
        return pi_lower, pi_upper

    def evaluate_prediction_quality(self, pi_lower, pi_upper, y_true=None):
        print("\n=== Evaluating Prediction Interval Quality ===")
        
        interval_width = pi_upper - pi_lower
        interval_center = (pi_lower + pi_upper) / 2
        
        print(f"Interval Statistics:")
        print(f"  Mean width: ${np.mean(interval_width):,.0f}")
        print(f"  Median width: ${np.median(interval_width):,.0f}")
        print(f"  Width std: ${np.std(interval_width):,.0f}")
        print(f"  Min width: ${np.min(interval_width):,.0f}")
        print(f"  Max width: ${np.max(interval_width):,.0f}")
        
        print(f"\nCenter Statistics:")
        print(f"  Mean center: ${np.mean(interval_center):,.0f}")
        print(f"  Center std: ${np.std(interval_center):,.0f}")
        print(f"  Min center: ${np.min(interval_center):,.0f}")
        print(f"  Max center: ${np.max(interval_center):,.0f}")
        
        if y_true is not None:
            coverage = np.mean((y_true >= pi_lower) & (y_true <= pi_upper))
            print(f"\nCoverage Analysis:")
            print(f"  Coverage rate: {coverage:.3f}")
            print(f"  Target coverage: {self.confidence_level:.3f}")
            
            within_bounds = (y_true >= pi_lower) & (y_true <= pi_upper)
            print(f"  Samples within bounds: {np.sum(within_bounds)}/{len(y_true)}")
            
            if coverage < self.confidence_level:
                print(f"  WARNING: Coverage below target! Consider adjusting uncertainty estimation.")
        
        return {
            'mean_width': np.mean(interval_width),
            'mean_center': np.mean(interval_center),
            'coverage': coverage if y_true is not None else None
        }

    def generate_interval_report(self, coverage_rate=None):
        print("\n=== Generating Prediction Interval Report ===")

def main(fast_mode=True, ultra_fast_mode=False, improved_mode=False):
    print("=== House Price Prediction Interval Model (Optimized) ===")
    if improved_mode:
        print("Running in IMPROVED MODE for better accuracy")
    elif ultra_fast_mode:
        print("Running in ULTRA FAST MODE for maximum speed")
    elif fast_mode:
        print("Running in FAST MODE for improved performance")
    else:
        print("Running in FULL MODE for maximum accuracy")
    predictor = PredictionIntervalModel(confidence_level=0.95)
    try:
        X_train, y_train, X_test = predictor.prepare_data()
        if improved_mode:
            print("\n=== Generating Final Prediction Intervals (Improved) ===")
            pi_lower, pi_upper = predictor.improved_approach(X_train, y_train, X_test)
        elif ultra_fast_mode:
            print("\n=== Generating Final Prediction Intervals (Ultra Fast) ===")
            pi_lower, pi_upper = predictor.ultra_fast_approach(X_train, y_train, X_test)
        else:
            print("\n===module are not ready!please check it===")
        if pi_lower is not None:
            pi_lower_opt, pi_upper_opt = predictor.optimize_interval_width(pi_lower, pi_upper, y_train)
            quality_metrics = predictor.evaluate_prediction_quality(pi_lower_opt, pi_upper_opt)
            submission_df = predictor.save_predictions(pi_lower_opt, pi_upper_opt)
            predictor.visualize_prediction_intervals(pi_lower_opt, pi_upper_opt, y_train)
            print("\n=== Prediction Interval Pipeline Completed Successfully! ===")
            print("Generated files:")
            print("- submission.csv: Prediction intervals")
            print("- prediction_intervals_analysis.png: Interval visualizations")
            return predictor, submission_df
        else:
            print("Failed to generate prediction intervals!")
            return predictor, None
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    predictor, submission = main(improved_mode=True)