import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_and_explore_data():

    print("=== Loading Data ===")
    train = pd.read_csv('dataset.csv')
    test = pd.read_csv('test.csv')

    print(f"Training set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")

    print("\nTraining set columns:")
    print(train.columns.tolist())
    print("\nTest set columns:")
    print(test.columns.tolist())

    return train, test


def analyze_data_issues(train, test):
    print("\n=== Data Issue Analysis ===")

    issues = []
    insights = []


    print("1. Missing Value Analysis:")
    train_missing = train.isnull().sum()
    test_missing = test.isnull().sum()

    print("Training set missing values:")
    for col, missing in train_missing[train_missing > 0].items():
        percentage = (missing / len(train)) * 100
        print(f"  {col}: {missing} ({percentage:.2f}%)")
        issues.append(f"Training set column '{col}' has {missing} missing values ({percentage:.2f}%)")


        if percentage > 50:
            insights.append(
                f"Column '{col}' has high missing rate ({percentage:.2f}%), consider dropping or special treatment")
        elif percentage > 20:
            insights.append(f"Column '{col}' has moderate missing rate ({percentage:.2f}%), need careful imputation")

    print("Test set missing values:")
    for col, missing in test_missing[test_missing > 0].items():
        percentage = (missing / len(test)) * 100
        print(f"  {col}: {missing} ({percentage:.2f}%)")
        issues.append(f"Test set column '{col}' has {missing} missing values ({percentage:.2f}%)")


    print("\n2. Data Type Analysis:")
    print("Training set data types:")
    print(train.dtypes.value_counts())
    print("\nTest set data types:")
    print(test.dtypes.value_counts())

    print("\n3. Outlier Analysis:")
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    outlier_insights = []

    for col in numeric_cols:
        if col in train.columns:
            Q1 = train[col].quantile(0.25)
            Q3 = train[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = train[(train[col] < Q1 - 1.5 * IQR) | (train[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                percentage = (len(outliers) / len(train)) * 100
                print(f"  {col}: {len(outliers)} outliers ({percentage:.2f}%)")
                issues.append(f"Column '{col}' has {len(outliers)} outliers ({percentage:.2f}%)")

                if percentage > 10:
                    outlier_insights.append(
                        f"Column '{col}' has high outlier percentage ({percentage:.2f}%), may need robust scaling")
                elif percentage > 5:
                    outlier_insights.append(
                        f"Column '{col}' has moderate outliers ({percentage:.2f}%), consider outlier treatment")

    insights.extend(outlier_insights)


    train_duplicates = train.duplicated().sum()
    test_duplicates = test.duplicated().sum()
    print(f"\n4. Duplicate Analysis:")
    print(f"  Training set duplicates: {train_duplicates}")
    print(f"  Test set duplicates: {test_duplicates}")

    if train_duplicates > 0:
        issues.append(f"Training set has {train_duplicates} duplicate rows")
        insights.append(f"Found {train_duplicates} duplicate rows in training set, data quality issue")
    if test_duplicates > 0:
        issues.append(f"Test set has {test_duplicates} duplicate rows")
        insights.append(f"Found {test_duplicates} duplicate rows in test set, data quality issue")


    print("\n5. Feature Distribution Analysis:")
    skewness_insights = []
    for col in numeric_cols:
        if col in train.columns:
            skewness = train[col].skew()
            if abs(skewness) > 1:
                print(f"  {col}: skewness = {skewness:.2f} (skewed distribution)")
                issues.append(f"Column '{col}' is skewed (skewness: {skewness:.2f})")

                if abs(skewness) > 2:
                    skewness_insights.append(
                        f"Column '{col}' is highly skewed ({skewness:.2f}), consider log transformation")
                else:
                    skewness_insights.append(
                        f"Column '{col}' is moderately skewed ({skewness:.2f}), may need transformation")

    insights.extend(skewness_insights)

    return issues, insights


def visualize_data_issues(train, test):

    print("\n=== Creating Data Issue Visualizations ===")


    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Missing Value Analysis', fontsize=16, fontweight='bold')


    train_missing = train.isnull().sum()
    train_missing = train_missing[train_missing > 0].sort_values(ascending=False)
    axes[0].bar(range(len(train_missing)), train_missing.values, color='lightcoral')
    axes[0].set_title('Training Set Missing Values')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Missing Count')
    axes[0].set_xticks(range(len(train_missing)))
    axes[0].set_xticklabels(train_missing.index, rotation=45, ha='right')


    test_missing = test.isnull().sum()
    test_missing = test_missing[test_missing > 0].sort_values(ascending=False)
    axes[1].bar(range(len(test_missing)), test_missing.values, color='lightblue')
    axes[1].set_title('Test Set Missing Values')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Missing Count')
    axes[1].set_xticks(range(len(test_missing)))
    axes[1].set_xticklabels(test_missing.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


    numeric_cols = train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        n_cols = min(6, len(numeric_cols))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Numeric Feature Distribution and Outlier Detection', fontsize=16, fontweight='bold')

        for i, col in enumerate(numeric_cols[:n_cols]):
            row = i // 3
            col_idx = i % 3


            axes[row, col_idx].boxplot(train[col].dropna())
            axes[row, col_idx].set_title(f'{col} Distribution')
            axes[row, col_idx].set_ylabel('Values')


            Q1 = train[col].quantile(0.25)
            Q3 = train[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = train[(train[col] < Q1 - 1.5 * IQR) | (train[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                axes[row, col_idx].text(0.02, 0.98, f'Outliers: {len(outliers)}',
                                        transform=axes[row, col_idx].transAxes,
                                        verticalalignment='top', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        for i in range(n_cols, 6):
            row = i // 3
            col_idx = i % 3
            axes[row, col_idx].axis('off')

        plt.tight_layout()
        plt.savefig('numeric_features_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()


def clean_data(train, test):

    print("\n=== Starting Data Cleaning ===")

    train_original = train.copy()
    test_original = test.copy()


    print("1. Handling duplicates...")
    train_duplicates = train.duplicated().sum()
    test_duplicates = test.duplicated().sum()

    if train_duplicates > 0:
        train = train.drop_duplicates()
        print(f"  Removed {train_duplicates} duplicate rows from training set")

    if test_duplicates > 0:
        test = test.drop_duplicates()
        print(f"  Removed {test_duplicates} duplicate rows from test set")


    print("\n2. Handling missing values...")


    for dataset, name in [(train, 'Training'), (test, 'Test')]:
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        categorical_cols = dataset.select_dtypes(include=['object']).columns


        for col in numeric_cols:
            if dataset[col].isnull().sum() > 0:
                median_val = dataset[col].median()
                dataset[col].fillna(median_val, inplace=True)
                print(f"  {name} set column '{col}' filled with median {median_val:.2f}")


        for col in categorical_cols:
            if dataset[col].isnull().sum() > 0:
                mode_val = dataset[col].mode()[0]
                dataset[col].fillna(mode_val, inplace=True)
                print(f"  {name} set column '{col}' filled with mode '{mode_val}'")


    print("\n3. Handling outliers...")
    outliers_info = {}
    numeric_cols = train.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
        if len(outliers) > 0:
            outliers_info[col] = len(outliers)
            print(f"  Column '{col}' has {len(outliers)} outliers")

            train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)


    print("\n4. Encoding categorical variables...")
    label_encoders = {}


    all_categorical_cols = set()
    for dataset in [train, test]:
        all_categorical_cols.update(dataset.select_dtypes(include=['object']).columns)

    for col in all_categorical_cols:
        le = LabelEncoder()


        all_values = pd.concat([train[col], test[col]]).dropna().unique()
        le.fit(all_values)


        if col in train.columns:
            train[col] = le.transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))

        label_encoders[col] = le
        print(f"  Column '{col}' encoded")

    print(f"\nCleaning completed!")
    print(f"Cleaned training set shape: {train.shape}")
    print(f"Cleaned test set shape: {test.shape}")

    return train, test, label_encoders, outliers_info


def feature_engineering(train, test):

    print("\n=== Starting Feature Engineering ===")


    print("1. Creating new features...")


    numeric_cols = train.select_dtypes(include=[np.number]).columns


    for col in numeric_cols[:5]:
        if col in train.columns and col in test.columns:
            train[f'{col}_squared'] = train[col] ** 2
            test[f'{col}_squared'] = test[col] ** 2
            print(f"  Created {col}_squared feature")

    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        if col1 in train.columns and col2 in train.columns:
            train[f'{col1}_{col2}_interaction'] = train[col1] * train[col2]
            test[f'{col1}_{col2}_interaction'] = test[col1] * test[col2]
            print(f"  Created {col1}_{col2}_interaction feature")

    print("\n2. Feature scaling...")


    scaler = RobustScaler()


    all_numeric_cols = train.select_dtypes(include=[np.number]).columns


    common_cols = [col for col in all_numeric_cols if col in train.columns and col in test.columns]

    if len(common_cols) > 0:
        train[common_cols] = scaler.fit_transform(train[common_cols])
        test[common_cols] = scaler.transform(test[common_cols])
        print(f"  Scaled {len(common_cols)} numeric features")


    print("\n3. Feature selection...")

    target_col = None
    for col in train.columns:
        if 'price' in col.lower() or 'target' in col.lower() or 'label' in col.lower():
            target_col = col
            break

    if target_col and target_col in train.columns:

        feature_cols = [col for col in train.columns if col != target_col]


        selector = SelectKBest(score_func=f_regression, k=min(20, len(feature_cols)))
        X_selected = selector.fit_transform(train[feature_cols], train[target_col])


        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]


        train = train[selected_features + [target_col]]
        test = test[selected_features]

        print(f"  Selected {len(selected_features)} most important features")
        print(f"  Selected features: {selected_features}")

    print(f"\nFeature engineering completed!")
    print(f"Final training set shape: {train.shape}")
    print(f"Final test set shape: {test.shape}")

    return train, test, scaler


def visualize_cleaning_results(train_original, train_cleaned, outliers_info):

    print("\n=== Creating Cleaning Results Visualization ===")

    numeric_cols = train_original.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric features for comparison")
        return


    cols_to_plot = numeric_cols[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Cleaning Before vs After Comparison', fontsize=16, fontweight='bold')

    for i, col in enumerate(cols_to_plot):
        row = i // 3
        col_idx = i % 3


        axes[row, col_idx].hist(train_original[col].dropna(), bins=30, alpha=0.7,
                                color='lightblue', label='Before Cleaning', edgecolor='black')

        if col in train_cleaned.columns:
            axes[row, col_idx].hist(train_cleaned[col], bins=30, alpha=0.7,
                                    color='lightcoral', label='After Cleaning', edgecolor='black')

        axes[row, col_idx].set_title(f'{col} Before vs After Cleaning')
        axes[row, col_idx].set_xlabel('Values')
        axes[row, col_idx].set_ylabel('Frequency')
        axes[row, col_idx].legend()


        if col in outliers_info:
            axes[row, col_idx].text(0.02, 0.98, f'Outliers: {outliers_info[col]}',
                                    transform=axes[row, col_idx].transAxes,
                                    verticalalignment='top', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('cleaning_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_feature_importance(train):

    print("\n=== Creating Feature Importance Visualization ===")


    target_col = None
    for col in train.columns:
        if 'price' in col.lower() or 'target' in col.lower() or 'label' in col.lower():
            target_col = col
            break

    if not target_col:
        print("Target variable not found, skipping feature importance analysis")
        return


    feature_cols = [col for col in train.columns if col != target_col]
    correlations = []

    for col in feature_cols:
        corr = train[col].corr(train[target_col])
        correlations.append((col, abs(corr)))


    correlations.sort(key=lambda x: x[1], reverse=True)


    plt.figure(figsize=(12, 8))
    features, corrs = zip(*correlations[:15])  # Show top 15 features

    plt.barh(range(len(features)), corrs, color='skyblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Absolute Correlation with Target Variable')
    plt.title('Feature Importance (Based on Correlation)')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_cleaned_data(train, test):

    print("\n=== Saving Cleaned Data ===")
    train.to_csv('train_cleaned.csv', index=False)
    test.to_csv('test_cleaned.csv', index=False)
    print("Cleaned data saved as 'train_cleaned.csv' and 'test_cleaned.csv'")


def create_insights_report(issues, insights, train_original, train_cleaned, test_original, test_cleaned, outliers_info):

    print("\n=== Generating Data Insights Report ===")

    report = f""
    for issue in issues:
        report += f"- {issue}\n"

    report += f""

    for insight in insights:
        report += f"- {insight}\n"

    report += f""
    for col, count in outliers_info.items():
        report += f"- {col}: {count} outliers treated\n"

    report += f""
    with open('data_insights_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("Data insights report saved as 'data_insights_report.md'")
def main():

    print("Starting house price dataset cleaning, feature engineering, and visualization analysis...")

    train, test = load_and_explore_data()
    issues, insights = analyze_data_issues(train, test)
    visualize_data_issues(train, test)
    train_cleaned, test_cleaned, label_encoders, outliers_info = clean_data(train, test)
    train_final, test_final, scaler = feature_engineering(train_cleaned, test_cleaned)
    visualize_cleaning_results(train, train_final, outliers_info)
    visualize_feature_importance(train_final)
    save_cleaned_data(train_final, test_final)
    create_insights_report(issues, insights, train, train_final, test, test_final, outliers_info)

    print("\n=== Analysis Completed! ===")
    print("Generated files:")
    print("- train_cleaned.csv: Cleaned training set")
    print("- test_cleaned.csv: Cleaned test set")
    print("- missing_values_analysis.png: Missing value distribution")
    print("- numeric_features_distribution.png: Numeric feature box plots")
    print("- cleaning_comparison.png: Before vs after cleaning comparison")
    print("- feature_importance.png: Feature importance ranking")
    print("- data_insights_report.md: Detailed insights report")

    print("\n=== Key Data Insights ===")
    print("1. Data Quality Issues:")
    for i, issue in enumerate(issues[:5], 1):
        print(f"   {i}. {issue}")

    print("\n2. Business Recommendations:")
    for i, insight in enumerate(insights[:5], 1):
        print(f"   {i}. {insight}")

    print("\n3. Model Development Strategy:")
    print("   - Use the robust scaling for outlier features")
    print("   - Focus on features with high correlation to target")
    print("   - Consider tree-based models for better outlier handling")
    print("   - Implement cross-validation for data distribution differences")


if __name__ == "__main__":
    main()