import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_preprocessing_pipeline():
    print("Initializing Logistics Data Preprocessing Pipeline...")
    
    # 1. Simulate Raw Logistics Dataset with Imperfections
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'order_id': [f'ORD-{i:04d}' for i in range(n_samples)],
        'distance_km': np.random.uniform(2.0, 30.0, n_samples),
        'package_weight_kg': np.random.uniform(0.5, 20.0, n_samples),
        'traffic_index': np.random.uniform(1.0, 5.0, n_samples),
        'delivery_time_hours': np.random.uniform(0.5, 4.0, n_samples)
    })

    # Introduce artificial missing values and duplicates for simulation
    df.loc[np.random.choice(n_samples, 30), 'delivery_time_hours'] = np.nan
    df = pd.concat([df, df.iloc[:10]], ignore_index=True) # Adding duplicates

    # 2. Remove Duplicates
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_count - len(df)} duplicate rows.")

    # 3. Handle Missing Values using Median Imputation
    print("\nMissing Values Before Imputation:\n", df.isnull().sum())
    median_time = df['delivery_time_hours'].median()
    df['delivery_time_hours'].fillna(median_time, inplace=True)
    print("Missing Values After Imputation:\n", df.isnull().sum())

    # 4. Outlier Detection and Capping (IQR Method)
    Q1 = df['distance_km'].quantile(0.25)
    Q3 = df['distance_km'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['distance_km'] = np.clip(df['distance_km'], lower_bound, upper_bound)
    print("\nOutliers treated via IQR capping on distance_km.")

    # 5. Feature Normalization / Standardization
    scaler = StandardScaler()
    features_to_scale = ['distance_km', 'package_weight_kg', 'traffic_index']
    df[[f'{col}_scaled' for col in features_to_scale]] = scaler.fit_transform(df[features_to_scale])
    print("\nFeature Standardization completed successfully!")
    print(df.head())

if __name__ == "__main__":
    run_preprocessing_pipeline()
