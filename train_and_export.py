import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def prepare_and_train(csv_path: str = 'ncr_ride_bookings.csv'):
    """Load CSV, preprocess, train RandomForest and save artifacts.

    Artifacts saved to `artifacts/`:
      - model.joblib
      - encoders.joblib (dict of LabelEncoders for categorical cols)
      - le_status.joblib (LabelEncoder for Booking Status)
      - le_cust_reason.joblib (LabelEncoder for customer reason) or None
      - le_driver_reason.joblib (LabelEncoder for driver reason) or None

    Returns the trained model and encoder dict.
    """
    df = pd.read_csv(csv_path)

    # Drop columns similar to the notebook
    drop_cols = [
        'Date', 'Time', 'Booking ID', 'Customer ID',
        'Reason for cancelling by Customer', 'Driver Cancellation Reason',
        'Incomplete Rides Reason'
    ]

    df_model = df.drop([c for c in drop_cols if c in df.columns], axis=1, errors='ignore')

    categorical_cols = [
        'Booking Status', 'Vehicle Type', 'Pickup Location',
        'Drop Location', 'Payment Method'
    ]

    # Fit encoders
    encoders = {}
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = df_model[col].astype(str)
            le.fit(df_model[col])
            encoders[col] = le

    # Prepare X and y
    if 'Booking Status' not in df_model.columns:
        raise ValueError('Column `Booking Status` not found in CSV')

    X = df_model.drop('Booking Status', axis=1)
    y = encoders['Booking Status'].transform(df_model['Booking Status'].astype(str))

    # For simplicity, convert any remaining non-numeric columns using LabelEncoder
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[f'__auto__{col}'] = le

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle imbalance with SMOTE if needed
    try:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    except Exception:
        # If SMOTE fails for some reason, fallback to original
        X_train_res, y_train_res = X_train, y_train

    # Train RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)

    # Save artifacts
    joblib.dump(rf, ARTIFACTS_DIR / 'model.joblib')
    joblib.dump(encoders, ARTIFACTS_DIR / 'encoders.joblib')
    joblib.dump(encoders.get('Booking Status'), ARTIFACTS_DIR / 'le_status.joblib')

    # Train and save Customer Cancellation Reason model if there is data
    if 'Reason for cancelling by Customer' in df.columns:
        df_cust = df[df['Booking Status'] == 'Cancelled by Customer'].copy()
        if not df_cust.empty:
            # Prepare features for reason model using the same feature columns as main X
            df_cust_feat = df_cust.reindex(columns=X.columns, fill_value=0).copy()

            # Encode features using existing encoders or fit new ones if needed
            for col in df_cust_feat.columns:
                if df_cust_feat[col].dtype == object or df_cust_feat[col].dtype.name == 'category':
                    # Prefer existing encoder
                    if col in encoders:
                        le = encoders[col]
                        df_cust_feat[col] = le.transform(df_cust_feat[col].astype(str))
                    elif f'__auto__{col}' in encoders:
                        le = encoders[f'__auto__{col}']
                        df_cust_feat[col] = le.transform(df_cust_feat[col].astype(str))
                    else:
                        le = LabelEncoder()
                        df_cust_feat[col] = le.fit_transform(df_cust_feat[col].astype(str))
                        encoders[f'__auto__{col}'] = le

            y_cust = LabelEncoder()
            y_cust.fit(df_cust['Reason for cancelling by Customer'].astype(str))
            y_cust_enc = y_cust.transform(df_cust['Reason for cancelling by Customer'].astype(str))

            rf_cust = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            try:
                rf_cust.fit(df_cust_feat, y_cust_enc)
                joblib.dump(rf_cust, ARTIFACTS_DIR / 'rf_cust_reason.joblib')
                joblib.dump(y_cust, ARTIFACTS_DIR / 'le_cust_reason.joblib')
            except Exception:
                # If training fails, still save encoder placeholder
                joblib.dump(None, ARTIFACTS_DIR / 'rf_cust_reason.joblib')
                joblib.dump(y_cust, ARTIFACTS_DIR / 'le_cust_reason.joblib')
        else:
            joblib.dump(None, ARTIFACTS_DIR / 'rf_cust_reason.joblib')
            joblib.dump(None, ARTIFACTS_DIR / 'le_cust_reason.joblib')
    else:
        joblib.dump(None, ARTIFACTS_DIR / 'rf_cust_reason.joblib')
        joblib.dump(None, ARTIFACTS_DIR / 'le_cust_reason.joblib')

    # Train and save Driver Cancellation Reason model if there is data
    if 'Driver Cancellation Reason' in df.columns:
        df_drv = df[df['Booking Status'] == 'Cancelled by Driver'].copy()
        if not df_drv.empty:
            df_drv_feat = df_drv.reindex(columns=X.columns, fill_value=0).copy()

            for col in df_drv_feat.columns:
                if df_drv_feat[col].dtype == object or df_drv_feat[col].dtype.name == 'category':
                    if col in encoders:
                        le = encoders[col]
                        df_drv_feat[col] = le.transform(df_drv_feat[col].astype(str))
                    elif f'__auto__{col}' in encoders:
                        le = encoders[f'__auto__{col}']
                        df_drv_feat[col] = le.transform(df_drv_feat[col].astype(str))
                    else:
                        le = LabelEncoder()
                        df_drv_feat[col] = le.fit_transform(df_drv_feat[col].astype(str))
                        encoders[f'__auto__{col}'] = le

            y_drv = LabelEncoder()
            y_drv.fit(df_drv['Driver Cancellation Reason'].astype(str))
            y_drv_enc = y_drv.transform(df_drv['Driver Cancellation Reason'].astype(str))

            rf_drv = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            try:
                rf_drv.fit(df_drv_feat, y_drv_enc)
                joblib.dump(rf_drv, ARTIFACTS_DIR / 'rf_driver_reason.joblib')
                joblib.dump(y_drv, ARTIFACTS_DIR / 'le_driver_reason.joblib')
            except Exception:
                joblib.dump(None, ARTIFACTS_DIR / 'rf_driver_reason.joblib')
                joblib.dump(y_drv, ARTIFACTS_DIR / 'le_driver_reason.joblib')
        else:
            joblib.dump(None, ARTIFACTS_DIR / 'rf_driver_reason.joblib')
            joblib.dump(None, ARTIFACTS_DIR / 'le_driver_reason.joblib')
    else:
        joblib.dump(None, ARTIFACTS_DIR / 'rf_driver_reason.joblib')
        joblib.dump(None, ARTIFACTS_DIR / 'le_driver_reason.joblib')

    print(f"Artifacts saved to {ARTIFACTS_DIR.resolve()}")
    return rf, encoders


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ncr_ride_bookings.csv', help='Path to CSV file')
    args = parser.parse_args()

    prepare_and_train(args.csv)

# End of train_and_export.py