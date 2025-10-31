import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import os
import requests


def _download_url_to_file(url: str, dest: Path):
    """Download a URL to the destination path in streaming mode."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _find_artifact_url(name: str):
    """Try to find a download URL for an artifact.

    Search order:
      1. Environment variable MODEL_URL_<NAME_UPPER>
      2. Environment variable MODEL_URL
      3. streamlit secrets: model_url, model_urls.<name>, model_storage.<name>
    Returns a URL string or None.
    """
    key = name.replace('.', '_').replace('-', '_').upper()
    env_key_specific = f"MODEL_URL_{key}"
    if env_key_specific in os.environ:
        return os.environ[env_key_specific]
    if 'MODEL_URL' in os.environ:
        return os.environ['MODEL_URL']

    # Try streamlit secrets if available (only available when running the app)
    try:
        import streamlit as st
        # top-level single URL
        if st.secrets.get('model_url'):
            return st.secrets.get('model_url')

        # keyed urls: model_urls: { name: url }
        model_urls = st.secrets.get('model_urls', None)
        if isinstance(model_urls, dict) and model_urls.get(name):
            return model_urls.get(name)

        # model_storage: { name: url }
        model_storage = st.secrets.get('model_storage', None)
        if isinstance(model_storage, dict) and model_storage.get(name):
            return model_storage.get(name)

        # fallback keyed secret like model_url_<name>
        secret_key = f"model_url_{name}"
        if st.secrets.get(secret_key):
            return st.secrets.get(secret_key)
    except Exception:
        # streamlit not installed or secrets not available
        return None



def load_artifacts(artifacts_dir: str = 'artifacts'):
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / 'model.joblib'
    encoders_path = artifacts_dir / 'encoders.joblib'
    le_status_path = artifacts_dir / 'le_status.joblib'
    le_cust_path = artifacts_dir / 'le_cust_reason.joblib'
    le_drv_path = artifacts_dir / 'le_driver_reason.joblib'

    # If artifacts are missing locally, attempt to download them using configured URLs
    # (environment variables or Streamlit secrets). This avoids committing large model
    # binaries directly to the repo.
    def _ensure(path: Path):
        if path.exists():
            return
        url = _find_artifact_url(path.name)
        if url:
            try:
                _download_url_to_file(url, path)
            except Exception:
                # If download fails, leave missing and let caller handle the error
                return

    _ensure(model_path)
    _ensure(encoders_path)
    _ensure(le_status_path)
    _ensure(le_cust_path)
    _ensure(le_drv_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train model first or provide a model_url in secrets.")

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path) if encoders_path.exists() else {}
    le_status = joblib.load(le_status_path) if le_status_path.exists() else None
    le_cust = joblib.load(le_cust_path) if le_cust_path.exists() else None
    le_drv = joblib.load(le_drv_path) if le_drv_path.exists() else None
    rf_cust_path = artifacts_dir / 'rf_cust_reason.joblib'
    rf_drv_path = artifacts_dir / 'rf_driver_reason.joblib'
    rf_cust = joblib.load(rf_cust_path) if rf_cust_path.exists() else None
    rf_drv = joblib.load(rf_drv_path) if rf_drv_path.exists() else None

    return {
        'model': model,
        'encoders': encoders,
        'le_status': le_status,
        'le_cust': le_cust,
        'le_drv': le_drv,
        'rf_cust': rf_cust,
        'rf_drv': rf_drv,
    }


def predict_from_input(input_dict: dict, artifacts_dir: str = 'artifacts') -> dict:
    """Take a single-row input dict, encode with saved encoders and return prediction.

    Returns a dict with keys: Predicted Booking Status (str), Predicted Cancellation Reason (str or Not Available)
    """
    arts = load_artifacts(artifacts_dir)
    model = arts['model']
    encoders = arts['encoders'] or {}
    le_status = arts['le_status']
    le_cust = arts['le_cust']
    le_drv = arts['le_drv']

    input_df = pd.DataFrame([input_dict])

    # Apply encoders that were explicitly saved (exact column names)
    for key, le in encoders.items():
        # remove the __auto__ prefix if present to get original column name
        if key.startswith('__auto__'):
            col = key[len('__auto__'):]
        else:
            col = key

        if col in input_df.columns:
            vals = input_df[col].astype(str)

            # If the input contains labels the encoder hasn't seen, extend classes_ so transform doesn't raise
            try:
                unseen = [v for v in vals.unique() if v not in le.classes_]
            except Exception:
                unseen = []

            if len(unseen) > 0:
                # append unseen labels to classes_. This assigns them new integer codes at the end.
                le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=le.classes_.dtype)])

            # Now transform (should not raise)
            input_df[col] = le.transform(vals)

    # Ensure column ordering expected by model
    try:
        feat_names = list(model.feature_names_in_)
    except Exception:
        # fallback: use current input columns order
        feat_names = list(input_df.columns)

    # Reindex and fill missing columns with zeros
    input_df = input_df.reindex(columns=feat_names, fill_value=0)

    pred_encoded = model.predict(input_df)[0]
    predicted_status = None
    if le_status is not None:
        try:
            predicted_status = le_status.inverse_transform([pred_encoded])[0]
        except Exception:
            predicted_status = str(pred_encoded)
    else:
        predicted_status = str(pred_encoded)

    predicted_reason = "Not available (no secondary reason model trained)"

    # If we have secondary models and the main prediction is a cancellation, predict reason
    rf_cust = arts.get('rf_cust')
    rf_drv = arts.get('rf_drv')

    try:
        if predicted_status == 'Cancelled by Customer' and rf_cust is not None:
            try:
                feat_names_cust = list(rf_cust.feature_names_in_)
            except Exception:
                feat_names_cust = list(input_df.columns)

            X_reason = input_df.reindex(columns=feat_names_cust, fill_value=0)
            pred_reason_enc = rf_cust.predict(X_reason)[0]
            if le_cust is not None:
                predicted_reason = le_cust.inverse_transform([int(pred_reason_enc)])[0]
            else:
                predicted_reason = str(pred_reason_enc)

        elif predicted_status == 'Cancelled by Driver' and rf_drv is not None:
            try:
                feat_names_drv = list(rf_drv.feature_names_in_)
            except Exception:
                feat_names_drv = list(input_df.columns)

            X_reason = input_df.reindex(columns=feat_names_drv, fill_value=0)
            pred_reason_enc = rf_drv.predict(X_reason)[0]
            if le_drv is not None:
                predicted_reason = le_drv.inverse_transform([int(pred_reason_enc)])[0]
            else:
                predicted_reason = str(pred_reason_enc)
    except Exception:
        # If anything fails predicting reasons, keep the default message
        predicted_reason = "Not available (reason model prediction failed)"

    return {
        'Predicted Booking Status': predicted_status,
        'Predicted Cancellation Reason': predicted_reason,
    }
