# Ride Booking Outcome Predictor (Streamlit)

This project trains a Random Forest model on the `ncr_ride_bookings.csv` dataset and provides a Streamlit UI to predict booking outcomes.

Quick start

1. Create a virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Put your dataset next to the project (default filename: `ncr_ride_bookings.csv`).

4. Train the model (optional from terminal):

```powershell
python train_and_export.py --csv ncr_ride_bookings.csv
```

5. Run the Streamlit app:

```powershell
streamlit run app.py
```

App notes

- The app will attempt to load artifacts from `artifacts/`. If missing, click the **Train model (from CSV)** button in the UI.
- The current implementation predicts Booking Status. Secondary models for cancellation reasons were not trained automatically; you can extend `train_and_export.py` to create those and adjust `model_utils.predict_from_input`.

Deployment to Streamlit Community Cloud
-------------------------------------

This app can be deployed to Streamlit Community Cloud (share.streamlit.io). Minimal requirements:

- Ensure `app.py` and `requirements.txt` are committed to the repository.
- If your `artifacts/` models are small, you can commit them directly. For larger model files, either enable Git LFS for `artifacts/*.joblib` or store models on cloud storage (S3, GCS) and download them at runtime using secrets.
- Optional files that improve deployment:
	- `.streamlit/config.toml` (controls server settings)
	- `runtime.txt` (forces a Python version)
	- `.gitattributes` (if you use Git LFS)
	- Add secrets via the Streamlit UI rather than committing them (`st.secrets` is available in the runtime).

Quick deploy steps:

1. Push the repository to GitHub.
2. On Streamlit Community Cloud, click "New app", connect your GitHub account, and point to this repo and `app.py`.
3. If your app needs secrets (API keys or model download URLs), add them in the Streamlit app settings under Secrets.

Notes:
- Do not commit `.streamlit/secrets.toml` with real keys. Use the platform secrets instead.
- If you want, enable Git LFS for large artifacts and add `artifacts/*.joblib` to `.gitattributes`.

Extending

- Add training code for the customer/driver cancellation reason models and save them to `artifacts/`.
- Add UI elements in `app.py` to show reason predictions when available.

<!-- end -->