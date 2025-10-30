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

Extending

- Add training code for the customer/driver cancellation reason models and save them to `artifacts/`.
- Add UI elements in `app.py` to show reason predictions when available.
