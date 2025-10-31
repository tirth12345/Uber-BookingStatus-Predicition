import streamlit as st
import pandas as pd
from pathlib import Path

from model_utils import load_artifacts, predict_from_input

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'


st.set_page_config(page_title='Ride Booking Outcome Predictor', layout='centered')

st.title('Ride Booking Outcome Predictor')

st.markdown(
    "This app loads a trained Random Forest model (if present) from `artifacts/` and predicts the booking status given ride features."
)

col1, col2 = st.columns(2)

with col1:
    # Basic categorical inputs
    vehicle_type = st.selectbox('Vehicle Type', ['Bike', 'Car', 'Auto', 'Other'])
    pickup_location = st.text_input('Pickup Location', 'Noida Extension')
    drop_location = st.text_input('Drop Location', 'Vishwavidyalaya')
    payment_method = st.selectbox('Payment Method', ['UPI', 'Cash', 'Card', 'Other'])

with col2:
    avg_vtat = st.number_input('Avg VTAT', value=5.9, step=0.1)
    avg_ctat = st.number_input('Avg CTAT', value=7.2, step=0.1)
    cancelled_by_customer = st.number_input('Cancelled Rides by Customer', min_value=0, value=0)
    cancelled_by_driver = st.number_input('Cancelled Rides by Driver', min_value=0, value=0)

col3, col4 = st.columns(2)
with col3:
    incomplete_rides = st.number_input('Incomplete Rides', min_value=0, value=0)
    booking_value = st.number_input('Booking Value', min_value=0.0, value=0.0, step=1.0)
with col4:
    ride_distance = st.number_input('Ride Distance', min_value=0.0, value=0.0, step=0.1)
    driver_ratings = st.number_input('Driver Ratings', min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    customer_rating = st.number_input('Customer Rating', min_value=0.0, max_value=5.0, value=4.5, step=0.1)

input_data = {
    'Vehicle Type': vehicle_type,
    'Pickup Location': pickup_location,
    'Drop Location': drop_location,
    'Avg VTAT': avg_vtat,
    'Avg CTAT': avg_ctat,
    'Cancelled Rides by Customer': cancelled_by_customer,
    'Cancelled Rides by Driver': cancelled_by_driver,
    'Incomplete Rides': incomplete_rides,
    'Booking Value': booking_value,
    'Ride Distance': ride_distance,
    'Driver Ratings': driver_ratings,
    'Customer Rating': customer_rating,
    'Payment Method': payment_method,
}

st.write('---')

colA, colB = st.columns([1, 1])
with colA:
    if st.button('Load model'):
        try:
            arts = load_artifacts(ARTIFACTS_DIR)
            st.success('Artifacts loaded successfully.')
        except Exception as e:
            st.error(f'Failed to load artifacts: {e}')

with colB:
    if st.button('Train model (from CSV)'):
        # lazy-import training function to avoid heavy deps on import
        try:
            from train_and_export import prepare_and_train

            with st.spinner('Training model (this may take a while)...'):
                prepare_and_train('ncr_ride_bookings.csv')
            st.success('Model trained and artifacts saved to artifacts/')
        except Exception as e:
            st.error(f'Training failed: {e}')

st.write('---')

if st.button('Predict for input'):
    try:
        result = predict_from_input(input_data, ARTIFACTS_DIR)
        st.subheader('Prediction')
        predicted_status = result.get('Predicted Booking Status')
        predicted_reason = result.get('Predicted Cancellation Reason')
        st.markdown(f"**Booking Status:** {predicted_status}")

        # Only show cancellation reason if the booking was predicted as cancelled
        if predicted_status in ('Cancelled by Customer', 'Cancelled by Driver'):
            st.markdown(f"**Cancellation Reason:** {predicted_reason}")
        else:
            st.markdown("**Cancellation Reason:** Not applicable (booking not cancelled)")
    except Exception as e:
        st.error(f'Prediction failed: {e}')

st.markdown('''
Notes:
- If `artifacts/model.joblib` is missing, click "Train model (from CSV)" (ensure `ncr_ride_bookings.csv` is in the working dir).
- The app now supports predicting cancellation reasons when the trained secondary models are available in `artifacts/` (they are trained automatically when `train_and_export.py` finds relevant rows).
''')
