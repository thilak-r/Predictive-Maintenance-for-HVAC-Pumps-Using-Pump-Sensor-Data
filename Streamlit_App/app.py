import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import time # To simulate loading

# --- Configuration ---
# Make sure these paths are correct for your setup
MODEL_PATH = 'random_forest_best_model.pkl' # Direct filename
SCALER_PATH = 'scaler.pkl'                  # Direct filename
DATA_PATH = 'sensor.csv' # Assumes sensor.csv is also in the same folder
WINDOW_SIZE = 10 # The number of data points (time steps) needed for one prediction
EXPECTED_FEATURES = 51 # Number of raw sensor inputs expected
N_HISTORICAL = WINDOW_SIZE - 1

# --- Load Model and Scaler (Cached) ---
@st.cache_resource
def load_artifacts():
    """Loads the model and scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model ({MODEL_PATH}) or scaler ({SCALER_PATH}) not found in the same folder as the script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

# --- Load Sample Data (Cached) ---
@st.cache_data
def load_sample_data(nrows=200): # Load a bit more initially for slider range
    """Loads a sample chunk from the original sensor data."""
    try:
        # Load only necessary columns to potentially save memory/time
        all_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist() # Get all column names
        sensor_cols_to_load = [f'sensor_{i:02d}' for i in range(EXPECTED_FEATURES + 1)] # Generate sensor_00 to sensor_51 potential names
        actual_sensor_cols = [col for col in sensor_cols_to_load if col in all_cols and col != 'sensor_15'] # Filter existing and exclude sensor_15
        
        if len(actual_sensor_cols) != EXPECTED_FEATURES:
             st.error(f"Error: Could only find {len(actual_sensor_cols)} expected sensor columns in {DATA_PATH} (expected {EXPECTED_FEATURES}). Check file and configuration.")
             st.stop()
             
        df = pd.read_csv(DATA_PATH, usecols=actual_sensor_cols, nrows=nrows)
        return df
    except FileNotFoundError:
        st.error(f"Error: Original data file ({DATA_PATH}) not found in the same folder as the script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        st.stop()

# --- Feature Engineering Function (remains the same internally) ---
def generate_features(current_readings_array, historical_readings_df):
    # ... (internal logic is the same as before) ...
    if current_readings_array.shape[0] != EXPECTED_FEATURES:
        st.error(f"Internal Error: Feature generation - Expected {EXPECTED_FEATURES} current readings array elements, got {current_readings_array.shape[0]}")
        return None
    if not isinstance(historical_readings_df, pd.DataFrame) or historical_readings_df.shape != (N_HISTORICAL, EXPECTED_FEATURES):
         st.error(f"Internal Error: Feature generation - Expected historical data as DataFrame with shape ({N_HISTORICAL}, {EXPECTED_FEATURES}), got {type(historical_readings_df)} with shape {getattr(historical_readings_df, 'shape', 'N/A')}")
         return None
    current_df = pd.DataFrame([current_readings_array], columns=historical_readings_df.columns)
    combined_df = pd.concat([historical_readings_df, current_df], ignore_index=True)
    if combined_df.shape != (WINDOW_SIZE, EXPECTED_FEATURES):
         st.error(f"Internal Error: Feature generation - Combined DataFrame shape mismatch. Expected ({WINDOW_SIZE}, {EXPECTED_FEATURES}), got {combined_df.shape}")
         return None
    rolling_mean = combined_df.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).mean().iloc[-1].values
    rolling_std = combined_df.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).std().iloc[-1].values
    rolling_std = np.nan_to_num(rolling_std)
    feature_vector = np.concatenate([current_readings_array, rolling_mean, rolling_std])
    if len(feature_vector) != 153:
        st.error(f"Internal Error: Feature generation - Generated feature vector has {len(feature_vector)} elements, expected 153.")
        return None
    return feature_vector.reshape(1, -1)

# === Streamlit App ===
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="HVAC Pump Monitor")
    st.title("⚙️ HVAC Pump Predictive Maintenance")
    st.markdown("Predict pump status ('NORMAL' or 'RECOVERING') using historical sensor data.")
    st.markdown("---")

    # --- Load Artifacts ---
    with st.spinner("Loading predictive model..."):
        model, scaler = load_artifacts()

    # --- Load and Prepare Sample Data ---
    with st.spinner("Loading sample sensor data..."):
        # Load enough data to provide context for the slider range
        sample_df_full = load_sample_data(nrows=200) # Ensure enough rows

    # --- User Input Sidebar ---
    st.sidebar.header("Simulation Control")
    st.sidebar.markdown(f"""
    To predict the pump's status, the model needs **{WINDOW_SIZE} consecutive data points**
    (representing {WINDOW_SIZE} minutes of operation, assuming 1-minute intervals).

    Use the slider below to select the **starting point** for this {WINDOW_SIZE}-minute window
    within the available sample data.
    """)

    max_start_index = len(sample_df_full) - WINDOW_SIZE
    if max_start_index < 0:
        st.error(f"Not enough data loaded ({len(sample_df_full)} rows) to form a {WINDOW_SIZE}-point window. Load more rows in `load_sample_data`.")
        st.stop()

    start_index = st.sidebar.slider(
        f"Select STARTING row for {WINDOW_SIZE}-point analysis window",
        min_value=0,
        max_value=max_start_index,
        value=max_start_index // 2, # Default to middle
        # REMOVE THE HELP PARAMETER FROM HERE:
        # help=f"This selects rows {start_index} through {start_index + N_HISTORICAL} as history, and row {start_index + WINDOW_SIZE - 1} as the 'current' point for prediction."
    )

    # ADD THE DESCRIPTIVE TEXT HERE, AFTER start_index is defined:
    st.sidebar.caption(f"This selects rows **{start_index}** through **{start_index + N_HISTORICAL}** as history, and row **{start_index + WINDOW_SIZE - 1}** as the 'current' point for prediction.")


    # Calculate the end index of the window (the row being predicted)
    end_index = start_index + WINDOW_SIZE - 1

    # Extract the 10 rows (9 historical + 1 current) based on slider
    sample_window_df = sample_df_full.iloc[start_index : start_index + WINDOW_SIZE]

    # Prepare data for feature generation
    historical_sample = sample_window_df.iloc[:N_HISTORICAL] # First 9 rows
    current_sample_series = sample_window_df.iloc[-1]        # Last row as Series
    current_sample_array = current_sample_series.values      # Last row as numpy array

    st.sidebar.markdown("---")
    st.sidebar.subheader("Selected Data Window:")
    st.sidebar.dataframe(sample_window_df, height=250)
    st.sidebar.caption(f"Using data from row **{start_index}** to row **{end_index}** from `sensor.csv` sample.")
    st.sidebar.caption(f"The model will predict the status corresponding to the **end** of this window (Row **{end_index}**).")


    # --- Main Panel: Prediction ---
    st.subheader(f"Predict Pump Status for Row {end_index}")
    st.markdown(f"""
    Based on the sensor readings from row **{start_index}** up to row **{end_index}** (shown in the sidebar),
    the model predicts the pump's status at the time corresponding to row **{end_index}**.
    """)

    # --- Prediction Trigger ---
    if st.button(f"▶️ Run Prediction for Row {end_index}"):

        col1, col2 = st.columns([2,1], gap="large") # Layout columns

        with col1:
            st.markdown("##### Sensor Readings for Prediction Target:")
            # Display current readings nicely (e.g., expandable)
            with st.expander(f"Show Sensor Readings for Row {end_index} (51 values)"):
                 # Convert Series to DataFrame for better table display
                 st.dataframe(current_sample_series.to_frame().T)


        with st.spinner("Analyzing sensor patterns..."):
            time.sleep(0.5) # Simulate work
            # 1. Generate Features
            features = generate_features(current_sample_array, historical_sample)

            if features is not None:
                # 2. Scale Features
                scaled_features = scaler.transform(features)

                # 3. Predict
                prediction_encoded = model.predict(scaled_features)
                prediction_proba = model.predict_proba(scaled_features)

                # 4. Format Output
                status_map = {0: 'NORMAL', 1: 'RECOVERING'}
                predicted_status = status_map.get(prediction_encoded[0], 'UNKNOWN')
                prob_normal = prediction_proba[0][0]
                prob_recovering = prediction_proba[0][1]

                # 5. Display Results
                with col2:
                    st.markdown("##### Prediction Result:")
                    if predicted_status == 'NORMAL':
                        st.success(f"Predicted Status: **NORMAL** 👍")
                        delta_val = None
                        st.metric(label="Confidence (Normal Operation)", value=f"{prob_normal:.1%}", delta=delta_val)
                        st.progress(prob_normal)

                    elif predicted_status == 'RECOVERING':
                        st.warning(f"Predicted Status: **RECOVERING** ⚠️")
                        # Optional: Show delta relative to 0.5 if desired
                        # delta_val = f"{((prob_recovering - 0.5) / 0.5) * 100:.1f}% vs normal" if prob_recovering > 0.5 else None
                        delta_val = None # Keep it simple
                        st.metric(label="Confidence (Recovering State)", value=f"{prob_recovering:.1%}", delta=delta_val, delta_color="inverse")
                        st.progress(prob_recovering)
                    else:
                        st.error("Predicted Status: UNKNOWN")


                    with st.expander("Show Detailed Probabilities"):
                         st.write(f"- Probability NORMAL: {prob_normal:.4f}")
                         st.write(f"- Probability RECOVERING: {prob_recovering:.4f}")

            else:
                # Error messages are now shown via st.error within generate_features
                 st.error("Prediction failed. Check error messages above or in the console.")

    else:
         st.info(f"Click the button above to predict the status for row **{end_index}** using the selected data window.")

    st.markdown("---")
    st.caption("Note: This simulation uses sample data. In a real system, live sensor data would be continuously fed into the model.")