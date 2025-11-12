import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Adoption Forecasting ⚡",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions (for performance) ---

@st.cache_resource
def load_model_and_encoders():
    """Load the model and label encoders once and cache them."""
    try:
        model = joblib.load('forecasting_ev_model.pkl')
        le_county = joblib.load('le_county.pkl')
        le_state = joblib.load('le_state.pkl')
        return model, le_county, le_state
    except FileNotFoundError:
        st.error("Model or encoder files not found. Please ensure 'forecasting_ev_model.pkl', 'le_county.pkl', and 'le_state.pkl' are in the same folder.")
        return None, None, None

@st.cache_data
def load_data():
    """Load and preprocess the main data file once and cache it."""
    try:
        df = pd.read_csv('preprocessed_ev_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        # Ensure we have the one-hot columns for the app logic
        if 'Vehicle_Passenger' not in df.columns:
            df['Vehicle_Passenger'] = 0
        if 'Vehicle_Truck' not in df.columns:
            df['Vehicle_Truck'] = 0
        return df
    except FileNotFoundError:
        st.error("Data file 'preprocessed_ev_data.csv' not found. Please ensure it is in the same folder.")
        return None

# --- Load All Assets ---
model, le_county, le_state = load_model_and_encoders()
df = load_data()

# Define the feature list the model was trained on
# This MUST match the feature list from your notebook
FEATURES = [
    'ev_total_lag1', 'ev_total_lag2', 'ev_total_lag3',
    'ev_total_pct_change_1', 'ev_total_pct_change_3', 'ev_growth_slope',
    'state_encoded', 'Vehicle_Passenger', 'Vehicle_Truck',
    'bev_phev_ratio', 'ev_growth_acceleration'
]

# --- Sidebar for User Inputs ---
st.sidebar.title("Forecast Parameters 🚗")

if df is not None and le_county is not None:
    # Get county names from the label encoder
    county_names = list(le_county.classes_)
    # Pre-select a default county, e.g., 'Kings'
    default_county_index = county_names.index('Kings') if 'Kings' in county_names else 0
    
    selected_county = st.sidebar.selectbox(
        "Select a County:",
        options=county_names,
        index=default_county_index
    )

    forecast_months = st.sidebar.slider(
        "Months to Forecast:",
        min_value=6,
        max_value=36,
        value=24,
        step=6
    )

    run_forecast = st.sidebar.button("⚡ Run Forecast")

else:
    st.sidebar.error("App cannot start. Required files are missing.")
    run_forecast = False

# --- Main Page ---
st.title("EV Adoption Forecasting Dashboard")
st.markdown("Use the sidebar to select a county and forecast horizon.")

if run_forecast:
    with st.spinner(f"Running forecast for {selected_county}..."):
        # 1. Get the encoded values for the selected county
        try:
            county_code = le_county.transform([selected_county])[0]
        except ValueError:
            st.error(f"County '{selected_county}' was not in the training data.")
            st.stop()

        # 2. Get the last 6 months of historical data for this county
        historical_df = df[df['county_encoded'] == county_code].sort_values("Date")
        
        if len(historical_df) < 6:
            st.warning(f"Not enough historical data for {selected_county} (need at least 6 months). Forecast may be inaccurate.")
            st.stop()

        last_known_rows = historical_df.tail(6).copy()
        
        # 3. Initialize loop variables
        forecast_predictions = []
        historical_ev = list(last_known_rows['Electric Vehicle (EV) Total'])
        cumulative_ev = list(last_known_rows['Electric Vehicle (EV) Total'].cumsum())
        slope_history = list(last_known_rows['ev_growth_slope'])

        # Get static features from the last row
        last_row = last_known_rows.iloc[-1]
        state_code = last_row['state_encoded']
        v_pass = last_row['Vehicle_Passenger']
        v_truck = last_row['Vehicle_Truck']
        last_date = last_row['Date']

        # 4. Run the forecast loop
        for i in range(forecast_months):
            # Calculate lag features
            lag1 = historical_ev[-1]
            lag2 = historical_ev[-2]
            lag3 = historical_ev[-3]

            # Calculate pct change features
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            
            # Calculate bev/phev ratio (use last known row's ratio)
            bev_phev_ratio = last_row['bev_phev_ratio']

            # Calculate growth slope
            if len(cumulative_ev) >= 6:
                ev_growth_slope = np.polyfit(range(6), cumulative_ev[-6:], 1)[0]
            else:
                ev_growth_slope = 0 # Fallback
            
            # Calculate growth acceleration
            ev_growth_acceleration = (ev_growth_slope - slope_history[-1]) / slope_history[-1] if slope_history[-1] != 0 else 0
            
            # 5. Construct feature row
            new_row = {
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_growth_slope,
                'state_encoded': state_code,
                'Vehicle_Passenger': v_pass,
                'Vehicle_Truck': v_truck,
                'bev_phev_ratio': bev_phev_ratio,
                'ev_growth_acceleration': ev_growth_acceleration
            }
            
            X_new = pd.DataFrame([new_row])[FEATURES]
            
            # 6. Predict
            pred = model.predict(X_new)[0]
            pred = max(0, np.round(pred)) # Ensure prediction is not negative
            
            # 7. Store results and update history
            current_date = last_date + relativedelta(months=i+1)
            forecast_predictions.append({
                'Date': current_date,
                'Electric Vehicle (EV) Total': pred,
                'Source': 'Forecast'
            })
            
            # Update history for next loop
            historical_ev.append(pred)
            cumulative_ev.append(cumulative_ev[-1] + pred)
            slope_history.append(ev_growth_slope)

        # --- Display Results ---
        st.header(f"📈 Forecast for {selected_county} County")

        # Create combined DataFrame for charting
        forecast_df = pd.DataFrame(forecast_predictions)
        historical_chart_df = historical_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        historical_chart_df['Source'] = 'Historical'
        
        combined_df = pd.concat([historical_chart_df, forecast_df])
        combined_df['Cumulative EVs'] = combined_df['Electric Vehicle (EV) Total'].cumsum()
        
        # Get key metrics
        last_hist_val = historical_chart_df.iloc[-1]['Electric Vehicle (EV) Total']
        last_fcst_val = forecast_df.iloc[-1]['Electric Vehicle (EV) Total']
        last_fcst_cumulative = combined_df[combined_df['Source'] == 'Forecast'].iloc[-1]['Cumulative EVs']
        
        # Create tabs for clean output
        tab1, tab2, tab3, tab4 = st.tabs(["Forecast Snapshot", "Monthly Adoption", "Cumulative Growth", "Raw Data"])

        with tab1:
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Last Recorded Monthly Count", f"{int(last_hist_val)} EVs")
            col2.metric(f"Forecasted Count (in {forecast_months} months)", f"{int(last_fcst_val)} EVs")
            col3.metric(f"Total Forecasted Cumulative EVs", f"{int(last_fcst_cumulative)} EVs")
            st.info(f"This forecast predicts the adoption trend for **{selected_county}** over the next **{forecast_months}** months based on historical patterns.")

        with tab2:
            st.subheader("Monthly EV Adoption (Historical vs. Forecast)")
            
            base = alt.Chart(combined_df).mark_bar().encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Electric Vehicle (EV) Total:Q', title='Monthly EV Count'),
                color=alt.Color('Source:N', title="Data Source", scale={'domain': ['Historical', 'Forecast'], 'range': ['#1f77b4', '#ff7f0e']}),
                tooltip=['Date', 'Electric Vehicle (EV) Total', 'Source']
            ).interactive()
            
            st.altair_chart(base, use_container_width=True)

        with tab3:
            st.subheader("Cumulative EV Growth (Historical vs. Forecast)")
            
            line_chart = alt.Chart(combined_df).mark_line(point=True).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Cumulative EVs:Q', title='Cumulative EV Count'),
                color=alt.Color('Source:N', title="Data Source", scale={'domain': ['Historical', 'Forecast'], 'range': ['#1f77b4', '#ff7f0e']}),
                tooltip=['Date', 'Cumulative EVs', 'Source']
            ).interactive()
            
            st.altair_chart(line_chart, use_container_width=True)
        
        with tab4:
            st.subheader("Raw Forecast Data")
            st.dataframe(forecast_df.style.format({'Electric Vehicle (EV) Total': "{:.0f}"}))

else:
    st.info("Please select a county and click 'Run Forecast' to see the results.")