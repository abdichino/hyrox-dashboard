import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client, Client
from datetime import date

st.set_page_config(page_title="Hyrox Tracker", layout="wide", page_icon="💀")

# Data client
@st.cache_resource
def init_connection() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_connection()

@st.cache_data(ttl=600)
def get_real_data():
    response = supabase.table("hyrox_results").select("*").execute()
    if not response.data:
        return pd.DataFrame(columns=['Athlete', 'Station', 'Time (min)', 'Date', 'Duration', 'RPE', 'Session_Load'])
    
    df = pd.DataFrame(response.data)
    df = df.rename(columns={
        'athlete_name': 'Athlete', 'station': 'Station', 'time_minutes': 'Time (min)', 
        'recorded_at': 'Date', 'duration_minutes': 'Duration', 'rpe': 'RPE'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df['Session_Load'] = df['Duration'] * df['RPE']
    df = df.sort_values('Date')
    return df

df = get_real_data()

# Math
def calculate_banister_model(daily_loads: list, tau_fitness: int = 45, tau_fatigue: int = 15) -> pd.DataFrame:
    days = len(daily_loads)
    fitness, fatigue = np.zeros(days), np.zeros(days)
    fitness[0], fatigue[0] = daily_loads[0], daily_loads[0]
    
    for t in range(1, days):
        fitness[t] = daily_loads[t] + np.exp(-1 / tau_fitness) * fitness[t-1]
        fatigue[t] = daily_loads[t] + np.exp(-1 / tau_fatigue) * fatigue[t-1]
        
    performance = fitness - (2.0 * fatigue)
    return pd.DataFrame({'Fitness': fitness, 'Fatigue': fatigue, 'Performance': performance})

def bayesian_race_predictor(observed_projections: list, prior_mean: float = 90.0, prior_std: float = 15.0) -> pd.DataFrame:
    if not observed_projections:
        return pd.DataFrame()
        
    predictions = []
    current_mu = prior_mean
    current_var = prior_std ** 2
    
    for i, obs in enumerate(observed_projections):
        obs_var = 10.0 ** 2 # High variance because they only did a fragment of the race
        
        # Bayesian Normal-Normal Update
        new_mu = ((current_mu / current_var) + (obs / obs_var)) / ((1 / current_var) + (1 / obs_var))
        new_var = 1 / ((1 / current_var) + (1 / obs_var))
        
        predictions.append({
            "Workout": i + 1,
            "Observed_Pace": obs,
            "Predicted_Time": new_mu,
            "Uncertainty_Std": np.sqrt(new_var)
        })
        
        current_mu, current_var = new_mu, new_var
        
    return pd.DataFrame(predictions)

def station_to_race_pace(station: str, time_min: float) -> float:
    # Heuristic to translate a single station time into a full 90-min race equivalent
    baselines = {
        '1km Run': 4.5, 'SkiErg': 4.0, 'Sled Push': 3.0, 'Sled Pull': 4.0, 
        'Burpee Broad Jumps': 4.5, 'Rowing': 4.5, 'Farmers Carry': 2.0, 
        'Sandbag Lunges': 3.5, 'Wall Balls': 5.0, 'Full Race': 90.0
    }
    baseline = baselines.get(station, 4.0)
    # If they are 10% slower than the baseline, their predicted race is 10% slower than 90 mins
    return (time_min / baseline) * 90.0

# UI
st.title("Hyrox Analytics:")

if df.empty:
    st.warning("Your database is completely empty. Log some data in the sidebar.")
else:
    athlete = st.sidebar.selectbox("Select Athlete", df['Athlete'].unique())
    athlete_data = df[df['Athlete'] == athlete]
    
    col1, col2 = st.columns(2)
    
    # Banister model, fatigue tracker
    with col1:
        st.subheader("Fatigue & Injury Risk")
        min_date = athlete_data['Date'].min()
        # Create a continuous timeline so rolling averages actually work
        all_dates = pd.date_range(start=min_date, end=pd.to_datetime(date.today()))
        
        # Fill rest days with 0 load
        daily_load = athlete_data.groupby('Date')['Session_Load'].sum().reindex(all_dates, fill_value=0).reset_index()
        daily_load.columns = ['Date', 'Session_Load']
        

        banister_df = calculate_banister_model(daily_load['Session_Load'].tolist())
        banister_df['Date'] = daily_load['Date']
        
        daily_load['Rolling_Mean'] = daily_load['Session_Load'].rolling(window=7, min_periods=1).mean()
        daily_load['Rolling_Std'] = daily_load['Session_Load'].rolling(window=7, min_periods=1).std().replace(0, 0.1)
        daily_load['Weekly_Load'] = daily_load['Session_Load'].rolling(window=7, min_periods=1).sum()
        
        daily_load['Monotony'] = daily_load['Rolling_Mean'] / daily_load['Rolling_Std']
        daily_load['Strain'] = daily_load['Weekly_Load'] * daily_load['Monotony']

        strain_mean = daily_load['Strain'].mean()
        strain_std = daily_load['Strain'].std()
        
        if pd.isna(strain_std) or strain_std == 0:
            danger_threshold = strain_mean * 2 if strain_mean > 0 else 5000
        else:
            danger_threshold = strain_mean + (1.5 * strain_std)

        fig1 = go.Figure()

        fig1.add_trace(go.Bar(x=daily_load['Date'], y=daily_load['Strain'], name='Weekly Strain', marker_color='rgba(255, 0, 0, 0.3)'))
        
        fig1.add_trace(go.Scatter(x=banister_df['Date'], y=banister_df['Performance'], mode='lines', name='Net Readiness', line=dict(color='cyan', width=2)))
        
        fig1.add_hline(y=danger_threshold, line_dash="dash", line_color="red", 
                       annotation_text="Injury Imminent (+1.5 SD)", annotation_position="top left",
                       annotation_font_color="red")
        
        fig1.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

    # Bayesian Model, race prediction
    with col2:
        st.subheader("Projected Race Time (Bayesian)")
        athlete_data['Projected_Pace'] = athlete_data.apply(lambda row: station_to_race_pace(row['Station'], row['Time (min)']), axis=1)
        
        bayes_df = bayesian_race_predictor(athlete_data['Projected_Pace'].tolist())
        
        if not bayes_df.empty:
            bayes_df['Date'] = athlete_data['Date'].values
            
            bayes_df['Upper_Bound'] = bayes_df['Predicted_Time'] + (bayes_df['Uncertainty_Std'] * 2) # 95% confidence
            bayes_df['Lower_Bound'] = bayes_df['Predicted_Time'] - (bayes_df['Uncertainty_Std'] * 2)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=pd.concat([bayes_df['Date'], bayes_df['Date'][::-1]]),
                y=pd.concat([bayes_df['Upper_Bound'], bayes_df['Lower_Bound'][::-1]]),
                fill='toself', fillcolor='rgba(255, 255, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='95% Probability Cone'
            ))
            fig2.add_trace(go.Scatter(x=bayes_df['Date'], y=bayes_df['Predicted_Time'], mode='lines+markers', name='Predicted Time', line=dict(color='gold', width=3)))
            fig2.add_trace(go.Scatter(x=bayes_df['Date'], y=bayes_df['Observed_Pace'], mode='markers', name='Raw Workout Paces', marker=dict(color='gray', size=6)))
            
            fig2.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Minutes")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data to run Bayesian prediction.")

# Data entry
st.sidebar.markdown("---")
st.sidebar.subheader("Log New Session")

with st.sidebar.form("data_entry_form", clear_on_submit=True):
    new_athlete = st.text_input("Athlete Name")
    new_station = st.selectbox("Focus Station", ['1km Run', 'SkiErg', 'Sled Push', 'Sled Pull', 'Burpee Broad Jumps', 'Rowing', 'Farmers Carry', 'Sandbag Lunges', 'Wall Balls', 'Full Race'])
    new_time = st.number_input("Time on Station (mins)", min_value=0.0, value=5.0, step=0.1)
    
    st.markdown("**Internal Load**")
    new_duration = st.number_input("Total Session Duration (mins)", min_value=1.0, value=60.0)
    new_rpe = st.slider("Session RPE (1 = Asleep, 10 = Dying)", 1, 10, 7)
    new_date = st.date_input("Date", date.today())
    
    if st.form_submit_button("Submit Pain"):
        if not new_athlete.strip():
            st.error("Enter a name.")
        else:
            try:
                supabase.table("hyrox_results").insert({
                    "athlete_name": new_athlete, "station": new_station,
                    "time_minutes": float(new_time), "recorded_at": str(new_date),
                    "duration_minutes": float(new_duration), "rpe": int(new_rpe)
                }).execute()
                st.success("Logged successfully.")
                get_real_data.clear()
            except Exception as e:
                st.error(f"Database error: {e}")
