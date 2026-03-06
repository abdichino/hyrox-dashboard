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

# Get existing athletes for the dropdown
existing_athletes = sorted(df['Athlete'].unique().tolist()) if not df.empty else []
existing_athletes = ["+ Add New Athlete"] + existing_athletes

with st.sidebar.form("data_entry_form", clear_on_submit=True):
    # Selection logic
    athlete_choice = st.selectbox("Select Athlete", existing_athletes)
    
    if athlete_choice == "+ Add New Athlete":
        new_athlete_name = st.text_input("New Athlete Name").strip().title()
    else:
        new_athlete_name = athlete_choice

    new_station = st.selectbox("Focus Station", ['1km Run', 'SkiErg', 'Sled Push', 'Sled Pull', 'Burpee Broad Jumps', 'Rowing', 'Farmers Carry', 'Sandbag Lunges', 'Wall Balls', 'Full Race'])
    new_time = st.number_input("Time on Station (mins)", min_value=0.0, value=5.0, step=0.1)
    
    st.markdown("**Internal Load**")
    new_duration = st.number_input("Total Session Duration (mins)", min_value=1.0, value=60.0)
    new_rpe = st.slider("Session RPE (1-10)", 1, 10, 7)
    new_date = st.date_input("Date", date.today())
    
    if st.form_submit_button("Submit Pain"):
        if athlete_choice == "+ Add New Athlete" and not new_athlete_name:
            st.error("Please enter a name for the new athlete.")
        else:
            try:
                supabase.table("hyrox_results").insert({
                    "athlete_name": new_athlete_name, 
                    "station": new_station,
                    "time_minutes": float(new_time), 
                    "recorded_at": str(new_date),
                    "duration_minutes": float(new_duration), 
                    "rpe": int(new_rpe)
                }).execute()
                st.success(f"Logged {new_athlete_name} successfully.")
                # Clear cache so the new name appears in the list immediately
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Database error: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Hoarding")

if not df.empty:
    # Convert the dataframe to a CSV string
    csv_data = df.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name=f"hyrox_victims_{date.today()}.csv",
        mime="text/csv",
    )
else:
    st.sidebar.info("No data to export.")

# Leaderboard
st.markdown("---")
st.header("Predicted Leaderboard")

if not df.empty:
    leaderboard_data = []
    
    # Calculate the latest Bayesian prediction for every athlete
    for athlete_name in df['Athlete'].unique():
        athlete_subset = df[df['Athlete'] == athlete_name].copy()
        athlete_subset['Projected_Pace'] = athlete_subset.apply(
            lambda row: station_to_race_pace(row['Station'], row['Time (min)']), axis=1
        )
        
        # Run the Bayesian engine on their history
        res = bayesian_race_predictor(athlete_subset['Projected_Pace'].tolist())
        
        if not res.empty:
            latest = res.iloc[-1]
            leaderboard_data.append({
                "Athlete": athlete_name,
                "Predicted Time (min)": round(latest['Predicted_Time'], 1),
                "Confidence Interval (±)": round(latest['Uncertainty_Std'] * 2, 1),
                "Last Station": athlete_subset['Station'].iloc[-1]
            })

    # Display as a clean, sorted table
    leader_df = pd.DataFrame(leaderboard_data).sort_values("Predicted Time (min)")
    
    # Add a 'Rank' column
    leader_df.insert(0, 'Rank', range(1, len(leader_df) + 1))
    
    st.table(leader_df.set_index('Rank'))
    st.caption("Lower is better. If you're at the bottom, the math suggests you try harder.")
else:
    st.info("Leaderboard is empty. Waiting for the first victim.")