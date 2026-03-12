import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client, Client
from datetime import date, timedelta
import urllib.parse

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
        # Added the new column to the empty state fallback
        return pd.DataFrame(columns=['Athlete', 'Station', 'Time (min)', 'Fraction_Completed', 'Date', 'Duration', 'RPE', 'Session_Load'])
    
    df = pd.DataFrame(response.data)
    
    # Idiot-proofing: If the column is missing or null, assume they somehow finished it.
    if 'fraction_completed' not in df.columns:
        df['fraction_completed'] = 1.0
    else:
        df['fraction_completed'] = df['fraction_completed'].fillna(1.0)
        
    df = df.rename(columns={
        'athlete_name': 'Athlete', 'station': 'Station', 'time_minutes': 'Time (min)', 
        'fraction_completed': 'Fraction_Completed',
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

def station_to_race_pace(station: str, time_min: float, fraction_completed: float = 1.0) -> float:
    # If they didn't even finish the station, extrapolate their time using Riegel's fatigue exponent.
    # The exponent 1.06 accounts for the fact that humans slow down as distance increases.
    # If fraction_completed is tiny, the math will correctly predict an abysmal time.
    if fraction_completed <= 0.0:
        return 999.0 # They did literally nothing. Give them a DNS.
        
    projected_station_time = time_min * ((1.0 / fraction_completed) ** 1.06)
    
    # Baselines for completing the FULL station
    baselines = {
        '1km Run': 4.5, 'SkiErg': 4.0, 'Sled Push': 3.0, 'Sled Pull': 4.0, 
        'Burpee Broad Jumps': 4.5, 'Rowing': 4.5, 'Farmers Carry': 2.0, 
        'Sandbag Lunges': 3.5, 'Wall Balls': 5.0, 'Full Race': 90.0
    }
    baseline = baselines.get(station, 4.0)
    
    return (projected_station_time / baseline) * 90.0

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
        # Change this line wherever it appears in your code (col2 AND the Leaderboard loop):
        athlete_data['Projected_Pace'] = athlete_data.apply(
            lambda row: station_to_race_pace(row['Station'], row['Time (min)'], row['Fraction_Completed']), 
            axis=1
        )
        
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

st.sidebar.markdown("---")
st.sidebar.subheader("Log New Session")

existing_athletes = sorted(df['Athlete'].unique().tolist()) if not df.empty else []
existing_athletes = ["+ Add New Victim"] + existing_athletes

with st.sidebar.form("data_entry_form", clear_on_submit=True):
    athlete_choice = st.selectbox("Select Athlete", existing_athletes)
    
    if athlete_choice == "+ Add New Victim":
        new_athlete_name = st.text_input("New Athlete Name").strip().title()
    else:
        new_athlete_name = athlete_choice

    new_station = st.selectbox("Focus Station", ['1km Run', 'SkiErg', 'Sled Push', 'Sled Pull', 'Burpee Broad Jumps', 'Rowing', 'Farmers Carry', 'Sandbag Lunges', 'Wall Balls', 'Full Race'])
    new_time = st.number_input("Time Survived (mins)", min_value=0.0, value=5.0, step=0.1)
    
    # NEW: Ask how much they actually did.
    fraction_done = st.slider("Percentage of Station Completed", 1, 100, 100) / 100.0
    
    st.markdown("**Internal Load**")
    new_duration = st.number_input("Total Session Duration (mins)", min_value=1.0, value=60.0)
    new_rpe = st.slider("Session RPE (1-10)", 1, 10, 7)
    new_date = st.date_input("Date", date.today())
    
    if st.form_submit_button("Submit Pain"):
        if athlete_choice == "+ Add New Victim" and not new_athlete_name:
            st.error("Enter a name before I throw an exception.")
        else:
            try:
                # Calculate their pathetic projected pace behind the scenes
                projected_race_pace = station_to_race_pace(new_station, float(new_time), fraction_done)
                
                # Notice we store exactly what they did, but you can use fraction_done in your analytics later
                supabase.table("hyrox_results").insert({
                    "athlete_name": new_athlete_name, 
                    "station": new_station,
                    "time_minutes": float(new_time), 
                    "fraction_completed": float(fraction_done), # The permanent record of their frailty
                    "recorded_at": str(new_date),
                    "duration_minutes": float(new_duration), 
                    "rpe": int(new_rpe)
                }).execute()

                st.success(f"Logged {new_athlete_name}. Projected Race Pace based on this tragedy: {round(projected_race_pace, 1)} minutes.")
                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Database error. Probably your fault: {e}")

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
    st.info("Leaderboard is empty.")

st.sidebar.markdown("---")
if st.sidebar.button("Delete Last Entry for Athlete"):
    if athlete_choice != "+ Add New Athlete":
        try:
            # Find the ID of the most recent entry for this specific human
            last_entry = supabase.table("hyrox_results") \
                .select("id") \
                .eq("athlete_name", athlete_choice) \
                .order("recorded_at", desc=True) \
                .limit(1) \
                .execute()
            
            if last_entry.data:
                record_id = last_entry.data[0]['id']
                supabase.table("hyrox_results").delete().eq("id", record_id).execute()
                st.sidebar.success(f"Deleted last entry for {athlete_choice}.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.warning("No data found to delete.")
        except Exception as e:
            st.sidebar.error(f"Deletion failed: {e}")
    else:
        st.sidebar.info("Select an existing athlete to delete their last entry.")

st.markdown("---")
st.header("30-Day Performance Audit")

if not df.empty:
    thirty_days_ago = pd.to_datetime(date.today() - timedelta(days=30))
    recent_df = df[df['Date'] >= thirty_days_ago]
    
    if not recent_df.empty:
        summary_data = []
        for athlete in recent_df['Athlete'].unique():
            a_df = recent_df[recent_df['Athlete'] == athlete]
            
            # Calculate metrics
            avg_strain = a_df['Session_Load'].mean()
            total_minutes = a_df['Duration'].sum()
            sessions = len(a_df)
            avg_completion = a_df['Fraction_Completed'].mean() * 100 # Turn it into a recognizable percentage
            
            summary_data.append({
                "Athlete": athlete,
                "Sessions": sessions,
                "Total Mat Time (min)": total_minutes,
                "Avg Session Intensity": round(avg_strain, 1),
                "Avg Completion %": f"{round(avg_completion, 1)}%" # The reality check
            })
        
        audit_df = pd.DataFrame(summary_data).sort_values("Avg Session Intensity", ascending=False)
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data recorded in the last 30 days.")
else:
    st.info("Database is empty.")

st.markdown("---")
st.header("Wall of Shame 📉")

if not df.empty:
    # Calculate all-time average completion percentage
    shame_df = df.groupby('Athlete')['Fraction_Completed'].mean().reset_index()
    shame_df['Average Completion (%)'] = (shame_df['Fraction_Completed'] * 100).round(1)
    
    # Filter out anyone who manages to complete at least half of their prescribed workout
    quitters = shame_df[shame_df['Fraction_Completed'] < 0.5].sort_values('Fraction_Completed')
    
    if not quitters.empty:
        st.error("Statistical liabilities. These individuals are averaging less than 50% completion across all sessions:")
        
        # Strip out the decimal fraction for the UI, keep it clean
        display_shame = quitters[['Athlete', 'Average Completion (%)']]
        
        # Paint their failure dark red
        st.dataframe(
            display_shame.style.applymap(
                lambda _: 'background-color: #4b0000; color: #ffcccc', 
                subset=['Average Completion (%)']
            ), 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.success("Miraculously, no one is currently averaging below 50% completion. Give it time.")
else:
    st.info("Gathering failure data...")

st.markdown("### Direct Harassment")
st.caption("Click to open WhatsApp and confront them with their own data.")

for _, row in quitters.iterrows():
    failing_athlete = row['Athlete']
    pitiful_percentage = row['Average Completion (%)']
    
    # The undeniable truth
    toxic_message = (
        f"Hi {failing_athlete}. My analytics dashboard flagged your recent Hyrox sessions. "
        f"You are currently completing exactly {pitiful_percentage}% of your prescribed workouts. "
        f"Are you pacing yourself for a 10-year race, or just giving up? Do better."
    )
    
    # URL encode the hostility so browsers don't choke on it
    safe_message = urllib.parse.quote(toxic_message)
    whatsapp_url = f"https://wa.me/?text={safe_message}"
    
    # Modern Streamlit link button. Don't use raw HTML hacks.
    st.link_button(f"📲 Text {failing_athlete}", whatsapp_url, use_container_width=True)

# Weekly monitor
st.markdown("---")
st.header("Weekly Fatigue Monitor")

if not df.empty:
    seven_days_ago = pd.to_datetime(date.today() - timedelta(days=7))
    week_df = df[df['Date'] >= seven_days_ago].copy()
    
    if not week_df.empty:
        fatigue_alerts = []
        for athlete in week_df['Athlete'].unique():
            # Get current week data
            a_week = week_df[week_df['Athlete'] == athlete]
            # Get all-time history for baseline
            a_history = df[df['Athlete'] == athlete]
            
            if len(a_history) > 3: # Need at least a small baseline
                curr_eff = (a_week['Time (min)'] * a_week['RPE']).mean()
                hist_eff = (a_history['Time (min)'] * a_history['RPE']).mean()
                
                # Efficiency Ratio: Current / Historical
                # > 1.10 means they are 10% less efficient than usual
                ratio = curr_eff / hist_eff
                
                status = "Recovered"
                if ratio > 1.15: status = "HIGH FATIGUE"
                elif ratio > 1.05: status = "Mild Strain"
                
                fatigue_alerts.append({
                    "Athlete": athlete,
                    "Efficiency Ratio": round(ratio, 2),
                    "Status": status,
                    "Recent Avg RPE": round(a_week['RPE'].mean(), 1)
                })
        
        if fatigue_alerts:
            alert_df = pd.DataFrame(fatigue_alerts).sort_values("Efficiency Ratio", ascending=False)
            
            # Highlight the red flags
            st.dataframe(alert_df.style.applymap(
                lambda x: 'background-color: #ff4b4b; color: white' if x == "HIGH FATIGUE" else 
                          ('background-color: #ffa500; color: black' if x == "Mild Strain" else ''),
                subset=['Status']
            ), use_container_width=True, hide_index=True)
        else:
            st.info("Gathering more baseline data...")
    else:
        st.info("No sessions logged in the last 7 days.")

st.markdown("---")
st.header("Weekly Autopsies 🪦")
st.caption("Generate a brutal summary of their last 7 days. Give them the cold, hard math.")

if not df.empty:
    # Look back exactly one week
    seven_days_ago = pd.to_datetime(date.today() - timedelta(days=7))
    last_week_df = df[df['Date'] >= seven_days_ago]
    
    if not last_week_df.empty:
        # Generate a report for every soul who bothered to show up
        for athlete in sorted(last_week_df['Athlete'].unique()):
            a_df = last_week_df[last_week_df['Athlete'] == athlete]
            
            sessions = len(a_df)
            total_duration = a_df['Duration'].sum()
            avg_completion = (a_df['Fraction_Completed'].mean() * 100).round(1)
            avg_rpe = a_df['RPE'].mean().round(1)
            
            # The Oracle's statistical verdict
            if avg_completion >= 90:
                verdict = "Anomaly detected. They might actually survive the warmup."
            elif avg_completion >= 50:
                verdict = "Consistently mediocre. Continuing to take their money is morally acceptable."
            else:
                verdict = "A statistical tragedy. If they register for a race, they will perish."

            with st.expander(f"View Report: {athlete} (Avg Completion: {avg_completion}%)"):
                report_text = f"""### Athlete Autopsy: {athlete}
**Dates:** {seven_days_ago.strftime('%Y-%m-%d')} to {date.today().strftime('%Y-%m-%d')}

#### The Damage
* **Sessions Attempted:** {sessions}
* **Total Mat Time:** {total_duration} minutes
* **Average Completion Rate:** {avg_completion}%
* **Claimed Effort (RPE):** {avg_rpe}/10 

#### Breakdown
"""
                # List their individual failures
                for _, row in a_df.iterrows():
                    station = row['Station']
                    pct = round(row['Fraction_Completed'] * 100, 1)
                    report_text += f"* {row['Date'].strftime('%A')}: {station} - {pct}% completed in {row['Time (min)']} mins.\n"
                
                report_text += f"\n**Oracle's Verdict:** {verdict}\n"
                
                st.markdown(report_text)
                
                # Let you download it so you can print it and staple it to their forehead
                st.download_button(
                    label=f"Download {athlete}'s Failure", 
                    data=report_text, 
                    file_name=f"{athlete.replace(' ', '_')}_weekly_report.txt",
                    mime="text/plain"
                )
    else:
        st.info("Nobody logged any sessions this week. Completely expected.")
else:
    st.info("Database empty. You have no clients.")