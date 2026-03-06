import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta, date

# --- 1. SETTINGS & MODERN STANDARDS ---
# Don't even think about using global variables for state management.
st.set_page_config(page_title="Hyrox Suffering Tracker", layout="wide", page_icon="💀")

# --- 2. DATA MOCKING (Because your backend doesn't exist) ---
@st.cache_data
def get_fake_data():
    """Generating data because you haven't wired up a PostgreSQL database yet."""
    np.random.seed(42)
    dates = pd.date_range(start=date.today() - timedelta(days=90), periods=12, freq='W')
    
    # The actual HYROX gauntlet
    stations = [
        '1km Run', 'SkiErg', 'Sled Push', 'Sled Pull', 
        'Burpee Broad Jumps', 'Rowing', 'Farmers Carry', 
        'Sandbag Lunges', 'Wall Balls'
    ]
    
    data = []
    for athlete in ['Victim A', 'Victim B', 'Victim C']:
        base_time = np.random.uniform(60, 90)  # Total minutes baseline
        for i, d in enumerate(dates):
            # Assume they get slightly better, or just randomly worse
            improvement = i * 0.5 + np.random.normal(0, 1)
            for station in stations:
                # Random minutes per station (rough approximation for the demo)
                time_taken = max(2.0, (base_time / 9) - improvement * 0.1 + np.random.normal(0, 0.5))
                data.append({'Athlete': athlete, 'Date': d, 'Station': station, 'Time (min)': time_taken})
    
    return pd.DataFrame(data)

df = get_fake_data()

# --- 3. UI LAYOUT ---
st.title("🏃‍♂️ Hyrox Class Dashboard")
st.markdown("Track your class's descent into madness. Or 'fitness.' Whatever you call it.")

# Sidebar filtering
athlete = st.sidebar.selectbox("Select Athlete to Interrogate", df['Athlete'].unique())
athlete_data = df[df['Athlete'] == athlete]

# Top Level Metrics
st.subheader(f"Current Stats for {athlete}")
col1, col2, col3 = st.columns(3)

# Calculate totals
latest_date = athlete_data['Date'].max()
first_date = athlete_data['Date'].min()
total_time_latest = athlete_data[athlete_data['Date'] == latest_date]['Time (min)'].sum()
total_time_first = athlete_data[athlete_data['Date'] == first_date]['Time (min)'].sum()

# Display metrics with delta
col1.metric("Latest Total Time", f"{total_time_latest:.2f} min", f"{total_time_first - total_time_latest:.2f} min (Improvement)")
col2.metric("Best Station", athlete_data.groupby('Station')['Time (min)'].mean().idxmin())
col3.metric("Worst Station", athlete_data.groupby('Station')['Time (min)'].mean().idxmax())

# --- 4. DATA VISUALIZATION & PROJECTION ---
st.subheader("Performance Trend & Projection")

# Group by date for total time
trend_df = athlete_data.groupby('Date')['Time (min)'].sum().reset_index()

# Simple Linear Regression for projection (since you probably failed Stats 101)
z = np.polyfit(range(len(trend_df)), trend_df['Time (min)'], 1)
p = np.poly1d(z)

# Project the next 4 weeks
future_dates = pd.date_range(start=latest_date + timedelta(days=7), periods=4, freq='W')
future_x = range(len(trend_df), len(trend_df) + 4)
future_y = p(future_x)

# Combine historical and projected data for the chart
proj_df = pd.DataFrame({'Date': future_dates, 'Projected Time (min)': future_y})
trend_df['Projected Time (min)'] = p(range(len(trend_df)))

fig = px.line(trend_df, x='Date', y=['Time (min)', 'Projected Time (min)'], 
              labels={'value': 'Total Time (Minutes)', 'variable': 'Legend'},
              title="Actual vs. Projected Times")

# Add the future projection as a dotted line
fig.add_scatter(x=proj_df['Date'], y=proj_df['Projected Time (min)'], mode='lines', 
                line=dict(dash='dot', color='red'), name='Future Projection')

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Breakdown of the latest session
st.subheader("Latest Station Breakdown")
fig_bar = px.bar(athlete_data[athlete_data['Date'] == latest_date], 
                 x='Station', y='Time (min)', color='Station',
                 title="Where They Are Bleeding Time")
fig_bar.update_layout(template="plotly_dark", showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)