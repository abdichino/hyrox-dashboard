import streamlit as st
import pandas as pd
from supabase import create_client, Client
import numpy as np
import plotly.express as px
from datetime import timedelta, date

# --- 1. SETTINGS & MODERN STANDARDS ---
# Don't even think about using global variables for state management.
st.set_page_config(page_title="Hyrox Suffering Tracker", layout="wide", page_icon="💀")

@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# --- 2. DATA ---
@st.cache_data(ttl=600) # Caches data for 10 minutes so you don't burn your free tier API limits
def get_real_data():
    # Fetching data like a grown-up
    response = supabase.table("hyrox_results").select("*").execute()
    data = response.data
    
    if not data:
        # Fallback if your database is empty so the app doesn't immediately crash
        return pd.DataFrame(columns=['athlete_name', 'station', 'time_minutes', 'recorded_at'])
    
    df = pd.DataFrame(data)
    # Rename columns to match the rest of the dashboard code you already copied
    df = df.rename(columns={'athlete_name': 'Athlete', 'station': 'Station', 'time_minutes': 'Time (min)', 'recorded_at': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = get_real_data()

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
