import streamlit as st
import fastf1
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Configure FastF1 cache
fastf1.Cache.enable_cache(cache_dir)

# Page config
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Title and description
st.title("🏎️ F1 Race Predictor")
st.markdown("Predict F1 race outcomes using historical data from FastF1 API")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.historical_data = None
    st.session_state.model = None
    st.session_state.driver_encoder = None
    st.session_state.team_encoder = None

def load_historical_data(years, sessions_per_year=5):
    """Load historical race data from FastF1"""
    all_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_sessions = len(years) * sessions_per_year
    current_session = 0
    
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] != 'testing'][:sessions_per_year]
            
            for _, race in races.iterrows():
                current_session += 1
                progress_bar.progress(current_session / total_sessions)
                status_text.text(f"Loading {year} - {race['EventName']}...")
                
                try:
                    session = fastf1.get_session(year, race['RoundNumber'], 'R')
                    session.load(telemetry=False)
                    
                    results = session.results
                    results['Year'] = year
                    results['Round'] = race['RoundNumber']
                    results['Circuit'] = race['Location']
                    results['EventName'] = race['EventName']
                    
                    # Add weather data if available
                    if hasattr(session, 'weather_data') and not session.weather_data.empty:
                        avg_temp = session.weather_data['AirTemp'].mean()
                        avg_humidity = session.weather_data['Humidity'].mean()
                        results['AvgTemp'] = avg_temp
                        results['AvgHumidity'] = avg_humidity
                    else:
                        results['AvgTemp'] = 25  # Default values
                        results['AvgHumidity'] = 50
                    
                    all_data.append(results)
                except Exception as e:
                    st.warning(f"Could not load {year} {race['EventName']}: {str(e)}")
                    continue
                    
        except Exception as e:
            st.error(f"Error loading {year} data: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def prepare_features(df):
    """Prepare features for machine learning model"""
    # Create features
    features_df = pd.DataFrame()
    
    # Encode categorical variables
    driver_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    circuit_encoder = LabelEncoder()
    
    # Handle missing values and encode
    df['Driver'] = df['Abbreviation'].fillna('UNK')
    df['Team'] = df['TeamName'].fillna('Unknown')
    df['Circuit'] = df['Circuit'].fillna('Unknown')
    
    features_df['DriverEncoded'] = driver_encoder.fit_transform(df['Driver'])
    features_df['TeamEncoded'] = team_encoder.fit_transform(df['Team'])
    features_df['CircuitEncoded'] = circuit_encoder.fit_transform(df['Circuit'])
    
    # Grid position (starting position)
    features_df['GridPosition'] = df['GridPosition'].fillna(20)
    
    # Previous race performance (simplified)
    features_df['Points'] = df['Points'].fillna(0)
    
    # Weather features
    features_df['Temperature'] = df['AvgTemp'].fillna(25)
    features_df['Humidity'] = df['AvgHumidity'].fillna(50)
    
    # Calculate driver's average finishing position in last 3 races
    driver_avg_pos = []
    for idx, row in df.iterrows():
        driver = row['Driver']
        year = row['Year']
        round_num = row['Round']
        
        prev_races = df[(df['Driver'] == driver) & 
                       ((df['Year'] == year) & (df['Round'] < round_num)) |
                       (df['Year'] < year)][-3:]
        
        if len(prev_races) > 0:
            avg_pos = prev_races['Position'].mean()
        else:
            avg_pos = 15  # Default middle position
            
        driver_avg_pos.append(avg_pos)
    
    features_df['AvgRecentPosition'] = driver_avg_pos
    
    # Target variable (finishing position)
    target = df['Position'].fillna(20)
    
    return features_df, target, driver_encoder, team_encoder, circuit_encoder

def train_model(features, target):
    """Train a Random Forest model to predict race positions"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score

def predict_race(model, driver_encoder, team_encoder, circuit_encoder, 
                 drivers_data, circuit, temperature=25, humidity=50):
    """Predict race results for given conditions"""
    predictions = []
    
    for _, driver in drivers_data.iterrows():
        try:
            features = pd.DataFrame({
                'DriverEncoded': [driver_encoder.transform([driver['Driver']])[0]],
                'TeamEncoded': [team_encoder.transform([driver['Team']])[0]],
                'CircuitEncoded': [circuit_encoder.transform([circuit])[0] if circuit in circuit_encoder.classes_ else 0],
                'GridPosition': [driver['GridPosition']],
                'Points': [driver['Points']],
                'Temperature': [temperature],
                'Humidity': [humidity],
                'AvgRecentPosition': [driver['AvgRecentPosition']]
            })
            
            predicted_position = model.predict(features)[0]
            predictions.append({
                'Driver': driver['Driver'],
                'Team': driver['Team'],
                'Predicted Position': round(predicted_position, 1),
                'Grid Position': driver['GridPosition']
            })
        except Exception as e:
            continue
    
    # Sort by predicted position
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('Predicted Position')
    predictions_df['Predicted Position'] = range(1, len(predictions_df) + 1)
    
    return predictions_df

# Main app
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Data Loading")
    
    # Year selection
    current_year = datetime.now().year
    years = st.multiselect(
        "Select years for training data:",
        options=list(range(2018, current_year + 1)),
        default=[current_year - 1, current_year]
    )
    
    sessions_per_year = st.slider(
        "Races per year to load:",
        min_value=1,
        max_value=10,
        value=5,
        help="Loading more races improves accuracy but takes longer"
    )
    
    if st.button("Load Data & Train Model", type="primary"):
        if years:
            with st.spinner("Loading historical data..."):
                historical_data = load_historical_data(years, sessions_per_year)
                
            if not historical_data.empty:
                st.success(f"Loaded {len(historical_data)} race results!")
                
                with st.spinner("Training prediction model..."):
                    features, target, driver_enc, team_enc, circuit_enc = prepare_features(historical_data)
                    model, train_score, test_score = train_model(features, target)
                
                st.session_state.data_loaded = True
                st.session_state.historical_data = historical_data
                st.session_state.model = model
                st.session_state.driver_encoder = driver_enc
                st.session_state.team_encoder = team_enc
                st.session_state.circuit_encoder = circuit_enc
                
                st.success("Model trained successfully!")
                st.metric("Training Accuracy", f"{train_score:.2%}")
                st.metric("Testing Accuracy", f"{test_score:.2%}")
            else:
                st.error("No data could be loaded. Please try different years.")
        else:
            st.warning("Please select at least one year.")

with col2:
    if st.session_state.data_loaded:
        st.subheader("🏁 Race Prediction")
        
        # Get current season data for prediction
        try:
            current_schedule = fastf1.get_event_schedule(current_year)
            races = current_schedule[current_schedule['EventFormat'] != 'testing']
            
            race_names = races['EventName'].tolist()
            selected_race = st.selectbox("Select upcoming race:", race_names)
            
            selected_circuit = races[races['EventName'] == selected_race]['Location'].iloc[0]
            
            col3, col4 = st.columns(2)
            with col3:
                temperature = st.slider("Expected Temperature (°C)", 10, 40, 25)
            with col4:
                humidity = st.slider("Expected Humidity (%)", 20, 90, 50)
            
            # Get current driver standings
            st.subheader("Starting Grid")
            
            # Create sample grid (you can modify this to get actual qualifying results)
            drivers = st.session_state.historical_data[
                st.session_state.historical_data['Year'] == st.session_state.historical_data['Year'].max()
            ][['Abbreviation', 'TeamName', 'Points']].drop_duplicates('Abbreviation')
            
            drivers.columns = ['Driver', 'Team', 'Points']
            drivers = drivers.dropna().head(20)  # Get top 20 drivers
            
            # Simulate grid positions
            grid_positions = st.columns(4)
            grid_data = []
            
            for i, (_, driver) in enumerate(drivers.iterrows()):
                with grid_positions[i % 4]:
                    grid_pos = st.number_input(
                        f"{driver['Driver']}",
                        min_value=1,
                        max_value=20,
                        value=i+1,
                        key=f"grid_{driver['Driver']}"
                    )
                    driver['GridPosition'] = grid_pos
                    driver['AvgRecentPosition'] = 10  # Simplified
                    grid_data.append(driver)
            
            if st.button("Predict Race Results", type="primary"):
                drivers_df = pd.DataFrame(grid_data)
                
                predictions = predict_race(
                    st.session_state.model,
                    st.session_state.driver_encoder,
                    st.session_state.team_encoder,
                    st.session_state.circuit_encoder,
                    drivers_df,
                    selected_circuit,
                    temperature,
                    humidity
                )
                
                st.subheader("🏆 Predicted Results")
                
                # Display podium
                if len(predictions) >= 3:
                    podium_cols = st.columns(3)
                    podium_emojis = ["🥇", "🥈", "🥉"]
                    
                    for i, col in enumerate(podium_cols):
                        with col:
                            st.metric(
                                f"{podium_emojis[i]} Position {i+1}",
                                predictions.iloc[i]['Driver'],
                                f"Team: {predictions.iloc[i]['Team']}"
                            )
                
                # Full results table
                st.dataframe(
                    predictions,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                fig = px.bar(
                    predictions.head(10),
                    x='Driver',
                    y='Predicted Position',
                    color='Team',
                    title="Top 10 Predicted Finishing Positions",
                    labels={'Predicted Position': 'Position'},
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_yaxis(autorange='reversed')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading current season data: {str(e)}")
    else:
        st.info("👈 Please load data and train the model first using the sidebar controls.")
        
        st.subheader("How it works:")
        st.markdown("""
        1. **Load Historical Data**: Select years of F1 data to train the model
        2. **Train Model**: The app uses Random Forest to learn from past race results
        3. **Set Race Conditions**: Choose the race and weather conditions
        4. **Configure Grid**: Set the starting positions from qualifying
        5. **Get Predictions**: The model predicts the most likely finishing order
        
        The model considers:
        - Historical driver and team performance
        - Starting grid positions
        - Circuit characteristics
        - Weather conditions
        - Recent form (last 3 races)
        """)

# Footer
st.divider()
st.caption("Data provided by FastF1 API | Predictions are estimates based on historical data")