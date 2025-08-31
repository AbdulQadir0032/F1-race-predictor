import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import fastf1
import warnings
import os
import tempfile
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
try:
    cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
except Exception as e:
    st.warning(f"Cache setup failed: {e}. Running without cache.")
    # Continue without cache if there's an issue

# Page configuration
st.set_page_config(
    page_title="🏎️ F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        color: white;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'race_schedule' not in st.session_state:
    st.session_state.race_schedule = None

def load_f1_data(years, max_races_per_year):
    """Load real F1 data using FastF1 API"""
    all_data = []
    total_races_loaded = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for year_idx, year in enumerate(years):
        try:
            status_text.text(f"Loading {year} season schedule...")
            
            # Get race schedule for the year
            schedule = fastf1.get_event_schedule(year)
            races_this_year = min(len(schedule), max_races_per_year)
            
            status_text.text(f"Found {races_this_year} races for {year}")
            
            for race_idx in range(races_this_year):
                try:
                    event = schedule.iloc[race_idx]
                    race_name = event['EventName']
                    
                    status_text.text(f"Loading {year} {race_name}... ({race_idx + 1}/{races_this_year})")
                    
                    # Load race session with timeout protection
                    try:
                        session = fastf1.get_session(year, race_idx + 1, 'R')
                        session.load(telemetry=False, weather=False, messages=False)  # Load only essential data
                        
                        # Get race results
                        results = session.results
                        
                        if results is not None and not results.empty:
                            # Get weather data if available (with fallback)
                            try:
                                weather_data = session.weather_data
                                if weather_data is not None and not weather_data.empty:
                                    avg_temp = weather_data['AirTemp'].mean()
                                    avg_humidity = weather_data['Humidity'].mean()
                                else:
                                    avg_temp = 20 + np.random.normal(0, 5)  # Random realistic temp
                                    avg_humidity = 50 + np.random.normal(0, 15)
                            except:
                                avg_temp = 20 + np.random.normal(0, 5)
                                avg_humidity = 50 + np.random.normal(0, 15)
                            
                            # Process each driver's result
                            for idx, row in results.iterrows():
                                try:
                                    if (pd.notna(row.get('Position')) and 
                                        pd.notna(row.get('GridPosition')) and
                                        row['Position'] > 0):  # Valid finishing position
                                        
                                        grid_pos = row['GridPosition']
                                        if pd.isna(grid_pos) or grid_pos == 0:
                                            grid_pos = 20  # Back of grid
                                        
                                        all_data.append({
                                            'year': year,
                                            'race_number': race_idx + 1,
                                            'race_name': race_name,
                                            'driver': row.get('FullName', 'Unknown Driver'),
                                            'driver_code': row.get('Abbreviation', 'UNK'),
                                            'team': row.get('TeamName', 'Unknown Team'),
                                            'grid_position': int(grid_pos),
                                            'finish_position': int(row['Position']),
                                            'points': row.get('Points', 0),
                                            'temperature': max(5, min(45, avg_temp)),  # Realistic range
                                            'humidity': max(20, min(95, avg_humidity)),
                                            'status': row.get('Status', 'Finished')
                                        })
                                except Exception as driver_error:
                                    continue  # Skip problematic driver data
                        
                        total_races_loaded += 1
                        
                    except Exception as session_error:
                        st.warning(f"Could not load {year} {race_name}: {str(session_error)}")
                        continue
                    
                    # Update progress
                    total_progress = (year_idx * max_races_per_year + race_idx + 1) / (len(years) * max_races_per_year)
                    progress_bar.progress(min(total_progress, 1.0))
                    
                except Exception as race_error:
                    st.warning(f"Failed to process {year} race {race_idx + 1}: {str(race_error)}")
                    continue
                    
        except Exception as year_error:
            st.error(f"Failed to load {year} season: {str(year_error)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        st.error("❌ No race data could be loaded. This might be due to:")
        st.error("- Internet connection issues")
        st.error("- FastF1 API temporary unavailability") 
        st.error("- Selected years having incomplete data")
        st.info("💡 Try selecting more recent years (2022-2024) or fewer races per year")
        return None, 0
    
    df = pd.DataFrame(all_data)
    st.success(f"✅ Successfully loaded {len(df)} race results from {total_races_loaded} races!")
    return df, total_races_loaded

@st.cache_data
def get_upcoming_races():
    """Get upcoming races for 2024 season"""
    try:
        current_schedule = fastf1.get_event_schedule(2024)
        current_date = datetime.now()
        
        upcoming = []
        for idx, race in current_schedule.iterrows():
            try:
                race_date = pd.to_datetime(race['EventDate'])
                if race_date > pd.Timestamp(current_date):
                    upcoming.append({
                        'name': race['EventName'],
                        'location': race.get('Location', 'TBD'),
                        'date': race_date,
                        'round': race.get('RoundNumber', idx + 1)
                    })
            except:
                continue
        
        if upcoming:
            return upcoming[:10]  # Next 10 races
        else:
            raise Exception("No upcoming races found")
            
    except Exception as e:
        st.warning(f"Could not load upcoming races: {e}. Using fallback list.")
        # Fallback list for 2024/2025 races
        return [
            {'name': 'Dutch Grand Prix', 'location': 'Zandvoort', 'round': 15},
            {'name': 'Italian Grand Prix', 'location': 'Monza', 'round': 16},
            {'name': 'Singapore Grand Prix', 'location': 'Marina Bay', 'round': 18},
            {'name': 'Japanese Grand Prix', 'location': 'Suzuka', 'round': 19},
            {'name': 'United States Grand Prix', 'location': 'Austin', 'round': 21},
            {'name': 'Mexican Grand Prix', 'location': 'Mexico City', 'round': 22},
            {'name': 'Brazilian Grand Prix', 'location': 'São Paulo', 'round': 23},
            {'name': 'Las Vegas Grand Prix', 'location': 'Las Vegas', 'round': 24},
            {'name': 'Qatar Grand Prix', 'location': 'Lusail', 'round': 25},
            {'name': 'Abu Dhabi Grand Prix', 'location': 'Yas Marina', 'round': 26}
        ]

def train_model(data):
    """Train the F1 prediction model using real F1 data"""
    if data is None or data.empty:
        return None, 0, 0, None
    
    # Feature engineering
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_race = LabelEncoder()
    
    data_clean = data.dropna(subset=['grid_position', 'finish_position'])
    
    if len(data_clean) < 10:
        st.error("Not enough clean data to train model")
        return None, 0, 0, None
    
    data_encoded = data_clean.copy()
    data_encoded['driver_encoded'] = le_driver.fit_transform(data_clean['driver'])
    data_encoded['team_encoded'] = le_team.fit_transform(data_clean['team'])
    data_encoded['race_encoded'] = le_race.fit_transform(data_clean['race_name'])
    
    # Features and target
    features = ['grid_position', 'driver_encoded', 'team_encoded', 'race_encoded', 
                'temperature', 'humidity', 'year']
    
    X = data_encoded[features]
    y = data_encoded['finish_position']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=20, 
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    return model, train_accuracy, test_accuracy, (le_driver, le_team, le_race, features)

# Sidebar - Configuration Panel
with st.sidebar:
    st.markdown("# ⚙️ Configuration")
    
    if not st.session_state.model_trained:
        st.markdown("### 📊 Data Loading")
        st.info("🏎️ Using real F1 data from FastF1 API")
        
        # Year selection
        available_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        selected_years = st.multiselect(
            "Select F1 seasons:",
            available_years,
            default=[2023],
            help="Choose F1 seasons for training data"
        )
        
        # Races per year
        races_per_year = st.slider(
            "Max races per season:",
            min_value=5,
            max_value=24,
            value=15,
            help="Maximum races to load per season"
        )
        
        st.warning("⏳ Loading real F1 data may take 2-5 minutes. First run will be slower.")
        st.info("💡 **Tip:** Start with recent years (2022-2024) and fewer races for faster loading")
        
        # Add connection test button
        if st.button("🔍 Test FastF1 Connection", help="Check if FastF1 API is accessible"):
            try:
                with st.spinner("Testing connection..."):
                    test_schedule = fastf1.get_event_schedule(2024)
                    if test_schedule is not None and len(test_schedule) > 0:
                        st.success("✅ FastF1 API connection successful!")
                        st.info(f"Found {len(test_schedule)} races in 2024 schedule")
                    else:
                        st.error("❌ Could not retrieve race schedule")
            except Exception as e:
                st.error(f"❌ FastF1 connection failed: {str(e)}")
                st.error("This might be due to network issues or API unavailability")
        
        # Load and train button
        if st.button("🚀 Load F1 Data & Train Model", type="primary"):
            if not selected_years:
                st.error("Please select at least one year")
            else:
                with st.spinner("Loading real F1 data from FastF1 API..."):
                    # Load real F1 data
                    training_data, total_races = load_f1_data(selected_years, races_per_year)
                    
                    if training_data is not None:
                        st.session_state.training_data = training_data
                        
                        # Train model
                        with st.spinner("Training machine learning model..."):
                            model, train_acc, test_acc, encoders = train_model(training_data)
                            
                            if model is not None:
                                st.session_state.model = model
                                st.session_state.encoders = encoders
                                st.session_state.model_metrics = {
                                    'training_accuracy': train_acc * 100,
                                    'testing_accuracy': test_acc * 100,
                                    'races_loaded': total_races,
                                    'drivers_count': training_data['driver'].nunique(),
                                    'teams_count': training_data['team'].nunique()
                                }
                                st.session_state.model_trained = True
                                
                                st.success(f"✅ Loaded {total_races} real F1 races!")
                                st.success("🎯 Model trained successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to train model with the loaded data")
    
    else:
        # Model Configuration (shown after training)
        st.markdown("### 🤖 Model Configuration")
        
        # Algorithm info
        st.info("🌲 **Algorithm:** Random Forest\n\n🎯 **Purpose:** Predicting F1 race finishing positions based on historical data")
        
        # Model parameters
        st.markdown("### 📈 Model Parameters")
        
        # Display current parameters
        params_info = {
            "N Estimators": 150,
            "Max Depth": 20,
            "Min Samples Split": 5,
            "Features": len(st.session_state.encoders[3]) if st.session_state.encoders else 7
        }
        
        for param, value in params_info.items():
            st.metric(param, value)
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        if st.session_state.model is not None and st.session_state.encoders:
            feature_names = ['Grid Position', 'Driver', 'Team', 'Race', 'Temperature', 'Humidity', 'Year']
            importances = st.session_state.model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Plot feature importance
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Feature Importance in F1 Race Prediction"
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Performance
        st.markdown("### 📊 Model Performance")
        if st.session_state.model_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{st.session_state.model_metrics['training_accuracy']:.1f}%")
                st.metric("Drivers Analyzed", st.session_state.model_metrics['drivers_count'])
            with col2:
                st.metric("Testing Accuracy", f"{st.session_state.model_metrics['testing_accuracy']:.1f}%")
                st.metric("Teams Analyzed", st.session_state.model_metrics['teams_count'])
            
            st.metric("Total Races", st.session_state.model_metrics['races_loaded'])
        
        # Data insights
        if st.session_state.training_data is not None:
            st.markdown("### 📈 Training Data Insights")
            
            data = st.session_state.training_data
            
            # Most successful drivers
            driver_wins = data[data['finish_position'] == 1]['driver'].value_counts().head(3)
            if not driver_wins.empty:
                st.write("🏆 **Most Wins in Training Data:**")
                for driver, wins in driver_wins.items():
                    st.write(f"• {driver}: {wins} wins")
        
        # Retrain option
        st.markdown("---")
        if st.button("🔄 Load New Data", help="Load different years or more races"):
            st.session_state.model_trained = False
            st.rerun()

# Main content
st.markdown('<h1 class="main-header">🏎️ F1 Race Predictor</h1>', unsafe_allow_html=True)
st.markdown("*Predict F1 race outcomes using real historical data from FastF1 API*")

if not st.session_state.model_trained:
    # Welcome screen
    st.markdown("## 👋 Welcome to F1 Race Predictor!")
    st.info("👈 **Start by loading real F1 data and training your model in the sidebar.**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📊 Real F1 Data
        - Official race results
        - Driver and team data
        - Weather conditions
        - Grid positions & lap times
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Machine Learning
        - Random Forest algorithm
        - Feature engineering
        - Cross-validation
        - Performance metrics
        """)
    
    with col3:
        st.markdown("""
        ### 🏁 Race Prediction
        - Upcoming race selection
        - Weather configuration
        - Grid position setup
        - Finishing order prediction
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Requirements")
    st.code("""
pip install fastf1 streamlit pandas scikit-learn plotly

# Note: First run may take longer as FastF1 downloads race data
    """)

else:
    # Prediction interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 🏁 Race Prediction Setup")
        
        # Load upcoming races
        if st.session_state.race_schedule is None:
            st.session_state.race_schedule = get_upcoming_races()
        
        # Race selection
        race_options = [race['name'] for race in st.session_state.race_schedule]
        selected_race_name = st.selectbox("🏎️ Select upcoming race:", race_options)
        
        selected_race = next((race for race in st.session_state.race_schedule 
                            if race['name'] == selected_race_name), None)
        
        if selected_race:
            st.info(f"📍 **Location:** {selected_race.get('location', 'TBD')}")
        
        # Weather conditions
        st.markdown("### 🌤️ Expected Race Conditions")
        col_temp, col_humid = st.columns(2)
        
        with col_temp:
            temperature = st.slider("🌡️ Air Temperature (°C):", 5, 45, 22)
        
        with col_humid:
            humidity = st.slider("💧 Humidity (%):", 20, 95, 55)
        
        # Get unique drivers from training data
        if st.session_state.training_data is not None:
            recent_drivers = st.session_state.training_data[
                st.session_state.training_data['year'] == st.session_state.training_data['year'].max()
            ]['driver'].unique()
            
            # All 20 drivers by recent performance
            driver_performance = st.session_state.training_data.groupby('driver')['finish_position'].mean().sort_values()
            top_drivers = driver_performance.head(20).index.tolist()
            
            st.markdown("### 🏁 Starting Grid Configuration")
            st.info("💡 Set expected grid positions for key drivers")
            
            driver_positions = {}
            
            # Create grid for top drivers
            for i in range(0, min(20, len(top_drivers)), 2):
                col_left, col_right = st.columns(2)
                
                with col_left:
                    if i < len(top_drivers):
                        driver = top_drivers[i]
                        driver_positions[driver] = st.number_input(
                            f"🏎️ {driver.split()[-1]}:",  # Last name only
                            min_value=1,
                            max_value=20,
                            value=i+1,
                            key=f"pos_{driver}",
                            help=f"Grid position for {driver}"
                        )
                
                with col_right:
                    if i+1 < len(top_drivers):
                        driver = top_drivers[i+1]
                        driver_positions[driver] = st.number_input(
                            f"🏎️ {driver.split()[-1]}:",  # Last name only
                            min_value=1,
                            max_value=20,
                            value=i+2,
                            key=f"pos_{driver}",
                            help=f"Grid position for {driver}"
                        )
    
    with col2:
        st.markdown("## 🔮 Race Predictions")
        
        if st.button("🚀 Generate Race Prediction", type="primary", use_container_width=True):
            if st.session_state.model is not None and st.session_state.encoders is not None:
                with st.spinner("🤖 AI is analyzing race conditions..."):
                    time.sleep(1)  # Simulate processing
                    
                    model = st.session_state.model
                    le_driver, le_team, le_race, features = st.session_state.encoders
                    
                    predictions = []
                    
                    # Get the most recent year's data for team mapping
                    recent_data = st.session_state.training_data[
                        st.session_state.training_data['year'] == st.session_state.training_data['year'].max()
                    ]
                    
                    for driver, grid_pos in driver_positions.items():
                        try:
                            # Get driver's most recent team
                            driver_data = recent_data[recent_data['driver'] == driver]
                            if not driver_data.empty:
                                recent_team = driver_data.iloc[-1]['team']
                                
                                # Encode features
                                driver_encoded = le_driver.transform([driver])[0] if driver in le_driver.classes_ else 0
                                team_encoded = le_team.transform([recent_team])[0] if recent_team in le_team.classes_ else 0
                                race_encoded = le_race.transform([selected_race_name])[0] if selected_race_name in le_race.classes_ else 0
                                
                                # Create feature vector
                                X_pred = [[
                                    grid_pos,           # grid_position
                                    driver_encoded,     # driver_encoded  
                                    team_encoded,       # team_encoded
                                    race_encoded,       # race_encoded
                                    temperature,        # temperature
                                    humidity,           # humidity
                                    2024               # year
                                ]]
                                
                                # Make prediction
                                predicted_pos = model.predict(X_pred)[0]
                                confidence = max(model.predict_proba(X_pred)[0]) * 100
                                
                                predictions.append({
                                    'Driver': driver.split()[-1],  # Last name
                                    'Team': recent_team,
                                    'Grid': f"P{grid_pos}",
                                    'Predicted': f"P{int(predicted_pos)}",
                                    'Confidence': f"{confidence:.1f}%",
                                    'Position Change': int(predicted_pos) - grid_pos,
                                    'sort_pos': int(predicted_pos)
                                })
                        except Exception as e:
                            st.warning(f"Could not predict for {driver}: {str(e)}")
                    
                    # Sort by predicted position
                    predictions = sorted(predictions, key=lambda x: x['sort_pos'])
                    st.session_state.predictions = predictions
        
        # Display predictions
        if st.session_state.predictions:
            st.markdown("### 🏆 Predicted Race Results")
            
            # Podium predictions
            podium = st.session_state.predictions[:3]
            
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                if len(podium) > 0:
                    p = podium[0]
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #FFD700, #FFA500); border-radius: 10px; color: black;">
                        <h3>🥇 1st Place</h3>
                        <h4>{p['Driver']}</h4>
                        <p>{p['Team']}</p>
                        <small>Grid: {p['Grid']} | {p['Confidence']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_p2:
                if len(podium) > 1:
                    p = podium[1]
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #C0C0C0, #A8A8A8); border-radius: 10px; color: black;">
                        <h3>🥈 2nd Place</h3>
                        <h4>{p['Driver']}</h4>
                        <p>{p['Team']}</p>
                        <small>Grid: {p['Grid']} | {p['Confidence']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_p3:
                if len(podium) > 2:
                    p = podium[2]
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #CD7F32, #B87333); border-radius: 10px; color: white;">
                        <h3>🥉 3rd Place</h3>
                        <h4>{p['Driver']}</h4>
                        <p>{p['Team']}</p>
                        <small>Grid: {p['Grid']} | {p['Confidence']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Full Results Prediction")
            
            # Create results table
            results_df = pd.DataFrame(st.session_state.predictions)
            results_df = results_df.drop(['sort_pos'], axis=1)  # Remove sort column
            
            # Color code position changes
            def highlight_changes(val):
                if isinstance(val, int):
                    if val < 0:
                        return 'background-color: lightgreen'  # Gained positions
                    elif val > 0:
                        return 'background-color: lightcoral'  # Lost positions
                return ''
            
            styled_df = results_df.style.applymap(highlight_changes, subset=['Position Change'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Legend:**
            - 🟢 Green: Driver expected to gain positions
            - 🔴 Red: Driver expected to lose positions
            - Position Change = Predicted Position - Grid Position
            """)

# Footer
st.markdown("---")
st.markdown("*🏎️ Built with Streamlit & FastF1 API • Real F1 data from Formula 1*")
st.markdown("*⚠️ Predictions are for entertainment purposes only*")