# 🏎️ F1 Race Predictor

A machine learning-powered Formula 1 race predictor that uses historical data from the FastF1 API to predict race outcomes. Built with Streamlit for an interactive web interface.

## 🌟 Features

- **Real-time Data Collection**: Pulls historical F1 race data directly from the FastF1 API
- **Machine Learning Predictions**: Uses Random Forest algorithm to predict race finishing positions
- **Interactive Dashboard**: Clean, user-friendly Streamlit interface
- **Weather Integration**: Factors in temperature and humidity conditions
- **Customizable Grid Positions**: Set starting positions based on qualifying results
- **Visual Analytics**: Interactive charts and podium predictions
- **Data Caching**: Efficient local caching to minimize API calls

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/f1-race-predictor.git
cd f1-race-predictor
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit fastf1 pandas numpy scikit-learn plotly
```

## 📦 Dependencies

Create a `requirements.txt` file with:
```txt
streamlit>=1.28.0
fastf1>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
```

## 🎮 Usage

1. **Run the application**
```bash
streamlit run f1_predictor.py
```

2. **Access the web interface**
   - The app will automatically open in your browser
   - If not, navigate to `http://localhost:8501`

3. **Using the Predictor**
   - **Step 1**: Select years of historical data to train the model (e.g., 2022-2024)
   - **Step 2**: Choose how many races per year to load (more races = better accuracy but slower loading)
   - **Step 3**: Click "Load Data & Train Model" and wait for completion
   - **Step 4**: Select an upcoming race from the dropdown
   - **Step 5**: Set expected weather conditions (temperature and humidity)
   - **Step 6**: Configure the starting grid positions
   - **Step 7**: Click "Predict Race Results" to see predictions

## 🧠 How It Works

### Data Collection
- Fetches historical race data from FastF1 API
- Includes driver performance, team data, circuit information, and weather conditions
- Caches data locally to improve performance

### Feature Engineering
The model considers:
- **Driver Performance**: Historical finishing positions and championship points
- **Team Performance**: Constructor standings and reliability
- **Grid Position**: Starting position from qualifying
- **Circuit Characteristics**: Track-specific historical data
- **Weather Conditions**: Temperature and humidity effects
- **Recent Form**: Average position over last 3 races

### Machine Learning Model
- **Algorithm**: Random Forest Regressor with 100 estimators
- **Training**: 80/20 train-test split for validation
- **Encoding**: Label encoding for categorical variables (drivers, teams, circuits)
- **Output**: Predicted finishing positions for all drivers

## 📊 Model Performance

The model typically achieves:
- Training accuracy: 75-85%
- Testing accuracy: 65-75%

Note: F1 races have high variability due to crashes, mechanical failures, and strategy calls that are difficult to predict.

## 🛠️ Configuration Options

### Customize Training Data
- Adjust years included in training set
- Modify number of races loaded per year
- Balance between accuracy and loading time

### Weather Parameters
- Temperature range: 10-40°C
- Humidity range: 20-90%

### Grid Positions
- Manually set qualifying positions
- Import from actual qualifying results (when available)

## 📁 Project Structure

```
f1-race-predictor/
│
├── f1_predictor.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # Documentation
├── f1_cache/                # FastF1 cache directory (auto-created)
│   └── [cached data files]
└── .gitignore              # Git ignore file
```

## ⚠️ Known Limitations

1. **Prediction Accuracy**: Cannot account for random events (crashes, mechanical failures, safety cars)
2. **Data Availability**: Recent race data may not be immediately available
3. **API Rate Limits**: FastF1 API may have rate limitations
4. **Processing Time**: Initial data loading can take several minutes
5. **Missing Features**: Doesn't include tire strategy, pit stop timing, or practice session data

## 🔧 Troubleshooting

### Cache Directory Error
If you see `NotADirectoryError: Cache directory does not exist`, the app will now automatically create it.

### Slow Loading Times
- Reduce the number of races per year
- Use fewer years of historical data
- Check your internet connection

### Missing Data Warnings
- Normal for some races (sprint races, cancelled events)
- Model will continue with available data

### Memory Issues
- Reduce the amount of historical data
- Close other applications
- Increase system swap space

## 🚀 Future Enhancements

Potential improvements to implement:
- [ ] Real-time qualifying data integration
- [ ] Tire strategy analysis
- [ ] Safety car probability prediction
- [ ] Driver head-to-head comparisons
- [ ] Neural network models
- [ ] Practice session performance analysis
- [ ] Historical weather pattern analysis
- [ ] Team strategy pattern recognition
- [ ] Mechanical reliability statistics
- [ ] Export predictions to CSV/PDF

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FastF1**: For providing comprehensive F1 data API
- **Streamlit**: For the excellent web app framework
- **Formula 1**: For the exciting sport that inspired this project

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Consult FastF1 documentation for data-related questions

## 🔗 Useful Links

- [FastF1 Documentation](https://docs.fastf1.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [F1 Official Website](https://www.formula1.com/)

---

**Disclaimer**: This is a prediction tool for entertainment and educational purposes. Actual race results depend on numerous unpredictable factors. Always gamble responsibly if using predictions for betting purposes.
---

**⭐ Star this repository if you found it helpful!**
