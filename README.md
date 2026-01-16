# Stock Market Trend Analysis - Streamlit Application

A comprehensive machine learning-powered stock market analysis tool with an interactive Streamlit interface.

## Project Structure

```
├── app.py              # Streamlit UI application
├── backend.py          # Core analysis functions and ML models
├── requirements.txt    # Python dependencies
├── Data/              # Data storage directory
│   ├── market_data_features.csv
│   ├── market_data_anomalies.csv
│   └── market_data_clusters.csv
├── Models/            # Trained ML models
│   ├── trend_model.pkl
│   ├── isolation_forest_model.pkl
│   ├── kmeans_model.pkl
│   └── scaler.pkl
└── Notebooks/         # Jupyter notebooks (original analysis)
```

## Features

### 🛡️ Mandatory Disclaimer
- Full-screen warning before accessing the application
- Clear statement that this is NOT financial advice
- Model accuracy disclosure (53.90%)
- User must acknowledge risks to proceed

### 📊 Interactive Dashboard
- **Ticker Selection**: Top 5 stocks (NVDA, AAPL, MSFT, GOOG, AMZN) + custom input
- **Date Range**: Flexible date selection (up to 6 months)
- **Real-time Analysis**: On-demand XGBoost analysis

### 🎯 Key Metrics Display
- **Trend Prediction**: UP/DOWN with color coding
- **Model Accuracy**: Historical performance (53.90%)
- **AI Confidence**: Probability score of predictions
- **Anomaly Status**: Detection of unusual price movements

### 📈 Advanced Visualizations
- **Interactive Price Charts**: Plotly-powered with zoom/pan
- **Bollinger Bands**: Technical volatility indicators
- **Anomaly Markers**: Red dots for detected anomalies
- **Technical Indicators**: RSI, MACD, SMA ratios

### 🔗 Peer Analysis
- **Similar Stocks**: K-Means clustering to find comparable stocks
- **Technical Similarity**: Based on recent pattern analysis
- **Behavioral Grouping**: Stocks with similar technical characteristics

## Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### First Time Setup
1. **Accept Disclaimer**: Click "I Understand & Accept" on the warning screen
2. **Select Stock**: Choose from dropdown or enter custom ticker
3. **Set Date Range**: Select analysis period (within last 6 months)
4. **Run Analysis**: Click "Run XGBoost Analysis" button

### Understanding Results

#### Trend Prediction
- **UP (📈)**: Model predicts price increase
- **DOWN (📉)**: Model predicts price decrease
- **Confidence**: Probability score (higher = more confident)

#### Anomaly Detection
- **Normal (🟢)**: No unusual activity detected
- **Detected (🔴)**: Unusual price movements identified
- **Count**: Number of anomalies in the period

#### Similar Stocks
- Based on technical indicator clustering
- Shows stocks with similar recent patterns
- Useful for comparative analysis

## Technical Details

### Machine Learning Models
1. **XGBoost Classifier**: Binary trend prediction (UP/DOWN)
2. **Isolation Forest**: Anomaly detection in price movements
3. **K-Means Clustering**: Stock similarity grouping

### Technical Indicators
- **RSI (14)**: Relative Strength Index for momentum
- **SMA Ratio (5/20)**: Short vs long-term moving averages
- **MACD Histogram**: Trend momentum indicator
- **Bollinger Bands**: Price volatility bands
- **Volume Ratio**: Trading volume analysis

### Data Sources
- **Yahoo Finance API**: Real-time stock data via yfinance
- **Historical Data**: Pre-processed CSV files for clustering
- **6-Month Window**: Maximum analysis period

## Architecture

### Backend (`backend.py`)
- `StockAnalyzer` class: Main analysis engine
- Model loading and training functions
- Data fetching and processing pipeline
- Technical indicator calculations

### Frontend (`app.py`)
- Streamlit interface components
- Interactive charts and visualizations
- User input handling and validation
- Results display and formatting

## Model Performance

- **Training Accuracy**: 53.90%
- **Prediction Type**: Binary classification (UP/DOWN)
- **Features Used**: 7 technical indicators + anomaly flags
- **Training Data**: Historical stock data from top 50 companies

## Important Notes

### ⚠️ Disclaimers
- **Not Financial Advice**: For educational purposes only
- **Limited Accuracy**: 53.90% historical accuracy
- **Market Volatility**: Models may not predict unprecedented events
- **Professional Advice**: Always consult qualified financial advisors

### 🔧 Technical Limitations
- **6-Month Window**: Maximum analysis period
- **Model Dependencies**: Requires sufficient historical data
- **API Limits**: Yahoo Finance rate limiting may apply
- **Processing Time**: Analysis may take 30-60 seconds

## Troubleshooting

### Common Issues

**Models Not Loading**
- Ensure all `.pkl` files exist in `Models/` directory
- Check file permissions and paths

**No Data Available**
- Verify ticker symbol is valid
- Try a different date range
- Check internet connection

**Analysis Fails**
- Ensure sufficient historical data (>50 days)
- Try with a different stock ticker
- Check for API rate limits

### Error Messages
- **"Could not fetch data"**: Invalid ticker or network issue
- **"Insufficient data"**: Not enough historical data points
- **"Models not loaded"**: Missing or corrupted model files

## Development

### Adding New Features
1. Add backend functions to `backend.py`
2. Create UI components in `app.py`
3. Update requirements if new dependencies added
4. Test with various stock tickers

### Model Retraining
- Models automatically retrain if not found
- Use Jupyter notebooks for experimentation
- Save new models to `Models/` directory

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure data and model files are present
4. Check console output for detailed error messages

---

**Remember**: This tool is for educational purposes only. Always do your own research and consult with financial professionals before making investment decisions.