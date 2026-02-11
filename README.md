# AI - Stock Market Trend Analysis - AI-Powered Predictive Analytics

A comprehensive machine learning system for stock market analysis featuring anomaly detection, clustering, and trend prediction with an interactive Streamlit interface.

**Project Report Link: https://docs.google.com/document/d/1F9g_Juo-4Gr9gBWMNUASkjTR6PAs3PoJE2WlL3wofLk/edit?usp=sharing**

**Project Presentation Link: https://docs.google.com/presentation/d/1rr_iXtF0ROMDM6uwtrK-Tw09z0VF5--jMn9ZddxPhBw/edit?usp=sharing**

**Project Demonstration Video: https://drive.google.com/file/d/1hXdy_NUFH7xgQRAr6rLJYpkPXmf9psQi/view?usp=sharing**

## 🚀 Live Demo

**[Try the Live App on Streamlit Cloud] https://ogezephwy73jhvsuz4rmcp.streamlit.app/** 

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Technical Architecture](#technical-architecture)
- [Model Performance](#model-performance)
- [Jupyter Notebook](#jupyter-notebook)
- [Important Disclaimers](#important-disclaimers)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a hybrid machine learning approach for stock market analysis, combining:
- **Isolation Forest** for anomaly detection
- **K-Means Clustering** for stock similarity analysis  
- **XGBoost Classifier** for trend prediction (UP/DOWN)

The system analyzes the top 50 companies by market capitalization using technical indicators and provides insights through an intuitive web interface.

## 📁 Project Structure

```
Stock-Market-Trend-Analysis/
├── app.py                          # Streamlit UI application
├── backend.py                      # Core ML models and analysis functions
├── requirements.txt                # Python dependencies
├── Data/                          # Data storage directory
│   ├── market_data_raw.csv        # Raw stock data from Yahoo Finance
│   ├── market_data_features.csv   # Processed data with technical indicators
│   ├── market_data_anomalies.csv  # Data with anomaly flags
│   ├── market_data_clusters.csv   # Data with cluster assignments
│   └── news_raw.csv               # News data (if available)
├── Models/                        # Trained ML models
│   ├── trend_model.pkl            # XGBoost classifier
│   ├── isolation_forest_model.pkl # Anomaly detection model
│   ├── kmeans_model.pkl           # Clustering model
│   └── scaler.pkl                 # Feature scaler
└── Notebooks/                     # Jupyter notebook with full analysis
    └── Stock_Market_Trend_Analysis.ipynb
```

## ✨ Features

### 🛡️ Mandatory Disclaimer System
- Full-screen warning before accessing the application
- Clear statement that this is **NOT financial advice**
- Model accuracy disclosure (~52-56%)
- User must acknowledge risks to proceed (session-based)

### 📊 Interactive Dashboard
- **Ticker Selection**: Top 5 stocks (NVDA, AAPL, MSFT, GOOG, AMZN) + custom input
- **Date Range**: Flexible date selection (up to 6 months)
- **Real-time Analysis**: On-demand ML analysis with progress indicators

### 🎯 AI-Powered Insights
- **Trend Prediction**: Binary UP/DOWN prediction with confidence scores
- **Anomaly Detection**: Identifies unusual price movements (10% contamination rate)
- **Pattern Recognition**: Groups stocks with similar technical behavior
- **Risk Assessment**: Highlights potential market anomalies

### 📈 Advanced Visualizations
- **Interactive Price Charts**: Plotly-powered with zoom/pan capabilities
- **Bollinger Bands**: Technical volatility indicators overlay
- **Anomaly Markers**: Red dots highlighting detected anomalies
- **Technical Indicators**: Real-time RSI, MACD, SMA calculations

### 🔗 Peer Analysis & Clustering
- **Similar Stocks**: K-Means clustering finds comparable stocks
- **Technical Similarity**: Based on 5 key technical indicators
- **Distance Metrics**: Quantified similarity percentages
- **Top 3 Matches**: Most similar stocks with similarity scores

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ 
- Internet connection for data fetching
- 2GB+ RAM recommended

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Stock-Market-Trend-Analysis.git
cd Stock-Market-Trend-Analysis
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

### 5. First-Time Setup
- The app will automatically download stock data on first run
- Models will be trained if not found in the `Models/` directory
- This initial setup may take 5-10 minutes

## 📖 Usage Guide

### Getting Started
1. **Accept Disclaimer**: Click "I Understand & Accept" on the mandatory warning screen
2. **Select Stock**: Choose from top 5 dropdown or enter custom ticker symbol
3. **Set Date Range**: Select analysis period (maximum 6 months)
4. **Run Analysis**: Click "Run Analysis" button and wait for results

### Understanding the Results

#### 🎯 Trend Prediction
- **📈 UP**: Model predicts next-day price increase
- **📉 DOWN**: Model predicts next-day price decrease  
- **Confidence**: Probability score (50-100%, higher = more confident)
- **Model Accuracy**: Historical performance (~52-56%)

#### 🚨 Anomaly Detection
- **🟢 Normal**: No unusual activity detected in recent data
- **🔴 Detected**: Unusual price movements identified
- **Count**: Number of anomalies detected in the analysis period
- **Markers**: Red dots on price chart show anomaly dates

#### 📊 Technical Indicators
- **RSI (14)**: Momentum indicator (0-100 scale)
- **SMA Ratio**: Short-term vs long-term trend strength
- **MACD Histogram**: Trend momentum and direction
- **BB Position**: Price position within Bollinger Bands
- **Volume Ratio**: Trading activity vs average

#### 🔗 Similar Stocks
- **Pattern Matching**: Stocks with similar technical behavior
- **Similarity %**: Quantified similarity score
- **Top 3 Matches**: Most comparable stocks based on recent patterns
- **Use Case**: Portfolio diversification and comparative analysis

## 🏗️ Technical Architecture

### Machine Learning Pipeline
```
Yahoo Finance API → Data Processing → Feature Engineering → ML Models → Streamlit UI
```

### Core Models
1. **Isolation Forest** (Anomaly Detection)
   - Unsupervised outlier detection
   - 10% contamination rate
   - Identifies unusual price movements

2. **K-Means Clustering** (Pattern Recognition)  
   - 5 clusters for stock grouping
   - StandardScaler normalization
   - Euclidean distance similarity

3. **XGBoost Classifier** (Trend Prediction)
   - Binary classification (UP/DOWN)
   - 70-30 train/test split
   - 500 estimators, 0.05 learning rate

### Technical Indicators (Features)
- **RSI (14)**: Relative Strength Index for momentum analysis
- **SMA Ratio (5/20)**: Short vs long-term moving average ratio
- **MACD Histogram**: Moving Average Convergence Divergence momentum
- **Bollinger Bands Position**: Price position within volatility bands
- **Volume Ratio**: Current volume vs 20-day average volume

### Data Pipeline
- **Source**: Yahoo Finance API (yfinance library)
- **Coverage**: Top 50 companies by market capitalization
- **Timeframe**: 6 months of daily data
- **Processing**: Robust downloading with retry logic
- **Storage**: CSV files for persistence and model training

## 📊 Model Performance

### XGBoost Classifier (Trend Prediction)
- **Accuracy**: 52-56% (varies with market conditions)
- **Baseline**: 50% (random prediction)
- **Improvement**: 4-12% over random baseline
- **Training Split**: 70% training, 30% testing
- **Features**: 7 technical indicators + anomaly flags

### Isolation Forest (Anomaly Detection)
- **Contamination Rate**: 10% (expected anomalies)
- **Detection Method**: Unsupervised outlier identification
- **Use Case**: Risk assessment and unusual activity flagging

### K-Means Clustering (Similarity Analysis)
- **Clusters**: 5 distinct behavioral groups
- **Silhouette Score**: Measures cluster quality
- **Distance Metric**: Euclidean distance for similarity
- **Applications**: Peer comparison and portfolio diversification

### Performance Notes
- Model accuracy varies by market regime (trending vs sideways)
- Better performance during clear trend periods
- Struggles with news-driven events and black swan scenarios
- Designed as decision support tool, not autonomous trading system

## 📓 Jupyter Notebook

The project includes a comprehensive Jupyter notebook (`Notebooks/Stock_Market_Trend_Analysis.ipynb`) with:

### Complete Analysis Pipeline
1. **Problem Definition & Objectives**
2. **Data Understanding & Preparation** 
3. **Model Design & Architecture**
4. **Core Implementation** (with performance metrics)
5. **Evaluation & Analysis**
6. **Ethical Considerations & Responsible AI**
7. **Conclusions & Future Scope**

### Key Features
- **Executable Code**: Run top-to-bottom without errors
- **Performance Metrics**: Comprehensive evaluation for each model
- **Visualizations**: Charts and plots for model analysis
- **Documentation**: Detailed explanations and methodology
- **Reproducible Results**: Fixed random seeds for consistency

### Running the Notebook
```bash
jupyter notebook Notebooks/Stock_Market_Trend_Analysis.ipynb
```

## ⚠️ Important Disclaimers

### Financial Disclaimer
- **NOT FINANCIAL ADVICE**: This tool is for educational purposes only
- **No Guarantees**: Past performance does not predict future results  
- **Limited Accuracy**: Model accuracy is approximately 52-56%
- **Professional Advice**: Always consult qualified financial advisors
- **Risk Warning**: Invest only what you can afford to lose

### Technical Limitations
- **6-Month Window**: Maximum historical analysis period
- **Large-Cap Focus**: Only top 50 companies by market cap
- **Daily Data**: No intraday or high-frequency analysis
- **Technical Only**: No fundamental analysis or news sentiment
- **Market Conditions**: Performance varies by market regime

## 🔧 Troubleshooting

### Common Issues & Solutions

#### "Could not fetch data for ticker"
- **Cause**: Invalid ticker symbol or network connectivity
- **Solution**: Verify ticker symbol, check internet connection, try different ticker

#### "Models not loaded" Error  
- **Cause**: Missing or corrupted model files
- **Solution**: Delete `Models/` folder and restart app (models will retrain)

#### "Insufficient data" Warning
- **Cause**: Not enough historical data for analysis
- **Solution**: Try different date range or ticker with more trading history

#### Slow Performance
- **Cause**: Large dataset processing or model training
- **Solution**: Use shorter date ranges, ensure sufficient RAM (2GB+)

#### Analysis Fails Silently
- **Cause**: API rate limiting or data processing errors
- **Solution**: Wait 1-2 minutes and retry, check console for error messages

### Debug Mode
Enable debug mode in the sidebar to see detailed error information and processing steps.

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 2GB minimum, 4GB recommended  
- **Storage**: 500MB for data and models
- **Internet**: Required for data fetching

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data API
- **Streamlit** for the excellent web app framework
- **scikit-learn & XGBoost** for machine learning capabilities
- **Plotly** for interactive visualizations

---

## 📞 Support & Contact

- **Email**: pranavarunkumaraj@example.com

---

**⚠️ Final Reminder**: This tool is for educational and research purposes only. Always conduct your own research and consult with qualified financial professionals before making any investment decisions. The creators are not responsible for any financial losses incurred from using this tool.
