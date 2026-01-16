"""
Stock Market Analysis Backend
Contains all the core functions for data processing, model training, and predictions
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import xgboost as xgb
from datetime import datetime, timedelta

# Configuration
TOP_50_TICKERS = [
    "NVDA", "AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "AVGO", "2222.SR", "TSM",
    "BRK-B", "LLY", "WMT", "JPM", "TCEHY", "V", "ORCL", "MA", "005930.KS", "XOM",
    "JNJ", "PLTR", "BAC", "ASML", "ABBV", "NFLX", "601288.SS", "COST", "MC.PA", "BABA",
    "1398.HK", "AMD", "HD", "601939.SS", "ROG.SW", "PG", "GE", "MU", "CSCO", "KO",
    "WFC", "CVX", "UNH", "MS", "SAP", "TM", "AZN", "IBM", "CAT", "000660.KS"
]

FEATURE_COLUMNS = [
    'RSI_14',
    'SMA_Ratio_5_20', 
    'MACD_Histogram',
    'BB_Position',
    'Volume_Ratio'
]

# Directory setup
current_dir = Path.cwd()
project_root = current_dir
data_dir = project_root / "Data"
models_dir = project_root / "Models"

# Create directories if they don't exist
data_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

class StockAnalyzer:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        try:
            self.models['isolation_forest'] = joblib.load(models_dir / "isolation_forest_model.pkl")
            self.models['kmeans'] = joblib.load(models_dir / "kmeans_model.pkl") 
            self.models['trend'] = joblib.load(models_dir / "trend_model.pkl")
            self.scaler = joblib.load(models_dir / "scaler.pkl")
            print("✓ Models loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load models: {e}")
            print("Models will be trained when needed")
    
    def fetch_tickers_in_batches(self, tickers, batch_size=10, period="6mo"):
        """Fetch stock data in batches to avoid API limits"""
        all_data = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            print(f"Downloading batch: {batch}")
            try:
                data = yf.download(batch, period=period, group_by='ticker', auto_adjust=True, threads=True)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                print(f"Batch download error: {e}")
            time.sleep(1)
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()

    def robust_downloader(self, tickers, period="6mo", max_retries=3):
        """Download stock data with retry logic"""
        print(f"Initiating resilient download for {len(tickers)} tickers...")
        df = self.fetch_tickers_in_batches(tickers, batch_size=15, period=period)
        
        for attempt in range(max_retries):
            existing_tickers = df.columns.get_level_values(0).unique().tolist()
            failed_tickers = [t for t in tickers if t not in existing_tickers]
            
            if not failed_tickers:
                print("All tickers downloaded successfully.")
                break
                
            wait_time = (attempt + 1) * 5
            print(f"Attempt {attempt + 1}/{max_retries}: {len(failed_tickers)} failures. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
            for ticker in failed_tickers:
                try:
                    retry_data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                    if not retry_data.empty and len(retry_data) > 20:
                        if not isinstance(retry_data.columns, pd.MultiIndex):
                            retry_data.columns = pd.MultiIndex.from_product([[ticker], retry_data.columns])
                        df = pd.concat([df, retry_data], axis=1)
                        print(f"✓ Successfully retrieved {ticker}")
                except Exception as e:
                    print(f"✗ Failed for {ticker}: {str(e)[:50]}")

        df = df.sort_index().interpolate(method='time').ffill().bfill()
        
        missing_final = [t for t in tickers if t not in df.columns.get_level_values(0).unique()]
        if missing_final:
            print(f"⚠ CRITICAL: Still missing {len(missing_final)} tickers: {missing_final}")
        
        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators for stock data"""
        df = df.copy()
        
        # RSI_14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # SMA Ratio 5/20
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_Ratio_5_20'] = df['SMA_5'] / df['SMA_20']

        # MACD Histogram
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']

        # Bollinger Bands Position
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + (std * 2)
        df['BB_Lower'] = df['SMA_20'] - (std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Volume Ratio
        df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Vol_SMA_20']

        return df.dropna()

    def fetch_and_process_stock(self, ticker, start_date=None, end_date=None):
        """Fetch and process a single stock"""
        try:
            print(f"Fetching data for {ticker}...")
            
            # Try different approaches to fetch data
            stock_data = None
            
            # If date range provided, fetch extra data before start_date for indicator calculation
            # We need ~30 extra days for SMA_20 and other indicators
            if start_date and end_date:
                try:
                    # Convert to datetime if needed
                    if isinstance(start_date, str):
                        start_dt = pd.to_datetime(start_date)
                    else:
                        start_dt = pd.to_datetime(start_date)
                    
                    # Fetch 40 extra days before start_date for indicator warmup
                    extended_start = start_dt - timedelta(days=40)
                    
                    stock_data = yf.download(ticker, start=extended_start, end=end_date, progress=False)
                    print(f"Fetched data from {extended_start.date()} to {end_date} for indicator calculation")
                except Exception as e:
                    print(f"Date range download failed for {ticker}: {e}")
            
            # Method 2: Use period if date range failed or not provided
            if stock_data is None or stock_data.empty:
                try:
                    stock_data = yf.download(ticker, period="6mo", progress=False)
                except Exception as e:
                    print(f"Period download failed for {ticker}: {e}")
            
            # Method 3: Try with different parameters
            if stock_data is None or stock_data.empty:
                try:
                    stock_data = yf.download(ticker, period="3mo", auto_adjust=False, progress=False)
                except Exception as e:
                    print(f"Alternative download failed for {ticker}: {e}")
            
            # Check if we got any data
            if stock_data is None or stock_data.empty:
                print(f"No data retrieved for {ticker}")
                return None
            
            print(f"Retrieved {len(stock_data)} rows for {ticker}")
            
            # Handle MultiIndex columns (when downloading single ticker)
            if isinstance(stock_data.columns, pd.MultiIndex):
                # Flatten the MultiIndex columns
                stock_data.columns = stock_data.columns.get_level_values(0)
            
            # Reset index to get Date as column
            stock_data = stock_data.reset_index()
            
            # Ensure we have the required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            
            if missing_columns:
                print(f"Missing columns for {ticker}: {missing_columns}")
                print(f"Available columns: {list(stock_data.columns)}")
                return None
            
            # Check for sufficient data
            if len(stock_data) < 30:
                print(f"Insufficient data for {ticker}: only {len(stock_data)} rows")
                return None
            
            # Calculate indicators
            processed_data = self.calculate_indicators(stock_data)
            
            if processed_data is None or len(processed_data) == 0:
                print(f"No data after processing indicators for {ticker}")
                return None
            
            # Filter to only include data from the requested start_date onwards
            if start_date:
                start_dt = pd.to_datetime(start_date)
                processed_data = processed_data[processed_data['Date'] >= start_dt]
                print(f"Filtered to {len(processed_data)} rows from {start_date} onwards")
            
            processed_data['Ticker'] = ticker
            print(f"Successfully processed {len(processed_data)} rows for {ticker}")
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect_anomalies(self, df, ticker=None):
        if 'isolation_forest' not in self.models:
            iso_model = IsolationForest(
                n_estimators=100, 
                max_samples='auto', 
                contamination=0.10, 
                random_state=42, 
                max_features=len(FEATURE_COLUMNS), 
                n_jobs=-1
            )
            clean_df = df.dropna(subset=FEATURE_COLUMNS).copy()
            iso_model.fit(clean_df[FEATURE_COLUMNS])
            self.models['isolation_forest'] = iso_model
            joblib.dump(iso_model, models_dir / "isolation_forest_model.pkl")
        
        clean_df = df.dropna(subset=FEATURE_COLUMNS).copy()
        if len(clean_df) == 0:
            return df
            
        anomaly_flags = self.models['isolation_forest'].predict(clean_df[FEATURE_COLUMNS])
        clean_df['Anomaly_Flag'] = anomaly_flags
        clean_df['Is_Anomaly'] = clean_df['Anomaly_Flag'].apply(lambda x: 1 if x == -1 else 0)
        
        result_df = df.copy()
        result_df['Anomaly_Flag'] = 1
        result_df['Is_Anomaly'] = 0
        
        result_df.loc[clean_df.index, 'Anomaly_Flag'] = clean_df['Anomaly_Flag']
        result_df.loc[clean_df.index, 'Is_Anomaly'] = clean_df['Is_Anomaly']
        
        return result_df

    def cluster_stocks(self, df):
        """Cluster stocks based on technical indicators"""
        if 'kmeans' not in self.models or self.scaler is None:
            # Train new models if not loaded
            clean_df = df.dropna(subset=FEATURE_COLUMNS).copy()
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(clean_df[FEATURE_COLUMNS])
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            self.models['kmeans'] = kmeans
            self.scaler = scaler
            
            joblib.dump(kmeans, models_dir / "kmeans_model.pkl")
            joblib.dump(scaler, models_dir / "scaler.pkl")
        
        clean_df = df.dropna(subset=FEATURE_COLUMNS).copy()
        if len(clean_df) == 0:
            return df
            
        X_scaled = self.scaler.transform(clean_df[FEATURE_COLUMNS])
        clusters = self.models['kmeans'].predict(X_scaled)
        clean_df['Cluster'] = clusters
        
        # Merge back with original dataframe
        result_df = df.copy()
        result_df['Cluster'] = 0  # Default cluster
        result_df.loc[clean_df.index, 'Cluster'] = clean_df['Cluster']
        
        return result_df

    def find_similar_tickers(self, query_features_dict, df, top_n=3):
        """Find similar tickers based on technical features"""
        if 'kmeans' not in self.models or self.scaler is None:
            return []
        
        try:
            query_df = pd.DataFrame([query_features_dict])
            query_scaled = self.scaler.transform(query_df[FEATURE_COLUMNS])
            
            query_cluster = self.models['kmeans'].predict(query_scaled)[0]
            
            # Get latest state for each ticker
            latest_states = df.sort_values('Date').groupby('Ticker').last().reset_index()
            cluster_peers = latest_states[latest_states['Cluster'] == query_cluster].copy()
            
            if cluster_peers.empty:
                return []
                
            peer_features_scaled = self.scaler.transform(cluster_peers[FEATURE_COLUMNS])
            distances = cdist(query_scaled, peer_features_scaled, metric='euclidean')[0]
            
            cluster_peers['Distance'] = distances
            similar_df = cluster_peers.nsmallest(top_n, 'Distance')
            
            # Return list of tuples (ticker, distance)
            similar_tickers = [(row['Ticker'], row['Distance']) for _, row in similar_df.iterrows()]
            
            return similar_tickers
            
        except Exception as e:
            print(f"Error finding similar tickers: {e}")
            return []

    def predict_trend(self, df, ticker=None):
        if 'trend' not in self.models:
            self.train_trend_model(df)
        
        try:
            trend_features = FEATURE_COLUMNS + ['Is_Anomaly', 'Cluster']
            clean_df = df.dropna(subset=trend_features).copy()
            
            if len(clean_df) == 0:
                return None, None
            
            X = clean_df[trend_features].iloc[-1:].values
            
            prediction = self.models['trend'].predict(X)[0]
            probability = self.models['trend'].predict_proba(X)[0]
            
            return prediction, probability
            
        except Exception as e:
            print(f"Error in trend prediction: {e}")
            return None, None

    def train_trend_model(self, df):
        try:
            df_copy = df.copy()
            df_copy['Target'] = (df_copy.groupby('Ticker')['Close'].shift(-1) > df_copy['Close']).astype(int)
            
            trend_features = FEATURE_COLUMNS + ['Is_Anomaly', 'Cluster']
            clean_df = df_copy.dropna(subset=['Target'] + trend_features).copy()
            
            if len(clean_df) < 100:
                print("Insufficient data for training trend model")
                return
            
            split_idx = int(len(clean_df) * 0.8)
            train_df = clean_df.iloc[:split_idx]
            test_df = clean_df.iloc[split_idx:]
            
            X_train, y_train = train_df[trend_features], train_df['Target']
            X_test, y_test = test_df[trend_features], test_df['Target']
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                eval_metric='logloss'
            )
            
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            y_pred = xgb_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Trend model accuracy: {accuracy:.2%}")
            
            self.models['trend'] = xgb_model
            joblib.dump(xgb_model, models_dir / "trend_model.pkl")
            
        except Exception as e:
            print(f"Error training trend model: {e}")

    def analyze_stock(self, ticker, start_date=None, end_date=None):
        print(f"Analyzing {ticker}...")
        
        stock_data = self.fetch_and_process_stock(ticker, start_date, end_date)
        if stock_data is None:
            return None
        
        stock_data = self.detect_anomalies(stock_data, ticker)
        
        stock_data = self.cluster_stocks(stock_data)
        
        prediction, probability = self.predict_trend(stock_data, ticker)
        
        similar_stocks = []
        if len(stock_data) > 0:
            latest_features = stock_data[FEATURE_COLUMNS].iloc[-1].to_dict()
            
            try:
                historical_df = pd.read_csv(data_dir / "market_data_clusters.csv")
                if len(historical_df) > 0:
                    similar_stocks = self.find_similar_tickers(latest_features, historical_df)
            except:
                similar_stocks = self.find_similar_tickers(latest_features, stock_data)
        
        results = {
            'ticker': ticker,
            'data': stock_data,
            'prediction': prediction,
            'probability': probability,
            'similar_stocks': similar_stocks,
            'anomaly_count': stock_data['Is_Anomaly'].sum() if 'Is_Anomaly' in stock_data.columns else 0,
            'latest_price': stock_data['Close'].iloc[-1] if len(stock_data) > 0 else None,
            'latest_date': stock_data['Date'].iloc[-1] if len(stock_data) > 0 else None
        }
        
        return results

# Global analyzer instance
analyzer = StockAnalyzer()

# Convenience functions for the UI
def get_stock_analysis(ticker, start_date=None, end_date=None):
    """Get complete stock analysis"""
    return analyzer.analyze_stock(ticker, start_date, end_date)

def get_available_tickers():
    """Get list of available tickers"""
    return TOP_50_TICKERS

def load_historical_data():
    """Load historical data if available"""
    try:
        features_df = pd.read_csv(data_dir / "market_data_features.csv")
        return features_df
    except:
        return None

def test_ticker_fetch(ticker="AAPL"):
    """Test function to verify data fetching works"""
    try:
        print(f"Testing data fetch for {ticker}...")
        data = yf.download(ticker, period="5d", progress=False)
        print(f"Success! Retrieved {len(data)} rows")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False