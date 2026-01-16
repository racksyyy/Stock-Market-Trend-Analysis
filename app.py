"""
Stock Market Analysis Streamlit UI
Uses backend.py for all analysis functions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from backend import get_stock_analysis, get_available_tickers, load_historical_data

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for disclaimer
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

def show_disclaimer():
    """Display the mandatory disclaimer page using native Streamlit components"""
    
    # Add spacing at top
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Center the content
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Warning container
        st.error("## ⚠️ CRITICAL DISCLAIMER ⚠️")
        
        st.warning("### THIS IS NOT FINANCIAL ADVICE")
        
        st.markdown("""
        The predictions and insights provided by this application are based on historical data 
        and machine learning models with an accuracy of approximately **53.90%**.
        """)
        
        st.warning("### PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS")
        
        st.markdown("""
        This tool is for **educational and informational purposes only**.
        
        Market conditions can change rapidly and unpredictably. Machine learning models 
        may not account for unprecedented events, regulatory changes, or market sentiment shifts.
        
        **Always consult with a qualified financial advisor** before making any investment decisions.
        """)
        
        st.warning("""
        By proceeding, you acknowledge that you understand these limitations and will not 
        hold the creators liable for any financial losses.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("✅ I Understand & Accept", type="primary", use_container_width=True, key="disclaimer_accept"):
            st.session_state.disclaimer_accepted = True
            st.rerun()

def create_price_chart(data):
    """Create interactive price chart with anomalies and Bollinger Bands"""
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Bollinger Bands
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
            hovertemplate='<b>BB Upper:</b> $%{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            hovertemplate='<b>BB Lower:</b> $%{y:.2f}<extra></extra>'
        ))
    
    # Anomaly markers
    if 'Is_Anomaly' in data.columns:
        anomalies = data[data['Is_Anomaly'] == 1]
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['Date'],
                y=anomalies['Close'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='circle',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate='<b>Anomaly Detected</b><br><b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Stock Price Analysis with Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def display_metrics(results):
    """Display key metrics in a 4-column layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🎯 Trend Prediction")
        if results['prediction'] is not None:
            trend_label = "📈 UP" if results['prediction'] == 1 else "📉 DOWN"
            trend_color = "green" if results['prediction'] == 1 else "red"
            st.markdown(f"<h1 style='color: {trend_color}; text-align: center;'>{trend_label}</h1>", 
                       unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center; color: gray;'>No Prediction</h3>", 
                       unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Model Accuracy")
        st.markdown("<h1 style='text-align: center;'>53.90%</h1>", unsafe_allow_html=True)
        st.caption("Historical Training Accuracy")
    
    with col3:
        st.markdown("### 🤖 AI Confidence")
        if results['probability'] is not None:
            confidence = max(results['probability']) * 100
            st.markdown(f"<h1 style='text-align: center;'>{confidence:.1f}%</h1>", 
                       unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: gray;'>N/A</h1>", 
                       unsafe_allow_html=True)
        st.caption("Prediction Confidence")
    
    with col4:
        st.markdown("### 🚨 Anomaly Status")
        anomaly_count = results['anomaly_count']
        if anomaly_count > 0:
            status = f"🔴 {anomaly_count} Detected"
            color = "red"
        else:
            status = "🟢 Normal"
            color = "green"
        st.markdown(f"<h2 style='color: {color}; text-align: center;'>{status}</h2>", 
                   unsafe_allow_html=True)
        st.caption("Unusual Price Movements")

def display_similar_stocks(similar_stocks, ticker, results=None):
    """Display similar stocks section with distance metrics"""
    st.markdown("---")
    st.markdown("### 🔗 Stocks Exhibiting Similar Technical Behavior")
    
    if similar_stocks and len(similar_stocks) > 0:
        # Filter out the current ticker from similar stocks
        filtered_similar = [(t, d) for t, d in similar_stocks if t != ticker]
        
        if filtered_similar:
            # Limit to top 3
            top_3_similar = filtered_similar[:3]
            
            st.info(f"Based on recent technical patterns, **{ticker}** exhibits similar behavior to these stocks:")
            
            # Display similar stocks with metrics
            cols = st.columns(3)
            
            for idx, (similar_ticker, distance) in enumerate(top_3_similar):
                with cols[idx]:
                    st.markdown(f"### {similar_ticker}")
                    
                    # Convert distance to similarity percentage
                    # Lower distance = higher similarity
                    # Normalize distance to 0-100% scale (assuming max distance ~5)
                    max_distance = 5.0
                    similarity = max(0, min(100, (1 - (distance / max_distance)) * 100))
                    
                    st.metric(
                        "Pattern Similarity",
                        f"{similarity:.1f}%",
                        help="How closely this stock's technical patterns match"
                    )
                    
                    st.caption(f"Rank #{idx + 1} • Distance: {distance:.3f}")
            
            st.markdown("---")
            st.caption("💡 **Note:** Similarity is based on technical indicators including RSI, MACD, Bollinger Bands, and volume patterns.")
        else:
            st.warning("No other similar stocks found in the current analysis.")
    else:
        st.warning("No similar stocks identified. This could indicate unique market behavior.")

def main_app():
    """Main application interface"""
    st.title("📈 AI-Powered Stock Market Trend Analysis")
    st.markdown("*Advanced machine learning insights for informed decision making*")
    
    # Sidebar controls
    st.sidebar.title("📊 Analysis Controls")
    st.sidebar.markdown("---")
    
    # Get available tickers
    available_tickers = get_available_tickers()
    top_5_tickers = ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN"]
    
    # Ticker selection - simplified without redundant "Choose from Top 5"
    ticker_option = st.sidebar.selectbox(
        "🎯 Select Stock",
        top_5_tickers + ["Other (Custom Input)"],
        help="Select a stock for analysis"
    )
    
    if ticker_option == "Other (Custom Input)":
        ticker = st.sidebar.text_input(
            "Enter Ticker Symbol", 
            value="AAPL",
            help="Enter any valid stock ticker symbol"
        ).upper().strip()
    else:
        ticker = ticker_option
    
    # Date range selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("📅 **Analysis Period**")
    
    max_date = datetime.now().date()
    min_date = max_date - timedelta(days=180)  # 6 months
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            help="Analysis start date (max 6 months ago)"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=start_date,
            max_value=max_date,
            help="Analysis end date"
        )
    
    # Analysis button
    st.sidebar.markdown("---")
    run_analysis = st.sidebar.button(
        "🚀 Run Analysis", 
        type="primary", 
        use_container_width=True,
        help="Click to start the AI analysis"
    )
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🔧 Debug Mode", help="Show detailed error information")
    
    # Main content area
    if not ticker:
        st.info("👆 Please select a stock ticker from the sidebar to begin analysis.")
        return
    
    # Display current selection
    st.markdown(f"## Analysis for: **{ticker}**")
    
    if run_analysis:
        with st.spinner(f"🔍 Analyzing {ticker}... This may take a moment."):
            try:
                # Get analysis results
                results = get_stock_analysis(ticker, start_date, end_date)
                
                if results is None:
                    st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol and try again.")
                    st.info("💡 **Troubleshooting Tips:**")
                    st.markdown("""
                    - Verify the ticker symbol is correct (e.g., AAPL, MSFT, GOOGL)
                    - Try a different date range
                    - Some international tickers may not be available
                    - Check your internet connection
                    """)
                    return
                
                if results['data'] is None or len(results['data']) == 0:
                    st.error(f"❌ No data available for {ticker} in the selected date range.")
                    st.info("💡 **Try:**")
                    st.markdown("""
                    - Selecting a wider date range
                    - Using a different ticker symbol
                    - Checking if the stock was trading during the selected period
                    """)
                    return
                
                # Display results
                st.success(f"✅ Analysis complete for {ticker}")
                
                # Metrics row
                display_metrics(results)
                
                # Price chart
                st.markdown("---")
                st.markdown("### 📊 Interactive Price Chart")
                chart = create_price_chart(results['data'])
                st.plotly_chart(chart, use_container_width=True)
                
                # Technical indicators summary
                if len(results['data']) > 0:
                    st.markdown("---")
                    st.markdown("### 📈 Latest Technical Indicators")
                    
                    latest_data = results['data'].iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "RSI (14)",
                            f"{latest_data.get('RSI_14', 0):.1f}",
                            help="Relative Strength Index - measures overbought/oversold conditions"
                        )
                    
                    with col2:
                        st.metric(
                            "SMA Ratio (5/20)",
                            f"{latest_data.get('SMA_Ratio_5_20', 0):.3f}",
                            help="Short-term vs Long-term moving average ratio"
                        )
                    
                    with col3:
                        st.metric(
                            "MACD Histogram",
                            f"{latest_data.get('MACD_Histogram', 0):.3f}",
                            help="MACD momentum indicator"
                        )
                    
                    with col4:
                        st.metric(
                            "BB Position",
                            f"{latest_data.get('BB_Position', 0):.3f}",
                            help="Position within Bollinger Bands (0-1)"
                        )
                
                # Similar stocks
                display_similar_stocks(results['similar_stocks'], ticker)
                
                # Additional information
                st.markdown("---")
                st.markdown("### ℹ️ Analysis Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **Analysis Period:** {start_date} to {end_date}  
                    **Data Points:** {len(results['data'])} trading days  
                    **Latest Price:** ${results['latest_price']:.2f}  
                    **Analysis Date:** {results['latest_date']}
                    """)
                
                with col2:
                    if results['prediction'] is not None:
                        direction = "upward" if results['prediction'] == 1 else "downward"
                        confidence = max(results['probability']) * 100 if results['probability'] is not None else 0
                        
                        st.warning(f"""
                        **AI Prediction:** {direction.title()} trend  
                        **Confidence Level:** {confidence:.1f}%  
                        **Anomalies Detected:** {results['anomaly_count']}  
                        **Model Accuracy:** 53.90% (Historical)
                        """)
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                if debug_mode:
                    st.code(f"Debug info: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                st.info("Please try again or contact support if the issue persists.")
    
    else:
        # Show placeholder content
        st.info("👆 Click 'Run Analysis' in the sidebar to start the analysis.")
        
        # Show some general information
        st.markdown("---")
        st.markdown("### 🎯 What This Analysis Provides")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🤖 AI-Powered Predictions**
            - XGBoost machine learning model
            - Binary trend prediction (UP/DOWN)
            - Confidence scoring
            
            **📊 Technical Analysis**
            - RSI, MACD, Bollinger Bands
            - Moving average analysis
            - Volume indicators
            """)
        
        with col2:
            st.markdown("""
            **🚨 Anomaly Detection**
            - Isolation Forest algorithm
            - Unusual price movement identification
            - Risk assessment indicators
            
            **🔗 Peer Analysis**
            - K-Means clustering
            - Similar stock identification
            - Comparative behavior analysis
            """)

# Main execution
if __name__ == "__main__":
    if not st.session_state.disclaimer_accepted:
        show_disclaimer()
    else:
        main_app()