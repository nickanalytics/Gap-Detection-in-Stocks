import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import requests
from io import StringIO
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


### ***** Set Variables *****

# Streamlit page configuration
st.set_page_config(page_title="Stock Dashboard", page_icon=":chart_with_upwards_trend:",layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# Header across all columns
# st.title('Financial Ticker Dashboard')
col0_1, col0_2, col0_3 = st.columns([1, 6, 1])

with col0_2:
    st.markdown("""
                <h1 style='text-align: center;'>&#x1F4C8; Gap Detection in Stocks</h1>
                <h3 style="text-align: center; color: gray;">--- by Nick Analytics ---</h3>""", unsafe_allow_html=True)

# Draw a line
st.markdown("------------------------------------------------")

# Create three rows of three columns each
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
row4_col1 = st.columns(1)[0]  # Correct way to assign a single column

# ***** Set Variables in the first row, first column *****
with row1_col1:
    st.subheader('Select Ticker and Date Range')
    # symbol = st.selectbox('Ticker', ['AAPL', 'ABBV', 'ABN.AS', 'ADBE', 'ADYEN.AS', '^AEX', 'AKZA', 'ALFEN.AS', 'AMD', 
    #'AMGN', 'AMZN', 'ASM.AS', 'ASML.AS', 'BAC', 'BESI.AS', 'BKNG', 'BRK.B', 'CAT', 'CITI', 
    #'CRM', 'CSCO', 'DIS', 'DJI', 'DSFIR', 'EURUSD=X', 'GC=F', 'GE', 'GILD', 'GLPG.AS', 
    #'GOOG','^iXIC','JNJ', 'LIGHT.AS', 'MA', 'MCD', 'META', 'MRNA', 'MSFT', 'MU', 'NFLX', 
    #'NKE', 'NN.AS', 'NVDA', 'PFE', 'PG', 'PINS', 'PLTR', 'RAND.AS', 'SBMO.AS', 
    #'SBUX', 'SHELL.AS', 'SHOP', 'TSLA', 'URW.PA', 'V', 'WKL.AS', 'WMT', 'ZM'])
    # Getting today's date

    ticker_to_company = {
    'AAPL': 'Apple Inc.',
    'ABBV': 'AbbVie Inc.',
    'ABN.AS': 'ABN AMRO Bank N.V.',
    'ADBE': 'Adobe Inc.',
    'ADYEN.AS': 'Adyen N.V.',
    '^AEX': 'AEX Index',
    'AKZA': 'Akzo Nobel N.V.',
    'ALFEN.AS': 'Alfen N.V.',
    'AMD': 'Advanced Micro Devices, Inc.',
    'AMGN': 'Amgen Inc.',
    'AMZN': 'Amazon.com Inc.',
    'ASM.AS': 'ASM International N.V.',
    'ASML.AS': 'ASML Holding N.V.',
    'BAC': 'Bank of America Corp.',
    'BESI.AS': 'BE Semiconductor Industries N.V.',
    'BKNG': 'Booking Holdings Inc.',
    'BRK.B': 'Berkshire Hathaway Inc. (Class B)',
    'CAT': 'Caterpillar Inc.',
    'CITI': 'Citigroup Inc.',
    'CRM': 'Salesforce.com Inc.',
    'CSCO': 'Cisco Systems, Inc.',
    'DIS': 'The Walt Disney Company',
    'DJI': 'Dow Jones Industrial Average',
    'DSFIR.AS': 'DSM-Firmenich',
    'EURUSD=X': 'Euro to US Dollar Exchange Rate',
    'GC=F': 'Gold Futures',
    'GE': 'General Electric Company',
    'GILD': 'Gilead Sciences, Inc.',
    'GLPG.AS': 'Galapagos NV',
    'GOOG': 'Alphabet Inc. (Class C)',
    '^iXIC': 'NASDAQ Composite Index',
    'JNJ': 'Johnson & Johnson',
    'LIGHT.AS': 'Signify N.V.',
    'MA': 'Mastercard Incorporated',
    'MCD': "McDonald's Corp.",
    'META': 'Meta Platforms Inc.',
    'MRNA': 'Moderna, Inc.',
    'MSFT': 'Microsoft Corporation',
    'MU': 'Micron Technology, Inc.',
    'NFLX': 'Netflix Inc.',
    'NKE': 'NIKE, Inc.',
    'NN.AS': 'NN Group N.V.',
    'NVDA': 'NVIDIA Corporation',
    'PFE': 'Pfizer Inc.',
    'PG': 'Procter & Gamble Co.',
    'PINS': 'Pinterest, Inc.',
    'PLTR': 'Palantir Technologies Inc.',
    'RAND.AS': 'Randstad N.V.',
    'SBMO.AS': 'SBM Offshore N.V.',
    'SBUX': 'Starbucks Corporation',
    'SHELL.AS': 'Royal Dutch Shell Plc',
    'SHOP': 'Shopify Inc.',
    'TSLA': 'Tesla, Inc.',
    'URW.PA': 'Unibail-Rodamco-Westfield SE',
    'V': 'Visa Inc.',
    'WKL.AS': 'Wolters Kluwer N.V.',
    'WMT': 'Walmart Inc.',
    'ZM': 'Zoom Video Communications, Inc.'
}

    # Streamlit widget to select a ticker
    symbol = st.selectbox('Ticker', list(ticker_to_company.keys()))

    # Retrieving the company name based on the selected ticker
    comp_name = ticker_to_company[symbol]

    # Displaying the company name
    st.write(f'The company name for the ticker {symbol} is {comp_name}.')
    
    today = datetime.today()

    # Slider to choose how many years ago, allowing for fractional years
    years_ago = st.slider("Years Ago", min_value=0.5, max_value=8.0, step=0.5)
    total_months = int(years_ago * 12)  # Convert years to a total number of months

    # Calculate the start date by subtracting months
    start = today - relativedelta(months=total_months)
    st.write(f'Start Date {start}')

    # Date input for the end date, defaulting to today
    end = st.date_input('End Date', value=today) 

    # Button to check data
    if st.button('Check'):
        # ***** Get the Data *****
        tickerData = yf.Ticker(symbol)
        df = tickerData.history(period='1d', start=start, end=end, auto_adjust = False)
        df = df.reset_index()
        # df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date

        df.columns = df.columns.str.lower()
        df = df.sort_values(by='date', ascending=True)
        # Round the 'open', 'high', 'low', 'close' columns to 3 decimal places
        df['open'] = df['open'].round(2)
        df['high'] = df['high'].round(2)
        df['low'] = df['low'].round(2)
        df['close'] = df['close'].round(2)

        # **** Create a list with all gaps (>1 day) *****
        all_gaps = pd.DataFrame(columns=['date', 'gap_size', 'from', 'to', 'gap_type'])
        
        for i in range(1, len(df)):
            prev_high = df.iloc[i-1]['high']
            prev_low = df.iloc[i-1]['low']
            curr_open = df.iloc[i]['open']
            curr_high = df.iloc[i]['high']
            curr_low = df.iloc[i]['low']
            gap_size = 0
            
            if curr_low > prev_high:
                gap_size = curr_low - prev_high
                new_row = {'date': df.iloc[i]['date'], 'gap_size': gap_size, 'from': prev_high, 'to': curr_low, 'gap_type': 'up'}
                all_gaps = pd.concat([all_gaps, pd.DataFrame([new_row])], ignore_index=True)
            elif curr_high < prev_low:
                gap_size = prev_low - curr_high
                new_row = {'date': df.iloc[i]['date'], 'gap_size': gap_size, 'from': prev_low, 'to': curr_high, 'gap_type': 'down'}
                all_gaps = pd.concat([all_gaps, pd.DataFrame([new_row])], ignore_index=True)

               
        # Display Gaps DataFrame in the third top column
        with row2_col2:                
            st.subheader(f'Detected Gaps > 1 day ({len(all_gaps)})')  # dynamically include the count of gaps
            st.dataframe(all_gaps, hide_index='True')  # display without the index column


        with row2_col3:
            # Initialize a list to store the data
            same_day_closes = []

            # Loop through the rows of the dataframe
            for i in range(1, len(df)):
                prev_high = df.iloc[i-1]['high']
                prev_low = df.iloc[i-1]['low']
                curr_open = df.iloc[i]['open']
                curr_high = df.iloc[i]['high']
                curr_low = df.iloc[i]['low']
                curr_close = df.iloc[i]['close']
                
                # Check for gap up and same day close within the previous day's high
                # if curr_open > prev_high and curr_low <= prev_high:
                if curr_open > prev_high and curr_low < prev_high:
                    gap_size = curr_open - prev_high
                    same_day_closes.append({'date': df.iloc[i]['date'], 'gap_size': gap_size})
                    
                # Check for gap down and same day close within the previous day's low
                # elif curr_open < prev_low and curr_high >= prev_low:
                elif curr_open < prev_low and curr_high > prev_low:
                    gap_size = prev_low - curr_open
                    same_day_closes.append({'date': df.iloc[i]['date'], 'gap_size': gap_size})

            # Create DataFrame from the list of data
            same_day_closes = pd.DataFrame(same_day_closes)

            # Print the resulting DataFrame

            st.subheader(f'Same day Gap closes ({len(same_day_closes)})')  # dynamically include the count of same day gap closes
            st.dataframe(same_day_closes, hide_index='True')  # display without the index column

        with row2_col1:
            # Initialize the DataFrames
            closed_gaps = pd.DataFrame(columns=['closed_date', 'gap_start', 'gap_size', 'duration'])
            open_gaps = all_gaps.copy()

            # Initialize a list to collect new rows for closed_gaps
            new_rows = []

            # Loop through open_gaps DataFrame
            for index, gap in open_gaps.iterrows():
                # Filter df based on gap start date and gap type
                if gap['gap_type'] == 'up':
                    filtered_data = df[(df['date'] >= gap['date']) & (df['low'] <= gap['from'])]
                elif gap['gap_type'] == 'down':
                    filtered_data = df[(df['date'] >= gap['date']) & (df['high'] >= gap['from'])]
                
                # Check if gap is closed
                if not filtered_data.empty:
                    # Calculate duration
                    duration = (filtered_data.iloc[0]['date'] - gap['date']).days
                    # Collect the new row
                    new_rows.append({
                        'closed_date': filtered_data.iloc[0]['date'],
                        'gap_start': gap['from'],
                        'gap_size': gap['gap_size'],
                        'duration': duration
                    })
                    # Drop row from open_gaps DataFrame
                    open_gaps.drop(index, inplace=True)

            # Concatenate all new rows to closed_gaps DataFrame in one operation
            if new_rows:
                closed_gaps = pd.concat([closed_gaps, pd.DataFrame(new_rows)], ignore_index=True)

            # Reset the index of open_gaps
            open_gaps.reset_index(drop=True, inplace=True)
            closed_gaps['duration'] = closed_gaps['duration'].astype(float)

            st.subheader(f'All Open Gaps ({len(open_gaps)})')  # dynamically include the count of same day gap closes
            st.dataframe(open_gaps, hide_index='True')
        with row2_col4:
            st.subheader(f'All Closed Gaps ({len(closed_gaps)})')  # dynamically include the count of same day gap closes
            st.dataframe(closed_gaps, hide_index='True')

        with row1_col2:            
            total = len(same_day_closes) + len(all_gaps)
            latest_price = df['close'].iloc[-1]
            idx = (open_gaps['from'] - latest_price).abs().idxmin() if not open_gaps.empty else None
            nearest_gap = open_gaps['from'].iloc[idx] if idx is not None else "N/A"
            nearest_gap_date = open_gaps['date'].iloc[idx].strftime('%Y-%m-%d') if idx is not None else "N/A"

            # Using Streamlit columns to layout metrics
            # st.subheader("Ticker Analysis: " + symbol)
            st.subheader(f"Ticker Analysis: {symbol} - Latest: {latest_price:.2f}")
            st.divider()
            metric_col1, metric_col2 = st.columns(2)
            # metric_col1.metric("Latest Stock Price", f"{latest_price:.2f}")
            metric_col1.metric("Nearest Gap", f"{nearest_gap}")
            metric_col2.metric("Nearest Gap Date", f"{nearest_gap_date}")
                        
            st.divider()
            metric_col4, metric_col5, metric_col6, metric_col7 = st.columns(4)
            metric_col4.metric("True Gaps", f"{len(all_gaps)}")
            metric_col5.metric("Closed Gaps", len(closed_gaps))
            metric_col6.metric("Open Gaps", len(open_gaps))
            # metric_col6.metric("Gaps Closed in Timeframe", f"{1 - (round(len(open_gaps) / total, 2) if total else 0):.0%}")
            metric_col7.metric("Mean Gap Size (Same Day)", f"{same_day_closes.gap_size.mean():.2f}" if not same_day_closes.empty else "N/A")
            
            st.divider()
            #metric_col8,metric_col9, metric_col10 = st.columns(3)
            #metric_col8.metric("Largest Gap Close (Same Day)", f"{same_day_closes.gap_size.max():.2f}" if not same_day_closes.empty else "N/A")
            #metric_col9.metric("Mean Gap Size (True Gaps)", f"{all_gaps.gap_size.mean():.2f}" if not all_gaps.empty else "N/A")
            
            # metric_col11, metric_col12 = st.columns(2)
            # metric_col11.metric("Max Gap Size (True Gaps)", f"{all_gaps.gap_size.max():.2f}" if not all_gaps.empty else "N/A")
            # metric_col12.metric("Total Gaps", total)
        with row1_col3:
            st.subheader("Ticker Chart: " + symbol)
            st.line_chart(data=df, x='date', y='close', color=None, width=0, height=0, use_container_width=True)

        with row3_col3:
            metric_col8 = st.columns(1)[0]
            metric_col8.metric("Same Day Closes", f"{len(same_day_closes)} ({len(same_day_closes)/total:.2%})")
        
        with row3_col4:
            # Calculate the correlation
            metric_col9 = st.columns(1)[0]
            correlation = closed_gaps['gap_size'].corr(closed_gaps['duration'])
            correlation_display = f"{correlation:.4f}" if isinstance(correlation, (float, int)) else "N/A"
            metric_col9.metric("Correlation between gap_size and duration:", correlation_display)

        with row4_col1:
            if not closed_gaps.empty:
                st.subheader(f'Days to close a Gap, given the Gap Size ({len(all_gaps)})')  # dynamically include the count of gaps
                st.scatter_chart(data=closed_gaps, x='duration', y='gap_size', use_container_width=True)
            else:
                st.write("No data available to display.")