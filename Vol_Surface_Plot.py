#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import sqlite3
import yfinance as yf
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit
from flask import request

# Retrieve the Dow Jones Industrial Average company tickers
def get_djia():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    djia_df = pd.read_html(url)[1]
    return djia_df['Symbol'].tolist()

# Generate options for Dash dropdown
djia_tickers = get_djia()
dropdown_options = [{'label': ticker, 'value': ticker} for ticker in djia_tickers]

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Volatility Surface Plotter"),
    html.Div([
        html.Div([
            html.Label("Ticker"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=dropdown_options,
                placeholder="Select a ticker",
                style={'width': '200px', 'display': 'inline-block'}
            ),
            dcc.Input(
                id='ticker-input',
                type='text',
                placeholder="Or enter a ticker",
                style={'display': 'inline-block', 'margin-left': '10px'}
            ),
            html.Button('Reset', id='reset-button', style={'display': 'inline-block', 'margin-left': '10px'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Length to Expiration"),
            dcc.RadioItems(
                id='exp-length',
                options=[
                    {'label': '1/2-year', 'value': 'half'},
                    {'label': '1-year', 'value': '1'}
                ],
                value='half',
                labelStyle={'display': 'inline-block'}
            ),
        ], style={'display': 'inline-block', 'margin-left': '20px'}),
        html.Div([
            html.Button('Go', id='go-button', style={'margin-right': '10px'}),
            html.Button('Download CSV', id='download-button', disabled=True)
        ], style={'display': 'inline-block', 'margin-left': '20px'}),
        html.Div(id='error-message', style={'color': 'red', 'margin-bottom': '20px'}),
            html.Div([
                html.Button('Go', id='go-button', style={'margin-right': '10px'}),
                html.Button('Download CSV', id='download-button', disabled=True)
            ], style={'display': 'inline-block', 'margin-left': '20px'}),
            html.Div(id='error-message', style={'color': 'red', 'margin-bottom': '20px'}),
    html.Div([
        "What to Know and How to Use? ",
        html.A("Click here.", href="https://github.com/BenjaminZYT/Volatility-Surface/blob/main/README.md", target="_blank")
    ], style={'margin-bottom': '20px', 'font-weight': 'bold'}),
    dcc.Download(id="download-dataframe-csv"),
    dcc.Graph(id='volatility-surface-call'),
    dcc.Graph(id='volatility-surface-put'),
])

@app.callback(
    Output('error-message', 'children'),
    Output('ticker-dropdown', 'value'),
    Output('ticker-input', 'value'),
    Input('go-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('ticker-dropdown', 'value'),
    State('ticker-input', 'value'),
    State('exp-length', 'value'),
    prevent_initial_call=True
)
def validate_and_reset(go_clicks, reset_clicks, dropdown_value, input_value, exp_choice):
    triggered_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if triggered_id == 'reset-button':
        return '', None, ''

    ticker = dropdown_value if dropdown_value else input_value
    if dropdown_value and input_value and dropdown_value != input_value:
        return 'Invalid Input. Please try again.', None, ''
    
    if not ticker or not ticker_exists(ticker):
        return 'Invalid Input. Please try again.', None, ''

    return '', dropdown_value, input_value

def ticker_exists(ticker):
    try:
        stock = yf.Ticker(ticker)
        options = stock.options
        return len(options) > 0
    except:
        return False

def record_user_query(ticker, exp_choice):
    query_data = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "expiration_choice": 0.5 if exp_choice == 'half' else 1
    }
    with sqlite3.connect('OptionsProj.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_queries (
                datetime TEXT,
                ticker TEXT,
                expiration_choice REAL
            )
        ''')
        cursor.execute('''
            INSERT INTO user_queries (datetime, ticker, expiration_choice)
            VALUES (:datetime, :ticker, :expiration_choice)
        ''', query_data)
        conn.commit()

@app.callback(
    [Output('volatility-surface-call', 'figure'),
     Output('volatility-surface-put', 'figure'),
     Output('download-button', 'disabled')],
    [Input('go-button', 'n_clicks')],
    [State('ticker-dropdown', 'value'),
     State('ticker-input', 'value'),
     State('exp-length', 'value')]
)
def update_plots(n_clicks, dropdown_value, input_value, exp_choice):
    if n_clicks is None:
        return {}, {}, True
    
    ticker = dropdown_value if dropdown_value else input_value
    if not ticker:
        return {}, {}, True
    
    # Retrieve data
    stock = yf.Ticker(ticker)
    now = datetime.now().strftime("%m/%d/%Y %I:%M %p")
    spot = stock.history(period='1d')['Close'].iloc[-1]
    rate_ticker = '^IRX'
    rate_data = yf.Ticker(rate_ticker).history(period='1d')
    rate = rate_data['Close'].iloc[-1] / 100

    # Retrieve options data
    options_dates = stock.options
    today = datetime.now()
    if exp_choice == 'half':
        expiration_dates = [d for d in options_dates if (datetime.strptime(d, '%Y-%m-%d') - today).days <= 183]
    else:
        expiration_dates = [d for d in options_dates if (datetime.strptime(d, '%Y-%m-%d') - today).days <= 365]

    rows = []
    for exp_date in expiration_dates:
        options_chain = stock.option_chain(exp_date)
        for option_type, option_data in zip(["calls", "puts"], [options_chain.calls, options_chain.puts]):
            for index, row in option_data.iterrows():
                rows.append({
                    "datetime": datetime.now(),
                    "exp_date": exp_date,
                    "type": option_type[0],
                    "strike": row['strike'],
                    "price": row['lastPrice'],
                    "vol": row['impliedVolatility'],
                    "inTheMoney": row['inTheMoney'],
                    "ticker": ticker
                })

    df_raw = pd.DataFrame(rows, columns=["datetime", "exp_date", "type", "strike", "price", "vol", "inTheMoney"])

    record_user_query(ticker, exp_choice)

     # Store df_raw in an auxiliary table
    with sqlite3.connect('OptionsProj.db') as conn:
        df_raw.to_sql('data_aux', conn, if_exists='replace', index=False)

    # Fit volatility surfaces
    def fit_volatility_surface(df_raw, option_type):
        df_raw_aux = df_raw[(df_raw['type'] == option_type) & (df_raw['vol'].abs() >= 0.01)].copy()
        df_raw_aux['diff'] = (pd.to_datetime(df_raw_aux['exp_date']) - pd.to_datetime(df_raw_aux['datetime'])).dt.days
        X = df_raw_aux['diff'].values
        Y = df_raw_aux['strike'].values
        Z = df_raw_aux['vol'].values
        def func(XY, a, b, c, d, e, f):
            X, Y = XY
            return a + b*X + c*Y + d*X*Y + e*X**2 + f*Y**2
        popt, _ = curve_fit(func, (X, Y), Z)
        return popt, df_raw_aux

    Surface_Call, df_raw_aux_call = fit_volatility_surface(df_raw, 'c')
    Surface_Put, df_raw_aux_put = fit_volatility_surface(df_raw, 'p')

    def plot_3d_scatter_with_surface(df, option_type, spot_price, surface_params):
        df_filtered = df[df['type'] == option_type].copy()
        df_filtered['In-The-Money'] = df_filtered['inTheMoney'].apply(lambda x: 'Yes' if x else 'No')
        def func(XY, a, b, c, d, e, f):
            X, Y = XY
            return a + b*X + c*Y + d*X*Y + e*X**2 + f*Y**2
        base_date = pd.to_datetime(df_filtered['datetime'].iloc[0])
        X = (pd.to_datetime(df_filtered['exp_date']) - pd.to_datetime(df_filtered['datetime'])).dt.days.values
        Y = df_filtered['strike'].values
        X_grid, Y_grid = np.meshgrid(np.linspace(X.min(), X.max(), 100), np.linspace(Y.min(), Y.max(), 100))
        Z_grid = func((X_grid.ravel(), Y_grid.ravel()), *surface_params).reshape(X_grid.shape)
        X_dates_grid = base_date + pd.to_timedelta(X_grid.ravel(), unit='D')
        X_dates_grid = X_dates_grid.values.reshape(X_grid.shape)
        min_vol = df_filtered['vol'].min()
        nonzero_min = df_filtered[df_filtered['vol'] > 0.01]['vol'].min()
        max_vol = df_filtered['vol'].max()
        min_row = df_filtered[df_filtered['vol'] == min_vol].iloc[0]
        max_row = df_filtered[df_filtered['vol'] == max_vol].iloc[0]
        min_caption = f"min volatility = {nonzero_min:.2f} (Strike: ${min_row['strike']:.2f}, Expiration Date: {min_row['exp_date']})"
        max_caption = f"max volatility = {max_vol:.2f} (Strike: ${max_row['strike']:.2f}, Expiration Date: {max_row['exp_date']})"
        fig = px.scatter_3d(
            df_filtered,
            x=pd.to_datetime(df_filtered['exp_date']),
            y=df_filtered['strike'],
            z=df_filtered['vol'],
            color='In-The-Money',
            title=f'Using {ticker} {("Calls" if option_type == "c" else "Puts")} (Spot Price: ${spot_price} @ {now} U.S. Eastern w/ Rate {round(rate, 2)})',
            labels={
                'x': 'Expiration Date',
                'y': 'Strike Price',
                'z': 'Implied Volatility'
            }
        )
        fig.add_trace(
            go.Surface(
                x=X_dates_grid,
                y=Y_grid,
                z=Z_grid,
                colorscale='Viridis',
                opacity=0.7,
                showscale=False
            )
        )
        fig.add_trace(
            go.Mesh3d(
                x=[df_filtered['exp_date'].min(), df_filtered['exp_date'].max(), df_filtered['exp_date'].max(), df_filtered['exp_date'].min()],
                y=[spot_price, spot_price, spot_price, spot_price],
                z=[df_filtered['vol'].min(), df_filtered['vol'].min(), df_filtered['vol'].max(), df_filtered['vol'].max()],
                opacity=0.5,
                color='lightblue',
                i=[0, 0],
                j=[1, 2],
                k=[2, 3]
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title='Expiration Date',
                yaxis_title='Strike Price',
                zaxis_title='Implied Volatility'
            ),
            coloraxis_colorbar=dict(
                title='In-The-Money',
                tickvals=['Yes', 'No'],
                ticktext=['Yes', 'No']
            ),
            width=1000,
            height=800
        )
        fig.update_layout(scene=dict(
            xaxis=dict(
                autorange='reversed'
            )
        ))
        fig.add_annotation(
            x=0.5,
            y=-0.1,
            text=f"{min_caption}<br>{max_caption}",
            showarrow=False,
            xref="paper",
            yref="paper",
            align="center",
            font=dict(size=12),
            bordercolor="black",
            borderwidth=1,
            bgcolor="white"
        )
        
        fig.add_annotation(
            text="By Benjamin Z.Y. Teoh @ 2024",
            xref="paper", yref="paper",
            x=0.5, y=1.05,  # Position it on the top center
            showarrow=False,
            font=dict(size=8, color="black")  # Adjust the font size and color
        )
        return fig

    fig_call = plot_3d_scatter_with_surface(df_raw, 'c', round(spot, 2), Surface_Call)
    fig_put = plot_3d_scatter_with_surface(df_raw, 'p', round(spot, 2), Surface_Put)

    return fig_call, fig_put, False

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-button", "n_clicks")],
    [State('ticker-dropdown', 'value'),
     State('ticker-input', 'value'),
     State('exp-length', 'value')],
    prevent_initial_call=True,
)
def download_csv(n_clicks, dropdown_value, input_value, exp_choice):
    ticker = dropdown_value if dropdown_value else input_value
    if not ticker:
        return None

    # Record user query
    record_user_query(ticker, exp_choice)

    # Retrieve df_raw from auxiliary table
    with sqlite3.connect('OptionsProj.db') as conn:
        df_raw = pd.read_sql('SELECT * FROM data_aux', conn)

    # Destroy auxiliary table
    with sqlite3.connect('OptionsProj.db') as conn:
        conn.execute('DROP TABLE IF EXISTS data_aux')
    
    return dcc.send_data_frame(df_raw.to_csv, filename=f"{ticker}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{exp_choice}.csv")

if __name__ == "__main__":
    app.run_server(debug=True)
