import json
import requests

import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from explainerdashboard.explainer_plots import plotly_contribution_plot
from explainerdashboard.explainer_methods import get_contrib_df, get_contrib_summary_df

apis = {
    "sagemaker": "https://4m53ma6yf8.execute-api.eu-central-1.amazonaws.com/test/credit-explainer",
    "lambda": "https://ksz29r10ri.execute-api.eu-west-1.amazonaws.com/dev/predict",
    "local": "http://localhost:5001/predict"
}

df = pd.read_csv("data/data.csv").drop("Unnamed: 0", axis=1)

def construct_sample_row(monthly_income, age, no_dependents, revolving_utilization, 
                         debt_ratio, open_creditlines, real_estate_loans,
                        no_30_59_days_due, no_60_89_days_due, no_90_days_late):
    row_dict = dict(MonthlyIncome=monthly_income,
                     age=age,
                     NumberOfDependents=no_dependents,
                     RevolvingUtilizationOfUnsecuredLines=revolving_utilization,
                     DebtRatio=debt_ratio,
                     NumberOfOpenCreditLinesAndLoans=open_creditlines,
                     NumberRealEstateLoansOrLines=real_estate_loans,
                     NumberOfTimes90DaysLate=no_90_days_late)
    row_dict["NumberOfTime30-59DaysPastDueNotWorse"]=no_30_59_days_due
    row_dict['NumberOfTime60-89DaysPastDueNotWorse']=no_60_89_days_due
    return pd.DataFrame(row_dict, index=[0])

class CreditExplainerDashboard:
    def __init__(self, df, apis, default_api):
        self.df = df
        self.apis = apis
        self.default_api = default_api
    
    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                   html.H1("Credit Delinquency Predictor"), 
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Input Data"),
                ], width=3),
                dbc.Col([
                    html.Button("Sample a row!", id='sample-button'),
                ])  
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label('MonthlyIncome'),
                    dbc.Input(id='monthly-income', 
                              type="number", min=0, max=100000),
                    dbc.Label('age'),
                    dbc.Input(id='age', 
                              type="number", min=0, max=120),
                    dbc.Label('NumberOfDependents'),
                    dbc.Input(id='no-dependents', 
                              type="number", min=0, max=20),
                    dbc.Label('RevolvingUtilizationOfUnsecuredLines'),
                    dbc.Input(id='revolving-utilization', 
                              type="number", min=0.0, max=1.0, step=0.01),
                    dbc.Label('DebtRatio'),
                    dbc.Input(id='debt-ratio', 
                              type="number", min=0.0, max=1.0, step=0.01),
                ]),
                dbc.Col([
                    dbc.Label('NumberOfOpenCreditLinesAndLoans'),
                    dbc.Input(id='open-creditlines', 
                              type="number", min=0, max=60),
                    dbc.Label('NumberRealEstateLoansOrLines'),
                    dbc.Input(id='no-real-estate-loans', 
                              type="number", min=0, max=55),
                    dbc.Label('NumberOfTime30-59DaysPastDueNotWorse'),
                    dbc.Input(id='no-30-59-days-due', 
                              type="number", min=0, max=10),
                    dbc.Label('NumberOfTime60-89DaysPastDueNotWorse'),
                    dbc.Input(id='no-60-89-days-due', 
                              type="number", min=0, max=10),
                    dbc.Label('NumberOfTimes90DaysLate'),
                    dbc.Input(id='no-90-days-late', 
                              type="number", min=0, max=100),  
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Select API"),
                ]), 
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Select API"),
                    dcc.Dropdown(id='api-dropdown',
                                options=[{'label': k, 'value': v} for k, v in self.apis.items()],
                                value=self.apis[self.default_api],
                                clearable=False)               
                ], width=3),
                dbc.Col([
                    html.Label("API URL:"),
                    dbc.Input(id='api-url', type="text"),
                ]),
                dbc.Col([
                    html.Button("Get prediction!", id='predict-button'),
                
                ], width=2, align="end"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H1("Result:"),
                    html.Div(id="api-online"),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Explanation Graph"),
                    dcc.Loading(id="loading-contrib-figure", children=dcc.Graph(id='contrib-figure'))
                    
                ]),
                dbc.Col([
                    html.H3("Explanation Table"),
                    dcc.Loading(id="loading-contrib-table", children=html.Div(id='contrib-table'))
                ]) 
            ])
        ])
    
    def register_callbacks(self, app):
        @app.callback(
            Output('api-url', 'value'),
            [Input('api-dropdown', 'value')]
        )
        def update_api(api):
            return api
        
        
        @app.callback(
            [Output('monthly-income', 'value'),
             Output('age', 'value'),
             Output('no-dependents', 'value'),
             Output('revolving-utilization', 'value'),
             Output('debt-ratio', 'value'),
             Output('open-creditlines', 'value'),
             Output('no-real-estate-loans', 'value'),
             Output('no-30-59-days-due', 'value'),
             Output('no-60-89-days-due', 'value'),
             Output('no-90-days-late', 'value')],
            [Input("sample-button", "n_clicks")])
        def update_inputs(n_clicks):
            sample_df = self.df.sample(1).iloc[0]
            return [sample_df['MonthlyIncome'],
                    sample_df['age'],
                    sample_df['NumberOfDependents'],
                    np.round(sample_df['RevolvingUtilizationOfUnsecuredLines'], 2),
                    np.round(sample_df['DebtRatio'], 2),
                    sample_df['NumberOfOpenCreditLinesAndLoans'],
                    sample_df['NumberRealEstateLoansOrLines'],
                    sample_df['NumberOfTime30-59DaysPastDueNotWorse'],
                    sample_df['NumberOfTime60-89DaysPastDueNotWorse'],
                    sample_df['NumberOfTimes90DaysLate']]

        @app.callback(
            [Output("contrib-figure", "figure"),
             Output("contrib-table", "children"),
             Output("api-online", "children")],
            [Input("predict-button", "n_clicks")],
            [State('monthly-income', 'value'),
             State('age', 'value'),
             State('no-dependents', 'value'),
             State('revolving-utilization', 'value'),
             State('debt-ratio', 'value'),
             State('open-creditlines', 'value'),
             State('no-real-estate-loans', 'value'),
             State('no-30-59-days-due', 'value'),
             State('no-60-89-days-due', 'value'),
             State('no-90-days-late', 'value'),
             State("api-url", "value")])
        def update_contrib_graph(n_clicks, monthly_income, age, 
                                 no_dependents, revolving_utilization, 
                                 debt_ratio, open_creditlines, real_estate_loans,
                                 no_30_59_days_due, no_60_89_days_due, no_90_days_late, api):
            if n_clicks is not None and n_clicks > 0:
                sample_df = construct_sample_row(monthly_income, age, no_dependents, revolving_utilization, debt_ratio, open_creditlines, real_estate_loans, no_30_59_days_due, no_60_89_days_due, no_90_days_late)

                sample_json = sample_df.to_json(orient='records')
                header = {'Content-Type': 'application/json', 'Accept': 'application/json'}
                #resp = requests.post(self.api_url, data=json.dumps(sample_json), headers=header)
                try:
                    resp = requests.post(api, data=json.dumps(sample_json), headers=header)

                    preds = np.asarray(resp.json()['prediction'])
                    shap_base = resp.json()['shap_base']
                    # make sure columns are in the right order:
                    shap_values = pd.DataFrame(resp.json()['shap_values'])[sample_df.columns].values[0]

                    contrib_df = get_contrib_df(shap_base, shap_values, sample_df.iloc[[0]])

                    contributions_table = dbc.Table.from_dataframe(get_contrib_summary_df(contrib_df, model_output='probability'))
                    fig = plotly_contribution_plot(contrib_df, model_output='probability')
                    api_online = dbc.Alert("API online and working", color="primary")
                    return fig, contributions_table, api_online
                except:
                    raise
                    api_offline = dbc.Alert("API appears offline or not working", color="danger")
                    return dash.no_update, dash.no_update, api_offline 
            raise PreventUpdate



db = CreditExplainerDashboard(df, apis, default_api="lambda")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "Credit Explainer"
app.layout = db.layout()
db.register_callbacks(app)

server = app.server

if __name__=="__main__":
    app.run_server(port=8070)