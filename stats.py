import pandas as pd
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def create_df():
    """Create grouped df
    """
    df = pd.read_csv('crime_c.csv')
    df['REPORTED_DATE'] = pd.to_datetime(df['REPORTED_DATE'])
    df['REPORTED_DATE'] = df['REPORTED_DATE'].dt.date
    grouped_df = pd.DataFrame(columns=['REPORTED_DATE', 'N_CRIMES'])
    grouped_df['REPORTED_DATE'] = df['REPORTED_DATE'].unique()
    grouped_df['N_CRIMES'] = [len(df[df['REPORTED_DATE'] == date]) for date in grouped_df['REPORTED_DATE']]
    grouped_df = grouped_df.sort_values('REPORTED_DATE')
    grouped_df.to_csv('crime_gr.csv', index=False)


def predict():
    """Compare stats models
    """
    df = pd.read_csv('crime_gr.csv')
    df = df['N_CRIMES']

    train, test = list(df[:-10]), list(df[-10:])

    model = AR(train)
    model_fit = model.fit()

    # make prediction
    prediction = model_fit.predict(len(train), len(train) + 9)
    metric = rmse(test, prediction)
    print('Autoregression', metric)

    model = ARMA(train, order=(0, 1))
    model_fit = model.fit(disp=False)

    # make prediction
    prediction = model_fit.predict(len(train), len(train) + 9)
    metric = rmse(test, prediction)
    print('Moving average', metric)

    model = SimpleExpSmoothing(train)
    model_fit = model.fit()

    # make prediction
    prediction = model_fit.predict(len(train), len(train) + 9)
    metric = rmse(test, prediction)
    print('Exp', metric)


predict()
