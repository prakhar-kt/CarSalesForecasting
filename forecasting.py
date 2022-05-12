import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
# st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)


st.title('Sales Forecasting')

st.write("A Forecasting Model for Car Sales")

appdata = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv',header=0)
display_data = st.checkbox('Show data')

if display_data:
     st.dataframe(appdata, 1200,200)

appdata.columns = ['ds', 'y']


appdata['ds'] = pd.to_datetime(appdata['ds'])

max_date = appdata['ds'].max()

st.write("SELECT FORECAST PERIOD")

# periods_input = st.number_input('How many months forecast do you want?', min_value=1, max_value=24)

periods_input = st.slider('How many months forecast do you want?', min_value=1, max_value=24,value=1,step=1)
model = Prophet()
model.fit(appdata)

st.write('VISUALISE FORECASTED DATA')


future = list()
for i in range(1, periods_input+1):
    if i <= 12:
        date = '1968-%02d' % i
    else:
        i = i-12
        date = '1969-%02d' % i
    future.append([date])

future = pd.DataFrame(future)
future.columns = ['ds']
future['ds'] = pd.to_datetime(future['ds'])


fcst = model.predict(future)

forecast = fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# forecast_filtered = forecast[forecast['ds'] > max_date]
st.write('This visual shows the actual (black dots) & predicted \
      (blue line) values over time.')

figure1 = model.plot(forecast)

st.write(figure1)


st.write("The following plot shows future predicted values. 'yhat' is the \
 predicted value; upper and lower limits are 80% confidence intervals by \
 default")

st.write(forecast)


# st.write(figure1)

# st.write("The following plots show a high level trend of predicted \
#       values, day of week trends and yearly trends (if dataset contains \
#       multiple years’ data).Blue shaded area represents upper and lower  \
#       confidence intervals.")
#
# figure2 = model.plot_components(fcst)
# st.write(figure2)
#
#
#


