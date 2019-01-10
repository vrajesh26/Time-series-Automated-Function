
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from pylab import rcParams
import datetime as dt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
from sklearn.metrics import mean_squared_error


# In[3]:


def time_series(Timestamp, Target, seasonality=12, forecast_steps=50):
    warnings.filterwarnings("ignore")
    data=pd.DataFrame(columns=["Timestamp","Target"])
    data['Timestamp']=Timestamp
    data.sort_values(by='Timestamp',inplace=True)
    data['Target']=Target
    y=pd.DataFrame(data)
    y.set_index('Timestamp',inplace=True)
    y['Target'].plot(figsize=(15,5))
    plt.title('Target')
    plt.show()
    print('\nSeasonal Decomposition Parameters\n')
    rcParams['figure.figsize'] = 18, 8
    decomposition=sm.tsa.seasonal_decompose(y)
    fig = decomposition.plot()
    plt.title('Seasonal Decomposition Parameters')
    plt.show()
    adf = adfuller(y['Target'])
    print("Dickey Fuller Test p-value: {}".format(float(adf[1])))

    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seasonality) for x in list(itertools.product(p, d, q))] 
    #SARIMAX
    aic_values=[]
    mape_values=[]
    pdq_param=[]
    seasonal_param=[]
    test_date=int(round(y.shape[0]*0.75,0))
    x=y.index[test_date]
    y_truth = y[x.date():]
    print("\nSARIMAX Model\n")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                results = mod.fit()
                pred = results.get_prediction(start=pd.to_datetime(x), dynamic=False)
                y_forecasted = pred.predicted_mean
                print('ARIMA{}x{} - AIC:{}, MAPE:{}'.format(param, param_seasonal, results.aic,mean_absolute_percentage_error(y_truth,y_forecasted)))
                a=results.aic
                aic_values.append(a)
                b=mean_absolute_percentage_error(y_truth,y_forecasted)
                mape_values.append(b)
                d=param
                e=param_seasonal
                pdq_param.append(d)
                seasonal_param.append(e)
            except:
                continue
    sarimax_validate=pd.DataFrame({'order':pdq_param,'season_order':seasonal_param,'aic':aic_values,'MAPE':mape_values})
    param_record=mape_values.index(min(mape_values))
    mod_sarimax = sm.tsa.statespace.SARIMAX(y, order=pdq_param[param_record],
                                seasonal_order=seasonal_param[param_record],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results_sarimax = mod_sarimax.fit()
    print("\nValidation Chart:")
    pred = results_sarimax.get_prediction(start=pd.to_datetime(x), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y[y.index[0]:].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Target')
    plt.legend()
    plt.show()
    y_forecasted = pred.predicted_mean
    y_truth = y[x.date():]
    sarimax_mse=mean_squared_error(y_forecasted,y_truth)
    sarimax_mape=mean_absolute_percentage_error(y_truth,y_forecasted)
    print('The Mean Absolute Percentage Error of our forecast is {}%'.format(round(mean_absolute_percentage_error(y_truth,y_forecasted),2)))
    print('The Mean Squared Error of our forecast is {}'.format(round(sarimax_mse, 2)))
    print("\nForecast Chart:")
    pred_uc = results_sarimax.get_forecast(steps=forecast_steps)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Target')
    plt.legend()
    plt.show()
    #ARIMA
    print("\nARIMA Model\n")
    test_date=int(round(y.shape[0]*0.75,0))
    x=y.index[test_date]
    aic_values=[]
    mape_values=[]
    pdq_param=[]
    for param in pdq:
        try:
            mod = ARIMA(y,order=param)
            results = mod.fit()
            forecast = results.predict(start=pd.to_datetime(x), end=pd.to_datetime(y.index[len(y)-1]))
            a=results.aic
            aic_values.append(a)
            d=param
            pdq_param.append(d)
            if param[1]==1:
                forecast_out=pd.DataFrame(forecast,columns=['out'])
                final_forecast=y.ix[x.date():,"Target"].shift(1)+forecast_out.ix[:,"out"]
                final_forecast=final_forecast[1:]
                y_truth=y[x.date():]
                y_truth=y_truth[1:]
                print('ARIMA{} - AIC:{}, MAPE:{}'.format(param, results.aic,mean_absolute_percentage_error(y_truth,final_forecast)))
                b=mean_absolute_percentage_error(y_truth,final_forecast)
                mape_values.append(b)
            else:
                print('ARIMA{} - AIC:{}, MAPE:{}'.format(param, results.aic,mean_absolute_percentage_error(y_truth,forecast)))
                b=mean_absolute_percentage_error(y_truth,forecast)
                mape_values.append(b)
        except:
            continue
    arima_validate=pd.DataFrame({'order':pdq_param,'aic':aic_values,'MAPE':mape_values})
    writer = pd.ExcelWriter("time_series_validation.xlsx")
    sarimax_validate.to_excel(writer,sheet_name="SARIMAX",index=False)
    arima_validate.to_excel(writer,sheet_name="ARIMA",index=False)
    writer.save()
    param_record=mape_values.index(min(mape_values))
    mod_arima = ARIMA(y, order=pdq_param[param_record])
    results_arima = mod_arima.fit()
    forecast = results_arima.predict(start=pd.to_datetime(x), end=pd.to_datetime(y.index[len(y)-1]))
    print("\nValidation Chart:")
    if pdq_param[param_record][1]==1:
        forecast_out=pd.DataFrame(forecast,columns=['out'])
        final_forecast=y.ix[x.date():,"Target"].shift(1)+forecast_out.ix[:,"out"]
        final_forecast=final_forecast[1:]
        y_truth=y[x.date():]
        y_truth=y_truth[1:]
        arima_mse=mean_squared_error(y_truth,final_forecast)
        arima_mape=mean_absolute_percentage_error(y_truth,final_forecast)
        print('The Mean Absolute Percentage Error of our forecast is {}%'.format(round(mean_absolute_percentage_error(y_truth,final_forecast),2)))
        plt.plot(y_truth,label="Actual")
        plt.plot(final_forecast,label="Predicted")
        plt.legend(loc='best')
        plt.title('ARIMA Prediction')
        plt.show()
    else:
        arima_mse=mean_squared_error(y_truth,final_forecast)
        arima_mape=mean_absolute_percentage_error(y_truth,final_forecast)
        print('The Mean Absolute Percentage Error of our forecast is {}%'.format(round(mean_absolute_percentage_error(y_truth,forecast),2)))
        plt.plot(y_truth,label="Actual")
        plt.plot(forecast,label="Predicted")
        plt.legend(loc='best')
        plt.title('ARIMA Prediction')
        plt.show()
    start_index = len(y)
    end_index = start_index + forecast_steps
    print("\nForecast Chart")
    results_arima.plot_predict(start=start_index, end=end_index,plot_insample=False)
    plt.show()
    print("\n")
    if sarimax_mape<arima_mape:
        print("The best model is SARIMAX")
        print("\n")
        print(results_sarimax.summary())
        results_sarimax.plot_diagnostics(figsize=(16, 8))
        plt.show()
        return results_sarimax
    else:
        print("The best model is ARIMA")
        print("\n")
        print(results_arima.summary())
        return results_arima


# In[ ]:


class time_series:
    pass

