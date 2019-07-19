import eia
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def retrieve_time_series(api, series_ID):
    """
    Return the time series dataframe, based on API and unique Series ID
    api: API that we're connected to
    series_ID: string. Name of the series that we want to pull from the EIA API
    """
    #Retrieve Data By Series ID 
    series_search = api.data_by_series(series=series_ID)
    ##Create a pandas dataframe from the retrieved time series
    df = pd.DataFrame(series_search)
    return df
    
def decompose_time_series(series):
    """
    Decompose a time series and plot it in the console
    Arguments: 
        series: series. Time series that we want to decompose
    Outputs: 
        Decomposition plot in the console
    """
    result = seasonal_decompose(series, model='additive')
    result.plot()
    pyplot.show()

def augmented_dickey_fuller_statistics(time_series):
    """
    Run the augmented Dickey-Fuller test on a time series
    to determine if it's stationary.
    Arguments: 
        time_series: series. Time series that we want to test 
    Outputs: 
        Test statistics for the Augmented Dickey Fuller test in 
        the console 
    """
    result = adfuller(time_series.values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value)) 
        
def calculate_model_accuracy_metrics(actual, predicted):
    """
    Output model accuracy metrics, comparing predicted values
    to actual values.
    Arguments:
        actual: list. Time series of actual values.
        predicted: list. Time series of predicted values
    Outputs:
        Forecast bias metrics, mean absolute error, mean squared error,
        and root mean squared error in the console
    """
    #Calculate forecast bias
    forecast_errors = [actual[i]-predicted[i] for i in range(len(actual))]
    bias = sum(forecast_errors) * 1.0/len(actual)
    print('Bias: %f' % bias)
    #Calculate mean absolute error
    mae = mean_absolute_error(actual, predicted)
    print('MAE: %f' % mae)
    #Calculate mean squared error and root mean squared error
    mse = mean_squared_error(actual, predicted)
    print('MSE: %f' % mse)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)

def main():
    """
    Run main script
    """
    #Create EIA API using your specific API key
    api_key = "YOR API KEY HERE"
    api = eia.API(api_key)
    
    #Pull the electricity price data
    series_ID='ELEC.PRICE.TX-ALL.M'
    electricity_df=retrieve_time_series(api, series_ID)
    electricity_df.reset_index(level=0, inplace=True)
    #Rename the columns for easer analysis
    electricity_df.rename(columns={'index':'Date',
            electricity_df.columns[1]:'Electricity_Price'}, 
            inplace=True)
    #Convert the Date column into a date object
    electricity_df['Date']=pd.to_datetime(electricity_df['Date'])
    #Set Date as a Pandas DatetimeIndex
    electricity_df.index=pd.DatetimeIndex(electricity_df['Date'])
    #Decompose the time series into parts
    decompose_time_series(electricity_df['Electricity_Price'])
    
    #Pull in natural gas time series data
    series_ID='NG.N3035TX3.M'
    nat_gas_df=retrieve_time_series(api, series_ID)
    nat_gas_df.reset_index(level=0, inplace=True)
    #Rename the columns
    nat_gas_df.rename(columns={'index':'Date',
            nat_gas_df.columns[1]:'Nat_Gas_Price_MCF'}, 
            inplace=True)
    #Convert the Date column into a date object
    nat_gas_df['Date']=pd.to_datetime(nat_gas_df['Date'])
    #Set Date as a Pandas DatetimeIndex
    nat_gas_df.index=pd.DatetimeIndex(nat_gas_df['Date'])
    #Decompose the time series into parts
    decompose_time_series(nat_gas_df['Nat_Gas_Price_MCF'])
    
    #Merge the two time series together based on Date Index
    master_df=pd.merge(electricity_df['Electricity_Price'], nat_gas_df['Nat_Gas_Price_MCF'], 
                       left_index=True, right_index=True)
    master_df.reset_index(level=0, inplace=True)
    
    #Plot the two variables in the same plot
    plt.plot(master_df['Date'], 
             master_df['Electricity_Price'], label="Electricity_Price")
    plt.plot(master_df['Date'], 
             master_df['Nat_Gas_Price_MCF'], label="Nat_Gas_Price")
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Natural Gas Price vs. TX Electricity Price over Time')
    plt.show()
    
    #Transform the columns using natural log
    master_df['Electricity_Price_Transformed']=np.log(master_df['Electricity_Price'])
    master_df['Nat_Gas_Price_MCF_Transformed']=np.log(master_df['Nat_Gas_Price_MCF'])
    
    #In order to make the time series stationary, difference the data by 1 month
    n=1
    master_df['Electricity_Price_Transformed_Differenced'] = master_df['Electricity_Price_Transformed'] - master_df['Electricity_Price_Transformed'].shift(n)
    master_df['Nat_Gas_Price_MCF_Transformed_Differenced'] = master_df['Nat_Gas_Price_MCF_Transformed'] - master_df['Nat_Gas_Price_MCF_Transformed'].shift(n)
    
    #Run each differenced time series thru the Augmented Dickey Fuller test
    print('Augmented Dickey-Fuller Test: Electricity Price Time Series')
    augmented_dickey_fuller_statistics(master_df['Electricity_Price_Transformed_Differenced'].dropna())
    print('Augmented Dickey-Fuller Test: Natural Gas Price Time Series')
    augmented_dickey_fuller_statistics(master_df['Nat_Gas_Price_MCF_Transformed_Differenced'].dropna())
    
    #Conver the dataframe to a numpy array
    master_array=np.array(master_df[['Electricity_Price_Transformed_Differenced', 
                                     'Nat_Gas_Price_MCF_Transformed_Differenced']].dropna())
    
    #Generate a training and test set for building the model: 95/5 split
    training_set = master_array[:int(0.95*(len(master_array)))]
    test_set = master_array[int(0.95*(len(master_array))):]
    
    #Fit to a VAR model
    model = VAR(endog=training_set)
    model_fit = model.fit()
    #Print a summary of the model results
    model_fit.summary()
    
    #Compare the forecasted results to the real data 
    prediction = model_fit.forecast(model_fit.y, steps=len(test_set))
    
    #Merge the array data back into the master dataframe, and un-difference and back-transform
    data_with_predictions=pd.DataFrame(np.vstack((training_set, 
                                        prediction))).rename(columns={0:'Electricity_Price_Transformed_Differenced_PostProcess',
                                                                      1:'Nat_Gas_Price_MCF_Transformed_Differenced_PostProcess'})
    #Define which data is predicted and which isn't in the 'Predicted' column
    data_with_predictions.loc[:,'Predicted']=1
    data_with_predictions.loc[(data_with_predictions.index>=0) & 
                                     (data_with_predictions.index<=(len(training_set)-1)),'Predicted']=0
    
    #Add a row of NaN at the begining of the df
    data_with_predictions.loc[-1] = [None, None, None]  # adding a row
    data_with_predictions.index = data_with_predictions.index + 1  # shifting index
    data_with_predictions.sort_index(inplace=True) 
    #Add back into the original dataframe
    master_df.loc[:,'Electricity_Price_Transformed_Differenced_PostProcess'] = data_with_predictions['Electricity_Price_Transformed_Differenced_PostProcess']
    master_df.loc[:,'Predicted'] = data_with_predictions['Predicted']
        
    #Un-difference the data
    for i in range(1,len(master_df.index)-1):
        master_df.at[i,'Electricity_Price_Transformed']= master_df.at[i-1,'Electricity_Price_Transformed']+master_df.at[i,'Electricity_Price_Transformed_Differenced_PostProcess']
    
    #Back-transform the data
    master_df.loc[:,'Predicted_Electricity_Price']=np.exp(master_df['Electricity_Price_Transformed'])
    
    #Compare the forecasted data to the real data
    print(master_df[master_df['Predicted']==1][['Date','Electricity_Price', 'Predicted_Electricity_Price']])
    
    #Evaluate the accuracy of the results, pre un-differencing and back-transformation
    calculate_model_accuracy_metrics(list(master_df[master_df['Predicted']==1]['Electricity_Price']), 
                                    list(master_df[master_df['Predicted']==1]['Predicted_Electricity_Price']))

    
#Run the main script
if __name__== "__main__":
    main()
