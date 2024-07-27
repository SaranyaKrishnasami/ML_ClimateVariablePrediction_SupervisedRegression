# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:43:47 2023

@author: Saranya
Linear model- Linear model
1) Previous modelling 
    - Feature scaling
    - Feature Transformation
    - Check residual
    - Best Model
"""
import DataModels as datamodels
import CommonFunctions as common
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score as R2
from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor, RandomForestRegressor
from datetime import datetime
import os
import ranking as ranking
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
modelno = 0
df = pd.read_csv(r'Data\FCB_MS100_u_Surf_CLT_out_F0_F7_train.csv')
df_test = pd.read_csv(r'Data\FCB_MS100_u_Surf_CLT_out_F0_F7_test.csv')
#print(df.head(5))


# FLAGS 
citywiseModels = 0

df = df.drop(columns=['ISEV' , 'Unnamed: 0.1' , 'Unnamed: 0' , 'RH2' , 'TEMP2' , 'PV2' ,'YEAR', 'CITY' , 'PERIOD' , 'RUN', 'MaxMC'] )
df_testCity = df_test.drop(columns=['ISEV' , 'Unnamed: 0.1' , 'Unnamed: 0' , 'RH2' , 'TEMP2' , 'PV2' ,  'PERIOD' , 'RUN', 'MaxMC'] )
df_test = df_test.drop(columns=['ISEV' , 'Unnamed: 0.1' , 'Unnamed: 0' , 'RH2' , 'TEMP2' , 'PV2' ,'YEAR', 'CITY' , 'PERIOD' , 'RUN', 'MaxMC'] )
results = []
XTrain = (df.iloc[ : , : -1 ])
yTrain = (df.iloc[ : , -1])
X = XTrain.copy()
Y = yTrain.copy() 
XTest = (df_test.iloc[ : , : -1 ])
yTest = (df_test.iloc[ : , -1])
y_ = (df_test.iloc[ : , -1])

power =  PowerTransformer(method='yeo-johnson', standardize=True)
power.fit(np.array(yTrain).reshape(-1,1))
Y = power.transform(np.array(yTrain).reshape(-1,1))
 

# Basic modeling 

#  Feature Transformation
XTrain_Transformed = common.FeatureTransformation_Tallwood(XTrain)
XTest_Transformed = common.FeatureTransformation_Tallwood(XTest)


# Feature scaling - Standard scalar  
scaler = StandardScaler()
scaler.fit(XTrain_Transformed)
XTrain_ScaledTransformed = pd.DataFrame(scaler.transform(XTrain_Transformed) , columns = XTrain_Transformed.columns)
XTest_ScaledTransformed  = pd.DataFrame(scaler.transform(XTest_Transformed) , columns = XTest_Transformed.columns)
#common.ViewFeatureDistribution(X , XTrain_ScaledTransformed , 'Before and After Scaling and Transformation' ,yTrain)

# Feature Selection and Modelling
#%%
# RUN 1 Baseline Feature selection
ModelDataobj = datamodels.ModelData(XTrain_ScaledTransformed , Y , XTest_ScaledTransformed , yTest , '','','','','', '', '' )
 
results = [] 
result = {}

from sklearn.model_selection import GridSearchCV
pipeline_quad = Pipeline([ ('scaler' ,  StandardScaler()),
                     ('model',linear_model.Ridge())])

search_quad = GridSearchCV(pipeline_quad,
                      {'model__alpha':np.arange(0,2,0.01)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3  )
search_quad.fit(XTrain_ScaledTransformed,Y) 
search_quad.best_params_
# the values of the coefficients of Lasso regression.
coefficients =  []
coefficients = search_quad.best_estimator_.named_steps['model'].coef_
print(coefficients)

#The features that survived the Lasso regression are:
importance_quad = np.abs(coefficients)
 

 
#************************************************************************************************************************
 
 
#%%

# Modeling with Rdige
results = []
modelno = 0
alphaR = search_quad.best_params_['model__alpha']
 
ridge = linear_model.Ridge(alpha = alphaR)
ridge.fit(XTrain_ScaledTransformed, Y)
Y_test_Predict = ridge.predict(XTest_ScaledTransformed)
Y_test_Predict = power.inverse_transform(Y_test_Predict)

TestDatametrics =  common.regressionMetrics(np.array(yTest).reshape(-1,1), Y_test_Predict)       


# check residual
plt.scatter( x = Y_test_Predict , y = yTest)
plt.xlabel('Predicted Value Mean MC (Ridge Regression)')
plt.ylabel('Actual Mean MC')
plt.title(modelno)
plt.show() 

residual = yTest - Y_test_Predict.flatten()
sns.regplot(y = residual, x = Y_test_Predict)
plt.xlabel('Predicted Value  (Ridge Regression)')
plt.ylabel('Residual')
plt.title(modelno)
plt.show() 
 
result = {'ModelNo' : 1} | {'Model' : 'Global Model'} | {'Model Type' : 'Baseline with Feature transformation , scaling and alpha value selection'} | {'Alpha Value' : alphaR}| {'RMSE' : TestDatametrics.RMSE }  | vars(TestDatametrics)    
results.append(result)
            
    
    
# # BEST MODEL - Lasso - Best Alpha with AIC 
# Test the model on City wise data        
 
FeatureSelectionReports = []
# XTrain, yTrain, XTest, yTest , hyp , mtricsTrain,  mtricsTest ,bestParams, version, modelName = ''):
cityresults = []

bestModel = linear_model.Ridge(alpha = alphaR)
bestModel.fit(XTrain_ScaledTransformed, Y)
Xtest_cities = XTest_ScaledTransformed
Xtest_cities = Xtest_cities.join(df_testCity['YEAR'])
Xtest_cities = Xtest_cities.join(df_testCity['CITY'])
Ytest_cities = pd.DataFrame((df_testCity.iloc[ : , -1]))
Ytest_cities = Ytest_cities.join(df_testCity['YEAR'])
Ytest_cities = Ytest_cities.join(df_testCity['CITY'])

cities =  Xtest_cities['CITY'].unique()   
for city in cities:
   
    Xtest_city = pd.DataFrame()
    Yactual_test_city = pd.DataFrame()    
    
    Xtest_city = Xtest_cities.loc[df_testCity['CITY'] == city]
    Yactual_test_city = Ytest_cities.loc[df_testCity['CITY'] == city]
     
    year_df = Xtest_city['YEAR']
    Xtest_city = Xtest_city.drop(columns = ['YEAR','CITY'])
    Yactual_test_city = Yactual_test_city.drop(columns = ['YEAR','CITY'])        
    
    cityModel = bestModel
    ypred_test_city = np.array(cityModel.predict(Xtest_city)).reshape(-1,1)
    ypred_test_city =  pd.DataFrame( ypred_test_city , columns= ['MeanMC'])    
    
    ypred_test_city = power.inverse_transform(ypred_test_city)
    ypred_test_city = np.array(ypred_test_city).reshape(-1,1)
    # Metrics
    metricsCitywise = ''
    metricsCitywise = common.regressionMetrics(Yactual_test_city , ypred_test_city )
   
    
    # city wise ranking 
    # make_scatter_plot_res_pred_runLevel(run_df, city, clad, period, ms, outputdir):
    s1 = pd.Series(year_df )
    s1 = s1.reset_index(drop=True)
    
    s2 = pd.Series(Yactual_test_city['MeanMC'])
    s2 = s2.reset_index(drop=True)
    
    s3 = pd.Series(ypred_test_city.flatten())
    s3.name = 'PredMC'
    s3 = s3.reset_index(drop=True)
    city_df = pd.concat([s1,s2,s3],axis=1)
    
    #city_df.rename(columns = {list(city_df)[1]:'MeanMC'},  inplace = True)
    city_df.rename(columns = {list(city_df)[2]:'PredMC'},  inplace = True)
    
    # check residual
    plt.scatter( x = ypred_test_city , y = Yactual_test_city)
    plt.xlabel('Predicted Value Mean MC (Ridge Regression)')
    plt.ylabel('Actual Mean MC')
    plt.title('Linear Regression - Best model ' + city)
    plt.show() 
    
    residual = s2 - s3
    sns.regplot(y = residual, x = s3)
    plt.xlabel('Predicted Value  (Ridge Regression)')
    plt.ylabel('Residual')
    plt.title('Ridge Regression - Best model ' + city)
    plt.show()
    
    InformationCriterion = common.logLiklihood(Yactual_test_city  , ypred_test_city  , Xtest_city.shape[1] )
    cityrank = ranking.make_scatter_plot_res_pred_runLevel(city_df, city, 'tallwood','F0-F7', '' , '/CityWiseRanking_Ridge/')
    result = ''
    result = {'ModelNo' : modelno} | {'Model' : 'Global Model - Best Model'} | {'ModelCityTest' : city } | {'CityRank': cityrank}  | {'Info Criterion' : InformationCriterion } |{'Model Type' : 'Baseline with Feature transformation , scaling and alpha value selection'} | {'Alpha Value' : alphaR} | vars(metricsCitywise) 
    cityresults.append(result)                 
 
#%%   Folder and File save - Metrics file     
today = datetime.now()

if today.hour < 12:
    h = "00"
else:
    h = "12"
dirname = 'ModelMetrics/'+ (today.strftime('%Y%m%d'))

if not os.path.exists(dirname):
    os.makedirs(dirname)
 
w1 = pd.DataFrame(results)
w2 = pd.DataFrame(cityresults)
writer = pd.ExcelWriter( dirname + '/' + 'ModelResults_Ridge.xlsx')
w1.to_excel(writer, sheet_name = 'Ridge', index=False)
w2.to_excel(writer, sheet_name = 'Ridge-CityWise', index=False)
writer.save()

#%% 
































 