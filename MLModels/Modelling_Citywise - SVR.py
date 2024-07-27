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
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression        


modelno = 0
df = pd.read_csv(r'Data\FCB_MS100_u_Surf_CLT_out_F0_F7_train.csv')
df_test = pd.read_csv(r'Data\FCB_MS100_u_Surf_CLT_out_F0_F7_test.csv')
#print(df.head(5))
today = datetime.now()

if today.hour < 12:
    h = "00"
else:
    h = "12"
dirname = 'ModelMetrics/'+ (today.strftime('%Y%m%d'))

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

# Basic modeling 

#  Feature Transformation
XTrain_Transformed = common.FeatureTransformation_Tallwood(XTrain)
XTest_Transformed = common.FeatureTransformation_Tallwood(XTest)


# Feature scaling - Standard scalar  
scaler_x = StandardScaler()
scaler_x.fit(XTrain_Transformed)
XTrain_ScaledTransformed = pd.DataFrame(scaler_x.transform(XTrain_Transformed) , columns = XTrain_Transformed.columns)
XTest_ScaledTransformed  = pd.DataFrame(scaler_x.transform(XTest_Transformed) , columns = XTest_Transformed.columns)

scaler_y = StandardScaler()
scaler_y.fit(np.array(yTrain).reshape(-1,1))
scaler_y.fit(np.array(yTest).reshape(-1,1))
#common.ViewFeatureDistribution(X , XTrain_ScaledTransformed , 'Before and After Scaling and Transformation' ,yTrain)
 

# Feature Selection and Modelling

# RUN 1 Baseline Feature selection
ModelDataobj = datamodels.ModelData(XTrain_ScaledTransformed , yTrain , XTest_ScaledTransformed , yTest , '','','','','', '', '' )

results = []
#kernals =  {'linear', 'poly', 'rbf', 'sigmoid' }
kernals =  {'linear'}

FeatureSelectionReports = []
result = {}    

for kernal in kernals: 
    for n in range(1,22):     
           for e in np.arange(0,2,0.1):           
                if kernal in ['poly' , 'sigmoid']:                        
                    #%% 1
                        for cf in np.arange(0,1,0.1):
                            X_SVR = pd.DataFrame()
                            Y_SVR = pd.DataFrame()
                            
                            X_TestSVR = pd.DataFrame()
                            Y_TestSVR = pd.DataFrame()
                            
                            X_SVR = XTrain_ScaledTransformed.copy()
                            Y_SVR = yTrain.copy()
                            X_TestSVR = XTest_ScaledTransformed.copy()
                            Y_TestSVR = yTest.copy()
                            FeatureSelectionReport.featureSelection = datamodels.FeatureSelection.RecursiveFeatureElimination.name 
                            regressor = SVR(kernel =  kernal , epsilon = e , coef0 = cf   )
                            rfe = RFE(estimator = regressor , n_features_to_select=n)
                            rfe.fit(X_SVR, Y_SVR)
                            
                            X_train_selected = rfe.transform(X_SVR)
                            X_test_selected  = rfe.transform(X_TestSVR)
                            FeatureSelectionReport.selectedFeatures = rfe.get_feature_names_out()           
                            FeatureSelectionReport.X_train_selected = X_train_selected
                            FeatureSelectionReport.X_test_selected = X_test_selected
                            FeatureSelectionReport.modelobject  = regressor
                            result = {}
                            modelno += 1
                            baseModel = 'SVR - ' + kernal                   
                             
                            # Train regressor with 
                            ftRegressor = SVR(kernel =  kernal , epsilon = e , gamma = gamma , coef0 = cf   )
                            ftRegressor.fit(X_train_selected)
                            ypred_Train = ftRegressor.predict(X_train_selected)
                            # check residual
                            #plt.scatter( x = ypred_Train , y = yTrain)
                            #plt.show()    
                            
                            
                            # MODEL PREDICTION
                            # Predict with best model
                            ypred_Test = ftRegressor.predict(X_test_selected)   
                            # check residual
                            plt.scatter( x = ypred_Test , y = Y_TestSVR)
                            plt.xlabel('Predicted Value Mean MC (SVR Regression)')
                            plt.ylabel('Actual Mean MC')
                            plt.title(modelno)
                            plt.show() 
                            
                            residual = Y_TestSVR - ypred_Test
                            sns.regplot(y = residual, x = ypred_Test)
                            plt.xlabel('Predicted Value  (SVR Regression)')
                            plt.ylabel('Residual')
                            plt.title(modelno)
                            plt.show()
                                    
                            
                            TestDatametrics = common.regressionMetrics(Y_TestSVR , ypred_Test )
                         
                            result = {'ModelNo' : modelno} | {'Model' : 'Global Model'} | {'Model Type' : 'Baseline with Feature transformation ,scaling and selection'} | {'FeatSelection' : FeatureSelectionReport.featureSelection} | {'Kernal' : kernal} |  vars(FeatureSelectionReport) | {'RMSE' : TestDatametrics.RMSE }  | vars(TestDatametrics)  
                            results.append(result)
                    #%%1   
                else:
                   FeatureSelectionReport = datamodels.FeatureSelectionReport('SVR','' , 0, [],pd.DataFrame(),pd.DataFrame() ,'')   
                   FeatureSelectionReport.noofFeatures = n
                   X_SVR = pd.DataFrame()
                   Y_SVR = pd.DataFrame()
                   
                   X_TestSVR = pd.DataFrame()
                   Y_TestSVR = pd.DataFrame()
                   
                   X_SVR = XTrain_ScaledTransformed.copy()
                   Y_SVR = yTrain.copy()
                   X_TestSVR = XTest_ScaledTransformed.copy()
                   Y_TestSVR = yTest.copy()
                   FeatureSelectionReport.featureSelection = datamodels.FeatureSelection.RecursiveFeatureElimination.name 
                   regressor = SVR(kernel =  kernal , epsilon = e)
                   rfe = RFE(estimator = regressor , n_features_to_select=n)
                   fit = rfe.fit(X_SVR, Y_SVR)
                   
                   X_train_selected = fit.transform(X_SVR)
                   X_test_selected  = fit.transform(X_TestSVR)
                   FeatureSelectionReport.selectedFeatures = fit.get_feature_names_out()           
                   FeatureSelectionReport.X_train_selected = X_train_selected
                   FeatureSelectionReport.X_test_selected = X_test_selected
                   
                   result = {}
                   modelno += 1
                   baseModel = 'SVR - ' + kernal                   
                    
                   # Train regressor with 
                   ftRegressor = SVR(kernel =  kernal , epsilon = e   )
                   ftRegressor.fit(X_train_selected , Y_SVR )
                   FeatureSelectionReport.modelobject  = ftRegressor
                   ypred_Train = ftRegressor.predict(X_train_selected )
                   # check residual
                   #plt.scatter( x = ypred_Train , y = yTrain)
                   #plt.show()    
                   
                   
                   # MODEL PREDICTION
                   # Predict with best model
                   ypred_Test = ftRegressor.predict(X_test_selected )   
                   # check residual
                   plt.scatter( x = ypred_Test , y = Y_TestSVR)
                   plt.xlabel('Predicted Value Mean MC (SVR Regression)')
                   plt.ylabel('Actual Mean MC')
                   plt.title(modelno)
                   plt.show() 
                   
                   residual = Y_TestSVR - ypred_Test
                   sns.regplot(y = residual, x = ypred_Test)
                   plt.xlabel('Predicted Value  (SVR Regression)')
                   plt.ylabel('Residual')
                   plt.title(modelno)     
                   plt.show()
                           
                   FeatureSelectionReports.append(FeatureSelectionReport)
                   TestDatametrics = common.regressionMetrics(Y_TestSVR , ypred_Test )
                
                   result = {'ModelNo' : modelno} | {'Model' : 'Global Model'} | {'Model Type' : 'Baseline with Feature transformation ,scaling and selection'} | {'FeatSelection' : FeatureSelectionReport.featureSelection} | {'Kernal' : kernal} | {'epsilon': e} | {'No of Features' : n} | vars(FeatureSelectionReport) | {'RMSE' : TestDatametrics.RMSE }  | vars(TestDatametrics)  
                   results.append(result)            
        
# BEST MODEL - SVR - Linear kernal as base model -  - Model no 9 after manual inspection
# Test the model on City wise data        
 

# XTrain, yTrain, XTest, yTest , hyp , mtricsTrain,  mtricsTest ,bestParams, version, modelName = ''):
cityresults = []
bestmodelno = 322 
bestModel =  FeatureSelectionReports[bestmodelno - 1]
Xtest_cities = pd.DataFrame(bestModel.X_test_selected)  # Feature transformed,scaled and feature selected test set 
Xtest_cities = Xtest_cities.join(df_testCity['YEAR'])
Xtest_cities = Xtest_cities.join(df_testCity['CITY'])
Ytest_cities   = pd.DataFrame((df_testCity.iloc[ : , -1]))
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
    
    cityModel = bestModel.modelobject
    ypred_test_city = np.array(cityModel.predict(Xtest_city)).reshape(-1,1)
    ypred_test_city =  pd.DataFrame( ypred_test_city , columns= ['MeanMC'])
    
    selectedFeature = bestModel.selectedFeatures    
    # Metrics
    metricsCitywise = ''
    metricsCitywise = common.regressionMetrics(Yactual_test_city , ypred_test_city )
    d1     = { 'Shapiro - Stats': metricsCitywise.shapiro.statistics , 'Shapiro - pvalue ': round(metricsCitywise.shapiro.pvalue,4)}
    d2     = { 'DAgusto - Stats': metricsCitywise.dagusto.statistics , 'DAgusto - pvalue ': round(metricsCitywise.dagusto.pvalue,4)}
    
    # city wise ranking 
    # make_scatter_plot_res_pred_runLevel(run_df, city, clad, period, ms, outputdir):
    s1 = pd.Series(year_df )
    s1 = s1.reset_index(drop=True)
    
    s2 = pd.Series(Yactual_test_city['MeanMC'])
    s2 = s2.reset_index(drop=True)
    
    s3 = pd.Series(ypred_test_city['MeanMC'])
    s3.name = 'PredMC'
    s3 = s3.reset_index(drop=True)
    city_df = pd.concat([s1,s2,s3],axis=1)
    
    #city_df.rename(columns = {list(city_df)[1]:'MeanMC'},  inplace = True)
    city_df.rename(columns = {list(city_df)[2]:'PredMC'},  inplace = True)
    
    # check residual
    plt.scatter( x = ypred_test_city , y = Yactual_test_city)
    plt.xlabel('Predicted Value Mean MC (Linear Regression)')
    plt.ylabel('Actual Mean MC')
    plt.title('Linear Regression - Best model ' + city)
    plt.show() 
    
    residual = s2 - s3
    sns.regplot(y = residual, x = s3)
    plt.xlabel('Predicted Value  (Linear Regression)')
    plt.ylabel('Residual')
    plt.title('Linear Regression - Best model ' + city)
    plt.show()
    
    
    cityrank = ranking.make_scatter_plot_res_pred_runLevel(city_df, city, 'tallwood','F0-F7', '' , dirname + '/CityWiseRanking_SVR/')
    InformationCriterion = common.logLiklihood(Yactual_test_city  , ypred_test_city  , Xtest_city.shape[1] )
    result = ''
    result = {'ModelNo' : bestmodelno} | {'Model' : 'Global Model - Best Model'} | {'ModelCityTest' : city } | {'CityRank': cityrank} |{'Model Type' : 'Baseline with Feature selection'} | { 'Selected Feature' : selectedFeature  } | vars(metricsCitywise)  
    cityresults.append(result)                 
 
#%%   Folder and File save - Metrics file     


if not os.path.exists(dirname):
   os.makedirs(dirname)
 
w1 = pd.DataFrame(results)
w2 = pd.DataFrame(cityresults)
writer = pd.ExcelWriter( dirname + '/' + 'ModelResults_SVR.xlsx')
w1.to_excel(writer, sheet_name = 'SVR', index=False)
w2.to_excel(writer, sheet_name = 'SVR-CityWise', index=False)
writer.save()
#%% 
































 