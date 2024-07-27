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
from sklearn.cross_decomposition import PLSRegression
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
scaler_x.fit(XTrain)
XTrain_Scaled = pd.DataFrame(scaler_x.transform(XTrain_Transformed) , columns = XTrain.columns)
XTest_Scaled  = pd.DataFrame(scaler_x.transform(XTest_Transformed) , columns = XTest.columns)

#scaler_y = StandardScaler()
#scaler_y.fit(np.array(yTrain).reshape(-1,1))
#scaler_y.fit(np.array(yTest).reshape(-1,1))
 

# Feature Selection and Modelling

# RUN 1 Baseline Feature selection
ModelDataobj = datamodels.ModelData(XTrain_Scaled , yTrain , XTest_Scaled , yTest , '','','','','', '', '' )
results = []
num_principal_components  = []
result = {}    

# Step 2 - Perform pls
featurelist = XTrain_Scaled.shape[1]
FeatureSelectionReport = datamodels.FeatureSelectionReport('pls','' , 0, [],pd.DataFrame(),pd.DataFrame() ,'')  
dfpls = pd.DataFrame()
for n in range(1,featurelist) :
        FeatureSelectionReport.noofFeatures = n
        X_PLS = pd.DataFrame()
        Y_PLS = pd.DataFrame()
        
        X_TestPLS = pd.DataFrame()
        Y_TestPLS = pd.DataFrame()
        
        X_PLS = XTrain_Scaled.copy()                    
        Y_PLS = yTrain.copy()
        X_TestPLS = XTest_Scaled.copy()
        Y_TestPLS = yTest.copy()        
        
        pls = PLSRegression(n_components = n)
        pls.fit(X_PLS , Y_PLS)
        
        
        pls.transform(X_PLS)
        pls.transform(X_TestPLS)
        # Step 3: Plot explained variance ratio
        
        
        # Get the principal component loadings
        loadings = pls.coef_
        
        # Calculate absolute loadings
        absolute_loadings = np.abs(loadings)
        
        # Sum loadings across principal components
        sum_loadings = np.sum(absolute_loadings, axis=0)
        
        # Create a list of feature names (e.g., for Iris dataset)
        feature_names =  X_PLS.columns
        
        # Create a ranked list of features based on sum of loadings
        plsRankRowSorted = [feature for _, feature in sorted(zip(absolute_loadings, feature_names), reverse=True)]
        # loop through features 
        
        for counter in  range(len(plsRankRowSorted)):
            i = 1
            features = []
            for i in range(counter + 1):
                features.append((plsRankRowSorted[i]) )
            data = {'pls Components' : [n],
                        'FeatureList'    : [features] ,
                        'No of Features' : [i + 1]}    
            dfpls = dfpls.append(data , ignore_index=True)
    # FeatureSelectionReport.selectedFeatures = fit.get_feature_names_out()           
    # FeatureSelectionReport.X_train_selected = X_train_selected
    # FeatureSelectionReport.X_test_selected = X_test_selected
    # FeatureSelectionReport.modelobject  = regressor
    # result = {}
    
    # modelno += 1
    # baseModel = 'SVR - ' + kernal                   
     
    # # Train regressor with 
    # ftRegressor = SVR(kernel =  kernal , epsilon = e   )
    # ftRegressor.fit(X_train_selected , Y_PLS )
    # ypred_Train = ftRegressor.predict(X_train_selected )
    # # check residual
    # #plt.scatter( x = ypred_Train , y = yTrain)
    # #plt.show()    
    
    
    # # MODEL PREDICTION
    # # Predict with best model
    # ypred_Test = ftRegressor.predict(X_test_selected )   
    # # check residual
    # plt.scatter( x = ypred_Test , y = Y_TestPLS)
    # plt.xlabel('Predicted Value Mean MC (SVR Regression)')
    # plt.ylabel('Actual Mean MC')
    # plt.title(modelno)
    # plt.show() 
    
    # residual = Y_TestPLS - ypred_Test
    # sns.regplot(y = residual, x = ypred_Test)
    # plt.xlabel('Predicted Value  (SVR Regression)')
    # plt.ylabel('Residual')
    # plt.title(modelno)     
    # plt.show()
            
    # FeatureSelectionReports.append(FeatureSelectionReport)
    # TestDatametrics = common.regressionMetrics(Y_TestPLS , ypred_Test )
 
    #result = {'ModelNo' : modelno} | {'Model' : 'Global Model'} | {'Model Type' : 'Baseline with Feature transformation ,scaling and selection'} | {'FeatSelection' : FeatureSelectionReport.featureSelection} | {'No of Features' : n} | vars(FeatureSelectionReport) | {'RMSE' : TestDatametrics.RMSE }  | vars(TestDatametrics)  
    #results.append(result)            
        
# BEST MODEL - pls
# Test the model on City wise data        
 

#%% citywise
# cityresults = []
# bestModel =  FeatureSelectionReports[321]
# Xtest_cities = pd.DataFrame(bestModel.X_test_selected)  # Feature transformed,scaled and feature selected test set 
# Xtest_cities = Xtest_cities.join(df_testCity['YEAR'])
# Xtest_cities = Xtest_cities.join(df_testCity['CITY'])
# Ytest_cities   = pd.DataFrame((df_testCity.iloc[ : , -1]))
# Ytest_cities = Ytest_cities.join(df_testCity['YEAR'])
# Ytest_cities = Ytest_cities.join(df_testCity['CITY'])

# cities =  Xtest_cities['CITY'].unique()   
# for city in cities:
   
#     Xtest_city = pd.DataFrame()
#     Yactual_test_city = pd.DataFrame()    
    
#     Xtest_city = Xtest_cities.loc[df_testCity['CITY'] == city]
#     Yactual_test_city = Ytest_cities.loc[df_testCity['CITY'] == city]
     
#     year_df = Xtest_city['YEAR']
#     Xtest_city = Xtest_city.drop(columns = ['YEAR','CITY'])
#     Yactual_test_city = Yactual_test_city.drop(columns = ['YEAR','CITY'])        
    
#     cityModel = bestModel.modelobject
#     ypred_test_city = np.array(cityModel.predict(Xtest_city)).reshape(-1,1)
#     ypred_test_city =  pd.DataFrame( ypred_test_city , columns= ['MeanMC'])
    

    
#     # Metrics
#     metricsCitywise = ''
#     metricsCitywise = common.regressionMetrics(Yactual_test_city , ypred_test_city )
#     d1     = { 'Shapiro - Stats': metricsCitywise.shapiro.statistics , 'Shapiro - pvalue ': round(metricsCitywise.shapiro.pvalue,4)}
#     d2     = { 'DAgusto - Stats': metricsCitywise.dagusto.statistics , 'DAgusto - pvalue ': round(metricsCitywise.dagusto.pvalue,4)}
    
#     # city wise ranking 
#     # make_scatter_plot_res_pred_runLevel(run_df, city, clad, period, ms, outputdir):
#     s1 = pd.Series(year_df )
#     s1 = s1.reset_index(drop=True)
    
#     s2 = pd.Series(Yactual_test_city['MeanMC'])
#     s2 = s2.reset_index(drop=True)
    
#     s3 = pd.Series(ypred_test_city['MeanMC'])
#     s3.name = 'PredMC'
#     s3 = s3.reset_index(drop=True)
#     city_df = pd.concat([s1,s2,s3],axis=1)
    
#     #city_df.rename(columns = {list(city_df)[1]:'MeanMC'},  inplace = True)
#     city_df.rename(columns = {list(city_df)[2]:'PredMC'},  inplace = True)
    
#     # check residual
#     plt.scatter( x = ypred_test_city , y = Yactual_test_city)
#     plt.xlabel('Predicted Value Mean MC (Linear Regression)')
#     plt.ylabel('Actual Mean MC')
#     plt.title('Linear Regression - Best model ' + city)
#     plt.show() 
    
#     residual = s2 - s3
#     sns.regplot(y = residual, x = s3)
#     plt.xlabel('Predicted Value  (Linear Regression)')
#     plt.ylabel('Residual')
#     plt.title('Linear Regression - Best model ' + city)
#     plt.show()
    
    
#     cityrank = ranking.make_scatter_plot_res_pred_runLevel(city_df, city, 'tallwood','F0-F7', '' , dirname + '/CityWiseRanking_LR/')
#     InformationCriterion = common.logLiklihood(Yactual_test_city  , ypred_test_city  , Xtest_city.shape[1] )
#     result = ''
#     result = {'ModelNo' : modelno} | {'Model' : 'Global Model - Best Model'} | {'ModelCityTest' : city } | {'CityRank': cityrank} |{'Model Type' : 'Baseline with Feature selection'} | vars(selectedFeature) | vars(metricsCitywise)  | d1 | d2 
#     cityresults.append(result)                 
#%%
#%%   Folder and File save - Metrics file     


if not os.path.exists(dirname):
   os.makedirs(dirname)
 
w1 = pd.DataFrame(results)
#w2 = pd.DataFrame(cityresults)
writer = pd.ExcelWriter( dirname + '/' + 'ModelResults_SVR.xlsx')
w1.to_excel(writer, sheet_name = 'SVR', index=False)
#w2.to_excel(writer, sheet_name = 'SVR-CityWise', index=False)
writer.save()
#%% 
































 