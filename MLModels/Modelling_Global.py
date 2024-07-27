# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:43:47 2023

@author: VaishuSistas
"""
import DataModels as datamodels
import CommonFunctions as common
import  pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score as R2
from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor, RandomForestRegressor
from datetime import datetime
import os
import pandas as pd
modelno = 0
df = pd.read_csv(r'Data\FCB_MS100_u_Surf_CLT_out_F0_F7_train.csv')
df_test = pd.read_csv(r'Data\FCB_MS100_u_Surf_CLT_out_F0_F7_test.csv')
#print(df.head(5))


# FLAGS 
citywiseModels = 1

df = df.drop(columns=['ISEV' , 'Unnamed: 0.1' , 'Unnamed: 0' , 'RH2' , 'TEMP2' , 'PV2' ,'YEAR', 'CITY' , 'PERIOD' , 'RUN', 'MaxMC'] )
if citywiseModels == 1:
    df_test = df_test.drop(columns=['ISEV' , 'Unnamed: 0.1' , 'Unnamed: 0' , 'RH2' , 'TEMP2' , 'PV2' ,'YEAR',  'PERIOD' , 'RUN', 'MaxMC'] )
else:
    df_test = df_test.drop(columns=['ISEV' , 'Unnamed: 0.1' , 'Unnamed: 0' , 'RH2' , 'TEMP2' , 'PV2' ,'YEAR', 'CITY' , 'PERIOD' , 'RUN', 'MaxMC'] )

X = (df.iloc[ : , : -1 ])
y = (df.iloc[ : , -1])


if citywiseModels == 1:
    FeatureSelectionReports = []
    # XTrain, yTrain, XTest, yTest , hyp , mtricsTrain,  mtricsTest ,bestParams, version, modelName = ''):
    cities =  Xtest['CITY'].unique()   
    for city in cities:
        Xtest = (df_test.iloc[ : , : -1 ])
        ytest = (df_test.iloc[ : , -1])
        Xtest_city = pd.DataFrame()
        Xtest_city = Xtest.loc[Xtest['CITY'] == city]
        Xtest_city = Xtest_city.drop(columns=['CITY'])
        
        ModelDataobj = datamodels.ModelData(X , y , Xtest_city , ytest , '','','','','', '' )
        selectedFeatureList = common.FeatureSelectionPipeline(ModelDataobj)
         
        results = []
        
        # TBD : Here we have to put the BEST models for each one
        dict_classifiers = {
        "LR": linear_model.LinearRegression(),
        "LASSO": linear_model.Lasso(),
        "Ridge": linear_model.Ridge(), #class_weight='balanced'
        "EN": linear_model.ElasticNet(random_state=0),
        "DT": DecisionTreeRegressor(random_state = 0) ,
        "RF": RandomForestRegressor(min_samples_leaf=5, random_state=0) ,
        "XGBoost": GradientBoostingRegressor(), }
        # RUN 1 Baseline Feature selection
        for selectedFeature in selectedFeatureList: 
            for modelname, model in  dict_classifiers.items(): 
                modelno += 1
                selectedFeature.baseModel = modelname
                result = {}
                XSelected = selectedFeature.X_train_selected     
                XTestSelected = selectedFeature.X_test_selected
                
                model.fit(XSelected, y)
                ypred = model.predict(XTestSelected)
                metrics = common.regressionMetrics(ytest , ypred )
                d1 = { 'Shapiro - Stats ': metrics.shapiro.statistics , 'Shapiro - pvalue ': round( metrics.shapiro.pvalue , 4 ) }
                d2 = { 'DAgusto - Stats ': metrics.dagusto.statistics , 'DAgusto - pvalue ': round(metrics.dagusto.pvalue,4)}
                result = {'ModelNo' : modelno} | {'Model' : 'Global Model'} | {'ModelCityTest' : city } | {'Model Type' : 'Baseline with Feature selection'} | vars(selectedFeature) | vars(metrics)  | d1 | d2 
                results.append(result)
                # city wise ranking 
        
# Run 2 Feature Transformation and Feature Selection

# XTransformed = df.iloc[: , : -1]
# yTransformed = df.iloc[: ,  -1]

# XTestTransformed = (df_test.iloc[ : , : -1 ])
# yTestTransformed = (df_test.iloc[ : , -1])

# XTransformed = common.FeatureTransformation_Tallwood(XTransformed)
# XTestTransformed = common.FeatureTransformation_Tallwood(XTestTransformed)

# #yTransformed = common.FeatureTransformation_Tallwood(yTransformed)
# #yTestTransformed = common.FeatureTransformation_Tallwood(yTestTransformed)

# ModelDataobj_transformed = datamodels.ModelData(XTransformed , yTransformed , XTestTransformed , yTestTransformed , '','','','','', '' )
# selectedFeatureList_transformed = common.FeatureSelectionPipeline( ModelDataobj_transformed)

# # Feature transformation and feature selection
# for selectedFeature in selectedFeatureList_transformed: 
#     for modelname, model in  dict_classifiers.items(): 
#         modelno += 1
#         selectedFeature.baseModel = modelname
#         result = {}
#         XSelected = selectedFeature.X_train_selected     
#         XTestSelected = selectedFeature.X_test_selected
        
#         model.fit(XSelected, y)
#         ypred = model.predict(XTestSelected)
#         metrics = common.regressionMetrics(yTestTransformed , ypred )
#         d1 = { 'Shapiro - Stats ': metrics.shapiro.statistics , 'Shapiro - pvalue ': round( metrics.shapiro.pvalue , 4 ) }
#         d2 = { 'DAgusto - Stats ': metrics.dagusto.statistics , 'DAgusto - pvalue ': round(metrics.dagusto.pvalue,4)}
#         result = {'ModelNo' : modelno} | {'Model' : 'Global Model'} | {'Model Type' : ' Feature Transformation -> Feature Selection -> Modelling'} | vars(selectedFeature) | vars(metrics)  | d1 | d2 
#         results.append(result)

# # Run 3 Feature Transformation , Target Transformation and Feature Selection

# XTargetTransformed = df.iloc[: , : -1]
# yTargetTransformed = pd.DataFrame( df.iloc[: ,  -1] , columns = ['MeanMC'])
# #yTargetTransformed = yTargetTransformed.squeeze()
# print(yTargetTransformed)

# XTestTargetTransformed = (df_test.iloc[ : , : -1 ])
# yTestTargetTransformed = pd.DataFrame(df_test.iloc[ : , -1] , columns = ['MeanMC'])
# #yTestTargetTransformed = yTestTargetTransformed.squeeze()

# XTargetTransformed = common.FeatureTransformation_Tallwood(XTargetTransformed)
# XTestTargetTransformed = common.FeatureTransformation_Tallwood(XTestTargetTransformed)

# yTargetTransformed = common.TargetLogTransformation_Tallwood(yTargetTransformed)
# yTestTargetTransformed = common.TargetLogTransformation_Tallwood(yTestTargetTransformed)

# ModelDataobj_transformed = datamodels.ModelData(XTargetTransformed , yTargetTransformed , XTestTargetTransformed , yTestTargetTransformed , '','','','','', '' )
# selectedFeatureList_transformed = common.FeatureSelectionPipeline( ModelDataobj_transformed)

# # Feature transformation and feature selection
# for selectedFeature in selectedFeatureList_transformed: 
#     for modelname, model in  dict_classifiers.items(): 
#         modelno += 1
#         selectedFeature.baseModel = modelname 
#         result = {}
#         XSelected = selectedFeature.X_train_selected     
#         XTestSelected = selectedFeature.X_test_selected
        
#         model.fit(XSelected, y)
#         ypred = model.predict(XTestSelected).reshape(-1,1)
#         yTestTargetTransformed_inverse = np.exp(yTestTargetTransformed)
#         print('TEST')
        
#         print(yTestTargetTransformed_inverse)
#         metrics = common.regressionMetrics(yTestTargetTransformed_inverse , ypred )
#         d1 = { 'Shapiro - Stats ': metrics.shapiro.statistics , 'Shapiro - pvalue ': round( metrics.shapiro.pvalue , 4 ) }
#         d2 = { 'DAgusto - Stats ': metrics.dagusto.statistics , 'DAgusto - pvalue ': round(metrics.dagusto.pvalue,4)}
#         result =  {'ModelNo' : modelno} |{'Model' : 'Global Model'} | {'Model Type' : ' Feature Transformation -> Target Transformed -> Feature Selection -> Modelling'} | vars(selectedFeature) | vars(metrics)  | d1 | d2 
#         results.append(result)
#%%   Folder and File save - Metrics file     
today = datetime.now()

if today.hour < 12:
    h = "00"
else:
    h = "12"
dirname = 'ModelMetrics/'+ (today.strftime('%Y%m%d'))

if not os.path.exists(dirname):
   os.makedirs(dirname)

 
df = pd.DataFrame(results)
writer = pd.ExcelWriter( dirname + '/' + 'ModelResults.xlsx')
df.to_excel(writer, sheet_name = 'results', index=False)
writer.save()
#%% 
































#%% - Practice code
# while featureSelMethod == 4:  
#     FeatureSelectionReport = datamodels.FeatureSelectionReport(datamodels.Models.LinearRegression.name,'' , 0, [] , '' , '')     
#     FeatureSelectionReport.noofFeatures = nooffeatures
#     df_sel  = []
#     if featureSelMethod == 1:
#         df_sel = common.modelSelection_UnivariateSelection(X , y , Xtest, nooffeatures)
#         FeatureSelectionReport.featureSelection = datamodels.FeatureSelection.UnivariateSelection.name
#         FeatureSelectionReport.selectedFeatures = list(df_sel[2])     
#         FeatureSelectionReports.append(FeatureSelectionReport)
#     elif featureSelMethod == 2:
#         df_sel = common.modelSelection_RecursiveFeatureElimination(X , y , Xtest, nooffeatures)
#         FeatureSelectionReport.featureSelection = datamodels.FeatureSelection.RecursiveFeatureElimination.name
#         FeatureSelectionReport.selectedFeatures = list(df_sel[2]) 
#         FeatureSelectionReports.append(FeatureSelectionReport)
#     elif featureSelMethod == 3:
#         df_sel = common.modelSelection_PCA(X , y , Xtest, nooffeatures)
#         FeatureSelectionReport.featureSelection = datamodels.FeatureSelection.PCA.name
#         FeatureSelectionReport.selectedFeatures = list(df_sel[2])
#         #print(type(df_sel[2]))
#         #print(df_sel[2].shape)
#         FeatureSelectionReports.append(FeatureSelectionReport)
#     elif featureSelMethod == 4:
#         df_sel = common.modelSelection_FeatureImportance(X , y , Xtest, nooffeatures)
#         FeatureSelectionReport.featureSelection = datamodels.FeatureSelection.FeatureImportance.name
#         FeatureSelectionReport.selectedFeatures = list(df_sel[2])
#         FeatureSelectionReports.append(FeatureSelectionReport)
#     featureSelMethod += 1 


# print(df_sel[0])
# print(df_sel[1])

# print(df_sel[1].shape)
#%% 