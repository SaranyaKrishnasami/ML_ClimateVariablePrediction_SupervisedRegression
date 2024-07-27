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
from sklearn.preprocessing import PowerTransformer
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

power =  PowerTransformer(method='yeo-johnson', standardize=True)
power.fit(np.array(yTrain).reshape(-1,1))
Y = power.transform(np.array(yTrain).reshape(-1,1))

# Feature scaling - Standard scalar  
scaler = StandardScaler()
scaler.fit(XTrain_Transformed)
XTrain_ScaledTransformed = pd.DataFrame(scaler.transform(XTrain_Transformed) , columns = XTrain_Transformed.columns)
XTest_ScaledTransformed  = pd.DataFrame(scaler.transform(XTest_Transformed) , columns = XTest_Transformed.columns)
#common.ViewFeatureDistribution(X , XTrain_ScaledTransformed , 'Before and After Scaling and Transformation' ,yTrain)
 

# Feature Selection and Modelling
#%%
# RUN 1 Baseline Feature selection
ModelDataobj = datamodels.ModelData(XTrain_ScaledTransformed , yTrain , XTest_ScaledTransformed , yTest , '','','','','', '', '' )
 
results = [] 
result = {}

import time

from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Selecting Lasso via an information criterion
start_time = time.time()
lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic")).fit(XTrain_ScaledTransformed, Y)
fit_time = time.time() - start_time

results = pd.DataFrame(
    {
        "alphas": lasso_lars_ic[-1].alphas_,
        "AIC criterion": lasso_lars_ic[-1].criterion_,
    }
).set_index("alphas")
alpha_aic = lasso_lars_ic[-1].alpha_


lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(XTrain_ScaledTransformed, Y)
results["BIC criterion"] = lasso_lars_ic[-1].criterion_
alpha_bic = lasso_lars_ic[-1].alpha_   
 

def highlight_min(x):
    x_min = x.min()
    return ["font-weight: bold" if v == x_min else "" for v in x]


results.style.apply(highlight_min)


ax = results.plot()
ax.vlines(
    alpha_aic,
    results["AIC criterion"].min(),
    results["AIC criterion"].max(),
    label="alpha: AIC estimate",
    linestyles="--",
    color="tab:blue",
)
ax.vlines(
    alpha_bic,
    results["BIC criterion"].min(),
    results["BIC criterion"].max(),
    label="alpha: BIC estimate",
    linestyle="--",
    color="tab:orange",
)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("criterion")
ax.set_xscale("log")
ax.legend()
_ = ax.set_title(
    f"Information-criterion for model selection (training time {fit_time:.2f}s)"
)
plt.show()

alphaAIC = alpha_aic # results.loc[results['AIC criterion'] ==  (results["AIC criterion"].min())].iloc[0].name
minAICCriterion = results.loc[results['AIC criterion'] ==  (results["AIC criterion"].min())].iloc[0]['AIC criterion']
alphaBIC = alpha_bic #results.loc[results['BIC criterion'] ==  (results["BIC criterion"].min())].iloc[0].name
minBICCriterion = results.loc[results['BIC criterion'] ==  (results["BIC criterion"].min())].iloc[0]["BIC criterion"]


alphaSelectedwithdelta = []

UpbAIC = minAICCriterion + 2
lwbAIC = minAICCriterion - 2

UpbBIC = minBICCriterion + 2
lwbBIC = minBICCriterion - 2

l1 = results.loc[results['AIC criterion'] >= lwbAIC ]
l1 =  l1 <= UpbAIC 
l1pd = pd.DataFrame(l1['AIC criterion'])
deltaAlphasAIC = l1pd[l1pd['AIC criterion'] == True].index
deltaAlphasAIC = list(deltaAlphasAIC)
deltaAlphasAICName = []

for i in range(len(deltaAlphasAIC) + 1):
    deltaAlphasAICName.append('AIC'+ str(i+1))


l2 = results.loc[results['BIC criterion'] >= lwbBIC ]
l2 =  l2 <= UpbBIC 
l2pd = pd.DataFrame(l2['BIC criterion'])
deltaAlphasBIC = l2pd[l2pd['BIC criterion'] == True].index
deltaAlphasBIC = list(deltaAlphasBIC)
deltaAlphasBICName = []

for i in range(len(deltaAlphasBIC) + 1):
    deltaAlphasBICName.append('BIC'+ str(i+1))
    
dictBIC=zip(deltaAlphasBICName , deltaAlphasBIC)
dictBIC=dict(dictBIC)

dictAIC=zip(deltaAlphasAICName , deltaAlphasAIC)
dictAIC=dict(dictAIC)


#************************************************************************************************************************
# Selecting Lasso via cross-validation
# Selection via coordinate descent

from sklearn.linear_model import LassoCV

start_time = time.time()
model = make_pipeline(StandardScaler(), LassoCV(cv=20)).fit(XTrain_ScaledTransformed, Y)
fit_time = time.time() - start_time
import matplotlib.pyplot as plt

ymin, ymax = -5, 25
lasso = model[-1]
plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
plt.plot(
    lasso.alphas_,
    lasso.mse_path_.mean(axis=-1),
    color="black",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

plt.ylim(ymin, ymax)
plt.xlabel(r"$\alpha$")
plt.ylabel("Mean square error")
plt.legend()
_ = plt.title(
    f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
)
alphaCV = lasso.alpha_
plt.show()
#************************************************************************************************************************
# Lasso via least angle regression

from sklearn.linear_model import LassoLarsCV

start_time = time.time()
model = make_pipeline(StandardScaler(), LassoLarsCV(cv=20)).fit(XTrain_ScaledTransformed, Y)
fit_time = time.time() - start_time
lasso = model[-1]
plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ":")
plt.semilogx(
    lasso.cv_alphas_,
    lasso.mse_path_.mean(axis=-1),
    color="black",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha CV")

plt.ylim(ymin, ymax)
plt.xlabel(r"$\alpha$")
plt.ylabel("Mean square error")
plt.legend()
_ = plt.title(f"Mean square error on each fold: Lars (train time: {fit_time:.2f}s)")

plt.show()
alphaLarsCV = lasso.alpha_
#%%

# Modeling with Elastic Net - Use alpha with AIC  , BIC Criteria 
results = []
modelno = 0
alphas = [ {'alphaAIC':alphaAIC} , { 'alphaBIC' : alphaBIC} , { 'alphaCV' : alphaCV }  , { 'alphaLarsCV' : alphaLarsCV } , {'AlphaPrevModelling' : 0.0011 } ]
for item in dictBIC.items():    
    alphas.append({item[0] : item[1]})
for item in dictAIC.items():    
    alphas.append({item[0] : item[1]}) 
l1Ratios = [0,1]
for dict_item in alphas:
    for key in dict_item:
        for l1 in l1Ratios: # α ∈ [0, 1] is called the mixing parameter . Lasso and ridge are special cases, respectively for α = 1 and α = 0.
            modelno += 1
            en = linear_model.ElasticNet(alpha = dict_item[key] , l1_ratio = l1 )
            en.fit(XTrain_ScaledTransformed, Y)
            Y_test_Predict = en.predict(XTest_ScaledTransformed)            
            
            Y_test_Predict = power.inverse_transform(Y_test_Predict.reshape(-1,1))
        
            TestDatametrics =  common.regressionMetrics(yTest, Y_test_Predict.flatten())   
            
            
            plt.scatter( x = Y_test_Predict.flatten() , y = yTest)
            plt.xlabel('Predicted Value Mean MC (Elastic net Regression)')
            plt.ylabel('Actual Mean MC')
            plt.title('Elastic net - Best model '  )
            plt.show() 
            
            residual = yTest - Y_test_Predict.flatten()
            sns.regplot(y = residual, x = Y_test_Predict.flatten())
            plt.xlabel('Predicted Value  (Elastic net)')
            plt.ylabel('Residual')
            plt.title('Elastic net - Best model '  )
            plt.show()
         
            result = {'ModelNo' : modelno} | {'Model' : 'Global Model'} | {'Model Type' : 'Baseline with Feature transformation , scaling and alpha value selection'} | {'Alpha Value' : dict_item}| {'RMSE' : TestDatametrics.RMSE }  | vars(TestDatametrics)    
            results.append(result)
                
    
    
# BEST MODEL - Lasso - Best Alpha with AIC 
# Test the model on City wise data        
 
FeatureSelectionReports = []
# XTrain, yTrain, XTest, yTest , hyp , mtricsTrain,  mtricsTest ,bestParams, version, modelName = ''):
cityresults = []

bestModel = linear_model.ElasticNet(alpha = alphaBIC , l1_ratio = 0 )
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
    plt.xlabel('Predicted Value Mean MC (Elastic net Regression)')
    plt.ylabel('Actual Mean MC')
    plt.title('Elastic net - Best model ' + city)
    plt.show() 
    
    residual = s2 - s3
    sns.regplot(y = residual, x = s3)
    plt.xlabel('Predicted Value  (Elastic net)')
    plt.ylabel('Residual')
    plt.title('Elastic net - Best model ' + city)
    plt.show()
    
    InformationCriterion = common.logLiklihood(Yactual_test_city  , ypred_test_city  , Xtest_city.shape[1] )
    cityrank = ranking.make_scatter_plot_res_pred_runLevel(city_df, city, 'tall wood','F0-F7', '' , dirname+ '/CityWiseRanking_ElasticNet/')
    result = ''
    result = {'ModelNo' : modelno} | {'Model' : 'Global Model - Best Model'} | {'ModelCityTest' : city } | {'CityRank': cityrank} | {'Info Criterion' : InformationCriterion } | {'Model Type' : 'Baseline with Feature transformation , scaling and alpha value selection'} | {'Alpha Value' : alphaBIC} | vars(metricsCitywise) 
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
writer = pd.ExcelWriter( dirname + '/' + 'ModelResults_ElasticNet.xlsx')
w1.to_excel(writer, sheet_name = 'ElasticNet', index=False)
w2.to_excel(writer, sheet_name = 'ElasticNet-CityWise', index=False)
writer.save()

# #%% 
































 