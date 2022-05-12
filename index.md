# Multiple Boosting and LightGBM
DATA 410 - Advanced Applied Machine Learning - Final Project
### By Anne Louise Seekford and Lisa Fukutoku
#### 05.11.2022

Created and implemented our own multiple boosting algorithm to combinations of different regressors on the "Major Social Media Stock Prices" dataset. Additional application of LightGBM algorithm. 


## Overview
For the final project, we created and applied a multiple boosting algorithm of our creation to five regression techniques on the Major Social Media Stock Price Dataset. This dataset is multivariate and our aim is to answer questions relevant to the peak and trough of buying and selling stocks. 


## Data
The Major Social Media Stock Price dataset (Kanawattanachai, 2022), retrieved from Kaggle, consists of stock price details from five dominant social media platforms: Facebook, Twitter, Snapchat, Etsy, and Pinterest. The stock symbol, adjusted-close price, open and close price, high and low price, trading volume and date of the stock are described for each platform, per period. We will delve into exploring the highest and lowest price at which a stock traded during a period - as these numbers indicate the financial health and stability of the company. We plan to cross reference our results with current events to see if a real-life explanation for the results exists.  

Snippet of Dataset:  

<img width="470" alt="Screen Shot 2022-05-11 at 2 12 21 PM" src="https://user-images.githubusercontent.com/71660299/167917862-8c75abc7-d37c-45ab-af44-ebbaf37ac273.png">    


## Analysis Methods and Preprocessing
For the data preprocessing, all data with missing values were removed. Then, we set y as the trading volume column with our feature variable, x, as the high and low stock price. Due to the excessive runtime and massive dataset, we had to subset the features into ninths, each containing roughly 1,000 rows each. We then created a function to call for each x0, x1, x2, …, x8 to make our code cleaner and easier to run and read. As the analytical/machine learning methods, we first created and applied a multiple boosting algorithm of our creation to two regression techniques. By utilizing Locally Weighted Linear Regression, Boosted LOWESS, Random Forest, XGBoost, and our own “super booster” regressors, we were able to compare MSE results to LightGBM. 


## Multiple Boosting Algorithm

Import libraries and create functions:

```python
# import libraries
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import xgboost as xgb

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
  
# Lowess regression model
def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)
    if len(X.shape)==1:
      X = X.reshape(-1,1)
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    # looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        # A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        # theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew)      # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output

# Booster for the multiple boosting
def booster(X, y, xnew, kern, tau, model_boosting, nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new
  
# Boosted lowess regression model
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  # model = DecisionTreeRegressor(max_depth=2, random_state=123)
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  # model = model_xgb
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output
```


Our multiple booster was then ran in a K-Fold Cross Validation Loop, along with other regressors.  

```python
def boosting_function(X,y, subset_num=''):

  mse_sboost = []
  mse_blwr = []
  mse_xgb = []
  mse_nn =[]
  mse_rf = []
  mse_lwr = []

  for i in range(2):
    kf = KFold(n_splits=10,shuffle=True,random_state=i)
    # this is the Cross-Validation Loop
    for idxtrain, idxtest in kf.split(X):
      xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      ytest = y[idxtest]
      xtest = X[idxtest]
      xtrain = scale.fit_transform(xtrain)
      xtest = scale.transform(xtest)

      dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
      dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)

      # LOWESS
      yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
      mse_lwr.append(mse(ytest,yhat_lwr))


      # BOOSTED LOWESS
      #yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
      yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
      mse_blwr.append(mse(ytest,yhat_blwr))

      # RANDOM FOREST
      model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
      model_rf.fit(xtrain,ytrain)
      yhat_rf = model_rf.predict(xtest)
      mse_rf.append(mse(ytest,yhat_rf))

      #XGBOOST
      model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
      model_xgb.fit(xtrain,ytrain)
      yhat_xgb = model_xgb.predict(xtest)
      mse_xgb.append(mse(ytest,yhat_xgb))

      # SUPER BOOSTER (Multiple boosting function that we created)
      super_yhat = booster(xtrain, ytrain, xtest, Tricubic, 1, model_boosting, 1)
      mse_sboost.append(mse(ytest, super_yhat))


  print('Here are the results for the ' + subset_num + ' stock prices:')    
  print('The Cross-validated MSE for LWR is : '+str(np.mean(mse_lwr)))
  print('The Cross-validated MSE for Boosted Locally Weighted Regression is : '+str(np.mean(mse_blwr)))
  print('The Cross-validated MSE for Random Forest is : '+str(np.mean(mse_rf)))
  print('The Cross-validated MSE for XGBoost is : '+str(np.mean(mse_xgb)))
  # print('The Cross-validated MSE for NN is : '+str(np.mean(mse_nn)))
  print('The Cross-validated Mean Squared Error for our own multiple booster is : '+str(np.mean(mse_sboost)))
```

### Results:
Due to the size of the dataset, we had to subset.
  
<img width="417" alt="Screen Shot 2022-05-11 at 2 29 08 PM" src="https://user-images.githubusercontent.com/71660299/167920646-3d9bed27-d4e3-4d28-9f72-4183b7f49e9e.png">  
  
**After taking the average of all nine subsets' MSE, we came to these results:**   

Average Cross-validated MSE for LOWESS is:  396960632219909.25   
Average Cross-validated MSE for Boosted LOWESS is:  368504086517037.25   
Average Cross-validated MSE for RF is:  429377062586663.3   
Average Cross-validated MSE for XGBoost is:  451751115850667.44   
Average Cross-validated MSE for our own multiple booster is:  387605191285508.25   



## LightGBM

LightGBM is a gradient-boosting framework that utilizes a vertically-based tree structure learning algorithm. Based on decision tree algorithm, it is used for ranking, regression, classification, and many other machine learning tasks. Due to the vertical flow, LightGBM can significantly outperform XGBoost and other regressors in terms of both computational speed and memory consumption (Guolin et al.). Therefore, when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. With that being said however, LightGBM is sensitive to overfitting, and is recommended to be used on larger datasets. There is a variety of parameters to include to improve results, a few examples including (Mandot, 2018):
  - learning_rate: determines the impact each tree has on the final outcome
  - num_leaves: number of leaves in a full tree (31 is default)
  - num_boost_round: number of boosting iterations
  - boosting: defines the type of algorithm running (can choose traditional Gradient Boosting Decision Tree (gbdt), Random Forest (rf), etc.)
  

Leaf-wise tree growth in LightGBM:  

<img width="838" alt="Screen Shot 2022-05-11 at 2 25 01 PM" src="https://user-images.githubusercontent.com/71660299/167919974-94b6d4dd-e80d-49ac-b2db-6ab9eefb719c.png">    
  
LightGBM is called “Light” because of its computation power and giving results faster. It takes less memory to run and is able to deal with large volumes of data having more than 10,000+ rows, especially when one needs to achieve a high accuracy of results. Due to LightGBM's extreme computational power, we were able to run the dataset complete, without subsetting like we had done to the prior algorithms.  
  

```python
mse_lgb = []

for i in range(2):
  # k-fold cross-validation for a even lower bias predictive modeling
  kf = KFold(n_splits=10,shuffle=True,random_state=i)

  # the main Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)

    model_lgb = lgb.LGBMRegressor()
    model_lgb.fit(xtrain,ytrain)
    yhat_lgb = model_lgb.predict(xtest)
    mse_lgb.append(mse(ytest,yhat_lgb))
print('The Cross-validated MSE for ' + subset_num +' LightGBM is : '+str(np.mean(mse_lgb)))
```

### Results:
Average Cross-validated MSE for LightGBM is: 402199053473542.5
  

## Conclusion

Since we aim to minimize the crossvalidated mean square error (MSE) of the model for the better results, we conclude that Boosted LOWESS achieved the best result in this analysis. The order of the model from the best to worst is Boosted LOWESS, our own multiple booster, LOWESS, LightGBM, Random Forest, and Extreme Gradient Boosting (XGBoost). We expected for lightGBM to demonstrate the best MSE result in this project, but lightGBM did not achieve a superior outcome.



## References

Guolin, et al. (n.d.). LightGBM: A highly efficient gradient boosting decision tree. Retrieved March 10, 2022, from [https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

Hartman, D. (2017, February 7). How do stock prices indicate financial health? Finance. Retrieved April 15, 2022, from [https://finance.zacks.com/stock-prices-indicate-financial-health-9096.html](https://finance.zacks.com/stock-prices-indicate-financial-health-9096.html)

Mandot, P. (2018, December 1). What is LIGHTGBM, how to implement it? how to fine tune the parameters? Medium. Retrieved March 10, 2022, from [https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)

Prasert Kanawattanachai. (April 2022). Major social media stock prices 2012-2022, Version 1. Retrieved April 14, 2022 from [https://www.kaggle.com/datasets/prasertk/major-social-media-stock-prices-20122022](https://www.kaggle.com/datasets/prasertk/major-social-media-stock-prices-20122022)
 


#### Support or Contact
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
