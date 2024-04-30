# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # **California Housing Prices**

# %% [markdown]
# ## **California Housing Prices Dataset**  
# ### **About the dataset**  
# The data contains information from the 1990 California census. So although it may not help you with predicting current housing prices like the Zillow Zestimate dataset, it does provide an accessible introductory dataset for teaching people about the basics of machine learning.  
#
# **Columns**  
#
#
# 1.   longitude
# 2.   latitude
# 3.   housing_median_age
# 4.   total_rooms
# 5.   total_bedrooms
# 6.   population
# 7.   households
# 8.   median_income
# 9.   median_house_value
# 10.  ocean_proximity
#
#

# %% [markdown]
# **Check and install the dependencies**

# %%
# !curl -sSL https://raw.githubusercontent.com/K14aNB/California-Housing-Prices/main/requirements.txt

# %%
# Run this command in terminal before running this notebook as .py script
# Installs dependencies from requirements.txt present in the repo
# %%capture
# !pip install -r https://raw.githubusercontent.com/K14aNB/California-Housing-Prices/main/requirements.txt

# %% [markdown]
# **Import the libraries**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import env_setup
import mlflow

# %% [markdown]
# **Environment Setup**

# %%
result_path=env_setup.setup(repo_name='California-Housing-Prices',nb_name='California-Housing-Prices')

# %% [markdown]
# **Read the data**

# %%
housing_data=pd.read_csv(os.path.join(result_path,'housing.csv'))

# %%
housing_data.head()

# %%
housing_data.info()

# %% [markdown]
# ### **Exploratory Data Analysis**

# %% [markdown]
# **Check for Missing values**

# %%
housing_data.isna().sum()

# %% [markdown]
# `total_bedrooms` column has missing_values

# %% [markdown]
# **Check for Duplicate values**

# %%
housing_data.duplicated().sum()

# %% [markdown]
# **Check the distribution of `median_house_value` column**

# %%
# Histogram of median_house_value
fig=plt.figure(figsize=(10,5))
sns.histplot(x='median_house_value',data=housing_data,kde=True)
plt.xlabel('Median House Value')
plt.title('Histogram of Median House Value')
plt.show()

# %% [markdown]
# **`median_house_value` column values have long tail distribution**

# %%
# Params for Experiment tracking
params={'random_state':42,
        'total_bedrooms_missing':0}

# %% [markdown]
# **Perform Train-Test split**

# %%
# Separate predictors and target
y=housing_data.loc[:,'median_house_value']
X=housing_data.drop('median_house_value',axis=1)

# %%
# Make a copy of X and y
X_copy=X.copy()
y_copy=y.copy()

# %%
# Split Test data from Training data for final model validation
test_len=int(X_copy.shape[0]*0.1)
n=np.arange(X_copy.shape[0])
np.random.seed(params.get('random_state'))
np.random.shuffle(n)
X_test=X_copy[:test_len]
X_copy=X_copy.drop(index=range(test_len))
y_test=y_copy[:test_len]
y_copy=y_copy.drop(index=range(test_len))

# %%
# Split Training and Validation data
X_train,X_valid,y_train,y_valid=train_test_split(X_copy,y_copy,train_size=0.8,test_size=0.2,random_state=params.get('random_state'))

# Reset the index
X_train=X_train.reset_index(drop=True)
X_valid=X_valid.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
y_valid=y_valid.reset_index(drop=True)

# %% [markdown]
# ### **Pre-processing**

# %% [markdown]
# **Check and impute missing values**

# %%
# Impute missing values in Training data
if X_train['total_bedrooms'].isna().sum()>0:
    X_train['total_bedrooms']=X_train['total_bedrooms'].replace(to_replace=np.NaN,value=params.get('total_bedrooms_missing'))

# Impute missing values in validation data
if X_valid['total_bedrooms'].isna().sum()>0:
    X_valid['total_bedrooms']=X_valid['total_bedrooms'].replace(to_replace=np.NaN,value=params.get('total_bedrooms_missing'))

# %% [markdown]
# **Encode categorical variables**

# %%
ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False,feature_name_combiner='concat')

ohe_train=pd.DataFrame(ohe.fit_transform(X_train[['ocean_proximity']]))
ohe_valid=pd.DataFrame(ohe.transform(X_valid[['ocean_proximity']]))

ohe_train.index=X_train.index
ohe_valid.index=X_valid.index

X_train=X_train.drop(columns=['ocean_proximity'],axis=1)
X_valid=X_valid.drop(columns=['ocean_proximity'],axis=1)

X_train=pd.concat([X_train,ohe_train],axis=1)
X_valid=pd.concat([X_valid,ohe_valid],axis=1)

X_train.columns=X_train.columns.astype(str)
X_valid.columns=X_train.columns.astype(str)


# %% [markdown]
# **Apply log1p transformation to `median_house_value`**

# %%
y_train=np.log1p(y_train.values)
y_valid=np.log1p(y_valid.values)
y_test=np.log1p(y_test.values)

# %% [markdown]
# ### **Linear Regression Model**

# %%
mlflow.end_run()
with mlflow.start_run():

    mlflow.log_params(params)

    lm=LinearRegression()

    lm.fit(X_train,y_train)

    y_pred=lm.predict(X_valid)

    mlflow.log_metrics({'mse':round(mean_squared_error(y_valid,y_pred),3)})

    mlflow.sklearn.log_model(sk_model=lm,artifact_path='linear-regression-model',input_example=X_train,registered_model_name='linear-regression-model')


# %% [markdown]
# **Model metrics**

# %%
# Model metrics
# Mean Squared Error
mse=round(mean_squared_error(y_valid,y_pred),3)
print(mse)

# %%
