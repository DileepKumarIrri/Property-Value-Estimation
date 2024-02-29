import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv('kc_house_data.csv')
y=data["price"].values
p=data.drop(['id','price','date','sqft_lot','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','sqft_living15','sqft_lot15'],axis=1)
X=p.values



