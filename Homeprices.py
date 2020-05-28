import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
df = pd.read_csv("homeprices.csv")
plt.scatter(df.area,df.price,color='red')
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
X_area=[[33000]]
print(reg.predict(X_area))
