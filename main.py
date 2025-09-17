import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(9)
n=20
h=np.random.randint(150,190,n)
w=50+0.9*h+np.random.randint(-10,10,n)
df=pd.DataFrame({"h":h,"w":w})
print(df.head())

X=df[["h"]]
y=df["w"]
reg=LinearRegression().fit(X,y)
print(reg.coef_,reg.intercept_)
pred=reg.predict(X)

plt.scatter(h,w,c="b")
plt.plot(h,pred,"r")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

val=int(input("Enter height in cm: "))
ans=reg.predict([[val]])
print("Predicted weight:",ans[0])