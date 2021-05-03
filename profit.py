import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Startups.csv")
print(data.head())

print(data.describe())

sns.heatmap(data.corr(), annot=True)
plt.show()
plt.savefig('heatmap.png')

#As this task is based on the problem of regression so I will be using the Linear regression algorithm to train the profit prediction model.
#So let’s prepare the data so that we can fit it into the model:

x = data[["R&D Spend", "Administration", "Marketing Spend"]]
y = data["Profit"]

x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

#Now let’s train a linear regression model on this data and have a look at the predicted values:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted Profit": ypred.flatten()})
print(data.head())
data.to_csv('Predection.csv')

'''
Summary
So this is how we can predict the profit of a company for a 
particular period by using machine learning algorithms.
Such tasks can help a company to set a target that can be achieved.
'''