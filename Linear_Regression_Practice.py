import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
#calculate r2 and mean square error
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("Salary_Data.csv")
print(df.head())
print("Shape of the data : ",df.shape)

df_corr = df.corr()
sns.heatmap(df_corr)
plt.show()

#Visualization between calories and duration
sns.lmplot(x="YearsExperience", y="Salary", data=df)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("YearsExperience vs Salary")
plt.show()


print("Information :", df.info())

#Pair plot
sns.pairplot(data=df, x_vars="YearsExperience", y_vars="Salary")
plt.show()

#Create x and y
X = df["YearsExperience"]
X.head()

y = df["Salary"]
y.head()

X = df.iloc[:, 0:1]
y = df.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

# X_train = X_train[:,np.newaxis]
# X_test = X_test[:,np.newaxis]



#Fitting the model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

plt.scatter(df['YearsExperience'], df['Salary'])
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel("X_train Data")
plt.ylabel("Salary")
plt.show()

plt.scatter(df["YearsExperience"], df["Salary"])
plt.plot(X_test, lr.predict(X_test), color='red')
plt.xlabel("X_test data")
plt.ylabel("Salary")
plt.show()

mse = mean_squared_error(y_test, y_pred)

#R square
r2 = r2_score(y_test, y_pred)

print("Mean square error :", mse)
print("R2 square :", r2)

#Intercept and coefficient
print("Intercept : ", lr.intercept_)
print("Coefficient :", lr.coef_)