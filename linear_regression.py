import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("data/energydata_complete.csv")
data = data.select_dtypes(include=['number']).dropna()

X = data.drop("Appliances", axis=1)
y = data["Appliances"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Linear Regression")
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
