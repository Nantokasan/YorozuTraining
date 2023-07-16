from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston

X, y = load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('seaborn-darkgrid')

X_train_single = X_train[:,5].reshape(-1,1)
X_test_single = X_test[:,5].reshape(-1,1)

lm_single = linear_model.LinearRegression()
lm_single.fit(X_train_single, y_train)

y_pred_train = lm_single.predict(X_train_single)

plt.xlabel('RM')
plt.ylabel('MEDV')
plt.scatter(X_train_single,y_train)
plt.plot(X_train_single, y_pred_train, color = 'red',linewidth=2)
plt.show()

print(f'intercept:{lm_single.intercept_:.2f}')
print(f'coef:{lm_single.coef_[0]:.2f}')