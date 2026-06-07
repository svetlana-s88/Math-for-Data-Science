# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
'''Простая линейная регрессия - обучение на синтетических данных.'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
np.random.seed(42)
X = 2.5*np.random.randn(100)+5
residual = np.random.randn(100)
Y = 20000 + X * 8000 + residual * 3000

# %%
data_ = pd.DataFrame({'Years_Experience': X, 'Salary': Y})
print(data_.head())

# %%
plt.figure(figsize = (8, 5))
sns.scatterplot(data = data_, x='Years_Experience', y ='Salary', color= 'blue')
plt.title('Зависимость зарплаты от опыта работы')
plt.xlabel('Опыт работы(лет)')
plt.ylabel('Зарплата(в долл)')
plt.grid(True)
plt.show()


# %%
X_matrix = data_[['Years_Experience']]
y_vector = data_['Salary']
X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_vector, test_size=0.2, random_state = 42)

# %%
model = LinearRegression()
model.fit(X_train, y_train)
print(model.intercept_)
print(model.coef_[0])

# %%
from sklearn.metrics import mean_absolute_error


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(mse) 
print(mae) 
print(r2)

# %%
plt.figure(figsize = (8,5))
plt.scatter(X_test, y_test,color='blue', label = 'Реальные данные')
plt.plot(X_test, y_pred, color = 'red', linewidth = 2, label = 'Линия регрессии')
plt.title('Итоговая модель линейной регрессии')
plt.xlabel('Опыт работы (лет)')
plt.ylabel('Зарплата ($)')
plt.legend()
plt.grid(True)
plt.show()
