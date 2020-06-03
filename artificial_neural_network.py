# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

import tensorflow as tf
tf.__version__

import seaborn as sn


# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Paddy.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Paddy.season" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:,3])
print(X)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#############---------------------      MLR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#its pred
y_pred_1 = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_1.reshape(len(y_pred_1),1), y_test.reshape(len(y_test),1)),1))


#visualisation
#i=1
#plt.Line2D(xdata, ydata, kwargs)(,y_test,color='red')
#plt.scatter(,regressor.predict(X),color='green')


###########------------------------   SVR

sc_y =StandardScaler()
y_train_sc = sc_y.fit_transform(y_train.reshape(-1,1))



from sklearn.svm import SVR
regressor_2=SVR(kernel='rbf')
regressor_2.fit(X_train,y_train_sc)

#prediction
y_pred_2=sc_y.inverse_transform(regressor_2.predict(X_test))

#visualisation


###########------------------------    RANDOM FORREST
from sklearn.ensemble import RandomForestRegressor
regressor_3=RandomForestRegressor(max_features=5,n_estimators=500)
regressor_3.fit(X,y)

#prediction
y_pred_3=regressor.predict(X_test)


plt.hist(y_pred_3,bins=100)


###########------------------------    ANN
# Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()


# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu',input_shape=(6,)))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

#adding third layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

#adding fourth layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

#adding fifth layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))


#adding fifth layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))


#adding fifth layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))

# Part 3 - Training the ANN

# Compiling the ANN
#optimizer = tf.keras.optimizers.RMSprop(0.001)
ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss = 'mse', metrics = ['mae'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 4 - Making the predictions and evaluating the model



# Predicting the Test set results



print("[INFO] predicting yields...")
y_pred_4 = ann.predict(X_test)

#print(y_pred_4)


##################################################for MLR


# compute the difference between the *predicted* and the
# *actual*, then compute the percentage difference and
# the absolute percentage difference
diff_1 = y_pred_1.flatten() - y_test
percentDiff_1 = (diff_1 / y_test) * 100
absPercentDiff_1 = np.abs(percentDiff_1)

# compute the mean and standard deviation of the absolute percentage
# difference
mean_1 = np.mean(absPercentDiff_1)
std_1 = np.std(absPercentDiff_1)


# finally, show some statistics on our model

plt.hist(y_test,bins=200,color='green')
plt.hist(y_pred_1,bins=200,color='red')
print("[INFO] mean: {:.2f}%, std: {:.2f}%    for  MLR".format(mean_1, std_1))


#########################################for SVR

diff_2 = y_pred_2.flatten() - y_test
percentDiff_2 = (diff_2 / y_test) * 100
absPercentDiff_2 = np.abs(percentDiff_2)

# compute the mean and standard deviation of the absolute percentage
# difference
mean_2 = np.mean(absPercentDiff_2)
std_2 = np.std(absPercentDiff_2)


# finally, show some statistics on our model

plt.hist(y_test,bins=200,color='green')
plt.hist(y_pred_2,bins=200,color='blue')
print("[INFO] mean: {:.2f}%, std: {:.2f}%    for  SVR".format(mean_2, std_2))


########################################for Random Forest
diff_3 = y_pred_3.flatten() - y_test
percentDiff_3 = (diff_3 / y_test) * 100
absPercentDiff_3 = np.abs(percentDiff_3)

# compute the mean and standard deviation of the absolute percentage
# difference
mean_3 = np.mean(absPercentDiff_3)
std_3 = np.std(absPercentDiff_3)


# finally, show some statistics on our model

plt.hist(y_test,bins=200,color='green')
plt.hist(y_pred_3,bins=200,color='yellow')
print("[INFO] mean: {:.2f}%, std: {:.2f}%    for  Random Forest".format(mean_3, std_3))

################################################ for ANN

diff_4 = y_pred_4.flatten() - y_test
percentDiff_4 = (diff_4 / y_test) * 100
absPercentDiff_4 = np.abs(percentDiff_4)

# compute the mean and standard deviation of the absolute percentage
# difference
mean_4 = np.mean(absPercentDiff_4)
std_4 = np.std(absPercentDiff_4)


# finally, show some statistics on our model

#plt.hist(y_test,bins=200,color='green')
plt.hist(y_pred_4,bins=200,color='cyan')
print("[INFO] mean: {:.2f}%, std: {:.2f}%    for  ANN".format(mean_4, std_4))



#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))