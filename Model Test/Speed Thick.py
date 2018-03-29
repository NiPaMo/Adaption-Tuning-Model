from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt
import pandas

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = pandas.read_csv("Jan 2018 Speed Thick.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset.iloc[:,1]
Y = dataset.iloc[:,0]

print(X)
print(Y)

# create model
model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='sgd')

# Fit the model
model.fit(X, Y, epochs=5, batch_size=100, verbose=1)

# evaluate the model
scores = model.evaluate(X, Y)

# calculate predictions
predictions = model.predict(X)

# print predictions
print(predictions)

# plot predictions
plt.plot(X, predictions)
plt.show()