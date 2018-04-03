from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy
import pandas
import matplotlib.pyplot as plt

# fix random seed for reproducibility
numpy.random.seed(7)

# load dataset
dataset = pandas.read_csv("Jan 2018 Speed Thick Adapt New.csv", delimiter=",", header = None)

# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:3]
Y = dataset.iloc[:,3]

print(X)
print(Y)

def new_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=3, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(1))  
    
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # fit model
    model.fit(X, Y, epochs=10, batch_size=250, verbose=1)
    
    # evaluate model
    scores = model.evaluate(X, Y)
    
    #save the model
    model.save('model.h5')
    return model, scores

model, scores = new_model()

# load trained model
#model = load_model('model.h5')

# calculate predictions
predictions = model.predict(X)

# print predictions
print(predictions)

# save predictions to a file
predFile = open(r"C:\Users\E20269\Documents\Visual Studio 2017\Projects\Adaption Tuning Model\Model Test\predictions.txt","w+")
predFile.write("\n".join(str(pred) for pred in predictions))
predFile.close()

# plot predictions
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex = 'col', sharey = 'row')
ax1.scatter(X.iloc[:,0], predictions)
ax1.set_title('Speed vs Predicted Adaption')
ax2.scatter(X.iloc[:,1], predictions)
ax2.set_title('Thickness vs Predicted Adaption')
ax3.scatter(X.iloc[:,0], Y)
ax3.set_title('Speed vs Actual Adaption')
ax4.scatter(X.iloc[:,1], Y)
ax4.set_title('Thickness vs Actual Adaption')
plt.show()