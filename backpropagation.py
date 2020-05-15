
# first check if all the prerequisites are there.
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random 

inputs  = []
outputs = []

d = {}
d['Iris-setosa\n']     = [1., 0., 0.]
d['Iris-versicolor\n'] = [0., 1., 0.]
d['Iris-virginica\n']  = [0., 0., 1.]


with open('iris.data') as f:
    lines=f.readlines()
    for line in lines:
        items=line.split(',')
        if len(items) == 5:
            inp =  [float(x) for x in items[0:4] ]
            inputs.append(inp)
            out = d[items[4]]
            outputs.append(out)
            
#create a training set
ids=random.sample(range(0,len(inputs)), 100) # generate 100 random ids
train_in = []
train_out=[]
for id in ids:
    train_in.append(inputs[id])
    train_out.append(outputs[id])
train_inputs  = np.array(train_in)
train_outputs = np.array(train_out)

#create a validation set
test_input =[]
test_output=[]
test_ids = list(set(range(0,len(inputs))) - set(ids))
for test_id in test_ids:
    test_input.append(inputs[test_id])
    test_output.append(outputs[test_id])
test_inputs  = np.array(test_input)
test_outputs = np.array(test_output)


model = tf.keras.Sequential()
# an mlp with a given number of input nodes. Four input nodes, three output nodes 
nr_hidden = 25
nr_in     = 4
nr_out    = 3 
model.add(layers.Dense(nr_in,activation='relu'))
model.add(layers.Dense(nr_hidden, activation = 'sigmoid'))
model.add(layers.Dense(nr_out,activation='sigmoid'))
model.compile(optimizer=tf.optimizers.SGD(0.05),loss='mse')

model.fit(train_inputs,train_outputs,epochs=1000,batch_size=30,verbose=0)
 	
# evaluate the model using validation set
loss = model.evaluate(test_inputs, test_outputs,verbose=0)

print("MSE: ", loss)

print("Sepal Length: ")
sLength = float(input())
print("\nSepal Width: ")
sWidth = float(input())
print("\nPetal Length: ")
pLength = float(input())
print("\nPetal Width: ")
pWidth = float(input())

class_input = [[sLength, sWidth, pLength, pWidth]]
class_inputs  = np.array(class_input)

print(model.predict(class_inputs))