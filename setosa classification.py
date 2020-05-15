import numpy as np

def perceptron(row, weights):
	output = weights[0]
	for i in range(len(row)-1):
		output += weights[i + 1] * row[i]
	return 1.0 if output >= 0.0 else 0.0

def train_weights(train, lRate, epoch):
	weights = [0.0, 0.0, 0.0, 0.0, 0.0]
	for e in range(epoch):
		for i in range(len(train)):
				activation= perceptron(train[i],weights)

				if( (labels[i]==1) and (activation <= 0)): 
						weights[0] += lRate
						for k in range(len(train[0])):
							weights[k+1]+=lRate*train[i][k]
          
				elif( (labels[i]==0) and (activation > 0)):
						weights[0] -= lRate
						for k in range(len(train[0])):
							weights[k+1]-=lRate*train[i][k] 

	return weights

values = []
labels = []
d = {}
d['Iris-setosa\n']     = 1
d['Iris-versicolor\n'] = 0
d['Iris-virginica\n']  = 0

with open('iris.data') as f:
    lines=f.readlines()
    for line in lines:
        items=line.split(',')
        if len(items) == 5:
            inp =  [float(x) for x in items[0:4] ]
            values.append(inp)
            out = d[items[4]]
            labels.append(out)

# test predictions
#non-setosa = 1, setosa = 0
learning = 1
epochs = 1000
weights = train_weights(values, learning, epochs)
print(weights)

misclassified = 0
i =0
for row in values:
	prediction = perceptron(row, weights)
	print("Expected= ", labels[i]," Predicted= ", prediction)
	if (labels[i] != prediction):
		print("MISSCLASSIFIED")
		misclassified +=1
	i+=1

print("Misclassified: ", misclassified, " / ", len(values))
print("Accuracy: ", (len(values)-misclassified)/len(values)*100)
