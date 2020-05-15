
import numpy as np

# Make a prediction with weights
def setosaPerceptron(row):
	i =0
	weights = [1.0, 1.0999999999999996, 4.6, -6.800000000000001, -3.0999999999999996]
	output = weights[0]
	output += (weights[1] * row[0]) + (weights[2] * row[1]) + (weights[3] * row[2]) + (weights[4] * row[3])
	return 1.0 if output >= 0.0 else 0.0


# Make a prediction with weights
def versisolorPerceptron(row):
	i = 0
	weights = [-49.0, 31.550000000006328, -28.799999999997357, -3.9999999999999667, -72.79999999999333]
	output = weights[0]
	output += (weights[1] * row[0]) + (weights[2] * row[1]) + (weights[3] * row[2]) + (weights[4] * row[3])
	return 1.0 if output >= 0.0 else 0.0

# Make a prediction with weights
def virginicaPerceptron(row):
	i = 0
	weights = [-180.0, -99.30000000000278, -125.90000000000005, 155.09999999999883, 246.39999999999864]
	output = weights[0]
	output += (weights[1] * row[0]) + (weights[2] * row[1]) + (weights[3] * row[2]) + (weights[4] * row[3])
	return 1.0 if output >= 0.0 else 0.0

#use weights from question 3 perceptrons

d = {}
d['Iris-setosa\n']     = [0., 1., 0.]
d['Iris-versicolor\n'] = [1., 0., 0.]
d['Iris-virginica\n']  = [0., 0., 1.]

inputs  = []
outputs = []

# test predictions
#non-setosa = 1, setosa = 0

with open('iris.data') as f:
    lines=f.readlines()
    for line in lines:
        items=line.split(',')
        if len(items) == 5:
            inp =  [float(x) for x in items[0:4] ]
            inputs.append(inp)
            out = d[items[4]]
            outputs.append(out)

classified =0
i =0
pred = []
for row in inputs:
	versicolor = versisolorPerceptron(row)
	setosa = setosaPerceptron(row)
	virginica = virginicaPerceptron(row)

	prediction = [versicolor, setosa, virginica]
	pred.append(prediction)
	#print("Expected= ", outputs[i]," Predicted= ", prediction)
	if (outputs[i] == prediction):
		classified +=1
	i+=1

print("----Test Set---")
print("Correctly classified: ", classified, " / ", len(values))
print("Accuracy: ", (classified)/len(values)*100)

print("\nManual Input")
print("Sepal Length: ")
sLength = float(input())
print("\nSepal Width: ")
sWidth = float(input())
print("\nPetal Length: ")
pLength = float(input())
print("\nPetal Width: ")
pWidth = float(input())

inputs = [[sLength, sWidth, pLength, pWidth]]

misclassified =0
i =0
for row in inputs:
	versicolor = versisolorPerceptron(row)
	setosa = setosaPerceptron(row)
	virginica = virginicaPerceptron(row)

prediction = [versicolor, setosa, virginica]

print("Predicted= ", prediction)
