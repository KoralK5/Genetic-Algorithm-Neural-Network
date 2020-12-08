#modules
import random, copy as cpy, pandas as pd, numpy as np, myNN as nn

#training data
    #https://www.kaggle.com/apollo2506/facial-recognition-dataset?
    #10000 pictures, 2304 pixels per picture (48x48)
    #23037696 inputs

#layers
    #training data: 23040000 inputs, 10000 outputs
    #testing example: 2304 inputs, 6 outputs

#batches
    #[2304, 1921, 1538, 1155, 772, 389, 6]
    #0.83,0.80,0.75,0.66,0.50

#transformed training data
trainingDataFile = r'expressions24x24.csv'
trainingData = pd.read_csv(trainingDataFile)

#variables
stringInputs = list(trainingData['pixels'])
outputs = list(trainingData['emotion'])
inputs = []
layerData = [478, 382, 286, 188, 94]
batchCount = 25

for e in stringInputs:
    for sub_e in e.split(" "):
        inputs.append(int(sub_e))

inputs = np.array([np.array(row) for row in inputs])
inputs = np.array_split(np.array(inputs), 9999)

for f in range(len(outputs)):
    tmp = np.array([0, 0, 0, 0, 0, 0, 0])
    tmp[outputs[f]] = 1
    outputs[f] = cpy.deepcopy(tmp)
outputs = np.array(outputs)

layerData.append(len(outputs[0]))
print("TRAINING DATA FORMATTED")

biases = [[[random.randrange(-50, 50) / 100 for j in range(neuronCount)] for neuronCount in layerData] for x in range(batchCount)]
biases = np.array([np.array([np.array(y) for y in x]) for x in biases])
print("STARTING BIASES GENERATED")

weights = [[[[random.randrange(-50, 50) / 100 for i in range(([len(inputs[0])] + layerData)[layerCount])] for j in range(layerData[layerCount])] for layerCount in range(len(layerData))] for x in range(batchCount)]
weights = np.array([np.array([np.array([np.array(z) for z in y]) for y in x]) for x in weights])
print("STARTING WEIGHTS GENERATED")
print("")

#neural network outline
for generation in range(1000):
    print(f"GENERATION {generation+1}") 
    weights, biases = nn.generate_batch(inputs, weights, biases, outputs, "D:\\Users\\Koral Kulacoglu\\python\\expressionsWB2.csv", 5, 50, 100)

#testing
