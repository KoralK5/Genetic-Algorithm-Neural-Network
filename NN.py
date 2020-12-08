import numpy as np, random, copy, ast
from csv import reader

def cost(a, b):
    return (a - b) ** 2

def sigmoid(x):
    e = 2.7182818284590452353602874713527
    return 1 / (1 + e ** -x)

def layer(inputs, weights, bias):
    return np.array(sigmoid(sum((inputs*weights).transpose()) + bias))

def network(inputs, weights, bias):
    ip = copy.deepcopy(inputs)

    if path == 'u':
        for row in range(len(bias)):
            ip = layer(ip, weights[row], bias[row])

        size = []
        for row in range(len(outputs)):
            size += [0]
        print(cost(ip, size))

    elif path == 't':
        for row in range(len(bias)):
            ip = layer(ip, weights[row], bias[row])
        return ip

def test(inputs, outputs, weights, bias):
    netOutputs = []
    for row in range(len(inputs)):
        netOutputs.append(network(inputs[row], weights, bias))
    netOutputs = np.array(netOutputs)
    costs = cost(outputs, netOutputs)
    print(netOutputs,'\n',costs)
    elementAmount = 1
    for row in np.shape(outputs):
        elementAmount *= row
    print(f"Error: {sum(sum(costs)) / elementAmount}")
    return sum(sum(costs)) / elementAmount

def randomize(weights, bias):
    newWeights = copy.deepcopy(weights)
    newBias = copy.deepcopy(bias)
    for row in range(changeAmount):
        a = random.randrange(0, len(weights))
        b = random.randrange(0, len(weights[a]))
        c = random.randrange(0, len(weights[a][b]))
        newWeights[a][b][c] += random.randrange(-randomAmount, randomAmount)/100
    for row in range(changeAmount):
        a = random.randrange(0, len(bias))
        b = random.randrange(0, len(bias[a]))
        newBias[a][b] += random.randrange(-randomAmount, randomAmount)/100
    return newWeights, newBias

def batch(inputs, outputs, weights, bias):
    scores = []
    netCount = len(weights)
    for row in range(netCount):
        print("")
        print(f"Network {row+1}")
        scores.append(test(inputs, outputs, weights[row], bias[row]))
    sortedScores = copy.deepcopy(scores)
    sortedScores.sort()
    save(weights, bias, scores, sortedScores)
    print(f"\nLowest Error: {sortedScores[0]}\n")
    newWeights = [weights[scores.index(sortedScores[0])] for row in range(netCount)]
    newBias = [bias[scores.index(sortedScores[0])] for row in range(netCount)]
    for row in range(netCount - 1):
        newWeights[row], newBias[row] = randomize(newWeights[row], newBias[row])
    return np.array(newWeights), np.array(newBias)

def save(weights, bias, scores, location):
    strWeights = ""
    strBias = ""
    file = open(store, 'w+')
    storedBias = list([list(x) for x in bias[scores.index(location[0])]])
    storedWeights = list([list([list(y)for y in x]) for x in weights[scores.index(location[0])]])
    for a in range(len(storedBias)):
        for b in range(len(storedBias[a])):
            strBias += f"{storedBias[a][b]},"

    for a in range(len(storedWeights)):
        for b in range(len(storedWeights[a])):
            for c in range(len(storedWeights[a][b])):
                strWeights += f"{storedWeights[a][b][c]},"
    
    save = f"{str(strWeights)}\n{str(strBias)}\n{str(location[0])}"
    file.write(save)
    file.close()

def initialize():
    global inputs, outputs, weights, bias

    inputs = [[1,1,0],[0,0,1],[0,1,0]]
    outputs = [[1,0],[0,1],[0,1]]
    inputs = np.array([np.array(row) for row in inputs])
    outputs = np.array([np.array(row) for row in outputs])

    bias = [[[random.randrange(-50, 50) / 100 for j in range(neuronCount)] for neuronCount in layerData] for x in range(batchCount)]
    bias = np.array([np.array([np.array(y) for y in x]) for x in bias])

    weights = [[[[random.randrange(-50, 50) / 100 for i in range(([len(inputs[0])] + layerData)[layerCount])] for j in range(layerData[layerCount])] for layerCount in range(len(layerData))] for x in range(batchCount)]
    weights = np.array([np.array([np.array([np.array(z) for z in y]) for y in x]) for x in weights])

def use():
    inputs = np.array(ast.literal_eval(input('Input: ')))
    
    strAll = []
    with open('D:\\Users\\Koral Kulacoglu\\Coding\\python\\testScores.csv', 'r') as read:
        weights = []
        bias = []
        csvReader = reader(read)
        strAll += csvReader
        
        for row in range(len(strAll[0][:-1])):
            weights += [float(strAll[0][:-1][row])]

        for row in range(len(strAll[1][:-1])):
            bias += [float(strAll[1][:-1][row])]

        score = float(strAll[2][0])

    it = iter(bias)
    bias = [[next(it) for _ in range(n)] for n in layerData]

    layerDataW = []
    for row in layerData:
        layerDataW += [[row]*row]

    it = iter(weights)
    weights = []
    for row in layerDataW:
        weights += [[[next(it) for _ in range(n)] for n in row]]

    bias = np.array([np.array(y) for y in bias])
    weights = np.array([np.array(y) for y in weights])

    outputs = network(inputs, weights, bias)

path = input('Train(t) or Use(u): ')

if path == 'u':
    layerData = [3, 3, 2, 1, 1]
    batchCount = 25
    use()

elif path == 't':
    layerData = [3, 3, 2, 1, 1]
    batchCount = 25
    changeAmount = 5
    randomAmount = 50
    genlen = 1
    store = "testScores.csv"
    initialize()

    for row in range(genlen):
        print(f"Generation {row+1}")
        weights, bias = batch(inputs, outputs, weights, bias)
