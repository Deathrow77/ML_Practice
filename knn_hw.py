import numpy as np
import operator

features = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
labels = ['A', 'A', 'B', 'B']
def classify(X, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diffMat = np.tile(X, (dataset_size,1)) - dataset
    sqdiffMat = diffMat**2
    sqDistances = sqdiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistances = distances.argsort()
    classCount = {}

    for i in range(k):
        votedLabel = labels[sortedDistances[i]]
        classCount[votedLabel] = classCount.get(votedLabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def image2Vect(filename):
    return_vect = np.zeros((1,1024))
    fr = open('')
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            return_vect[0,32*(i+j)] = int(linestr[j])

def handwritingClassTest():
    handwLabels = []
    trainingFileList = listdir('./optdigits.tes.txt')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        handwLabels.append(classNumStr)
        trainingMat[i,:] = image2Vect('')
    testFileList = listdir(' ')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        filestr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = image2Vect('')
        classifierResult = classify(vectorUnderTest, trainingMat, handwLabels, 3)
        print("Classifier Returned %s ; Real Result is %s" % (classifierResult, classNumStr))

