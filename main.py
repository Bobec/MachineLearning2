import numpy as np
import matplotlib.pyplot as plt
import math

f = open("mfeat-pix.txt", "r")
N = 1000
DIM = 240

# Simple function that plots the digit recived as list of numbers
def plotDigit ( vectDigit ):
	array = []
	while vectDigit != []:
		array.append(vectDigit[:15])
		vectDigit = vectDigit[15:]	
	plt.imshow(array, cmap = 'gray', vmin = 0, vmax = 6)
	plt.show()

# Function that returns the center datapoind given a nested list of lists of digits
def center( dataInput ):
	centerPoint = np.zeros(DIM)
	for sameDigit in dataInput:
		for digit in sameDigit:
			for inx in xrange(0, len(digit)):
				centerPoint[inx] += digit[inx]
	for x in xrange(0, len(centerPoint)):
		centerPoint[x] /= N

	return centerPoint

# Create the matrix from the datapaoins needed to create the matrix of PCs
def makeMatrixFromDatapoints ( dataInput ):
	s = (DIM, N)
	myMatrix = np.zeros(s)

	for i in xrange(0,DIM):
		for j in xrange(0, len(dataInput[0])):
			for sameDigit in xrange(0, 10):
				myMatrix[i][j * 10 + sameDigit] = dataInput[sameDigit][j][i]
	return myMatrix

# function computing the matrix of the features of the dataPoints 
def computeFeatureMatrix ( dataPoints, k ):
	size1 =  len(dataPoints[0]) * len(dataPoints)
	featureMatrix = np.arange(size1 * (k + 1) ).reshape(size1, k + 1)
	s = (size1, 10)
	zMatrix = np.zeros(s)

	thisDigit = -1
	digitClass = -1
	for sameDigit in dataPoints:
		digitClass += 1
		for digit in sameDigit:
			thisDigit += 1
			zMatrix[thisDigit][digitClass] = 1
			for x in xrange(0,k):
				featureMatrix[thisDigit][x] = digit[x]
			featureMatrix[thisDigit][k] = 1

	return (featureMatrix, zMatrix)

# function that computes the matrix that contains the PCs of the entire dataSet after we decided on the k = nr of features we want to use 
def computeMatrixOfFeatures ( dataPoints, k):
	Mat = makeMatrixFromDatapoints( dataPoints )
	MatT = Mat.transpose();

	MatAux = np.matmul(Mat, MatT)

	for i in xrange(0,DIM):
		for j in xrange(0,DIM):
			MatAux[i][j] /= N;

	#Generate the matrix decompostition. Now the matrix U contains the PCs of out dataset. We can discard as many columns as we want in order to meet the bias-variance dilemma
	U, S, VH = np.linalg.svd(MatAux, full_matrices = True)

	U = U[:, 0:k]
	U = U.transpose()

	return U

#function computing Wopt of the data points sending as parameters
def computeWopt ( dataPoints, k, alpha):
	#first we need the matrix of the datapoints compressed on the features that we selected based on the PCA model
	U = computeMatrixOfFeatures (dataPoints, k)


	featureData = [[] for x in xrange(0,10)] 

	thisDigit = -1
	for sameDigit in dataPoints:
		thisDigit += 1
		for digit in sameDigit:
			featureData[thisDigit].append( np.matmul(U, digit) )

	U = U.transpose()

	#plotDigit( trainData[6][9])
	#plotDigit( np.add(centerPoint, np.matmul(U, featureData[7][9]) ))

	 #Now, we compute Wopt using ridge regression. We construct the matrices Fi, Fi' and Z

	(featureMatrix, zMatrix) = computeFeatureMatrix(featureData, k)

	featureMatrixT = featureMatrix.transpose()

	# We use the formula 39 from LNs in order to compute Wopt

	Wopt = np.true_divide(np.matmul(featureMatrixT, featureMatrix), 0.001)
	Wopt = np.add(Wopt, np.multiply(np.identity(k + 1), alpha * alpha))
	Wopt = np.linalg.inv( Wopt)
	auxMatrix = np.true_divide( np.matmul(featureMatrixT, zMatrix), 0.001)
	Wopt = np.matmul(Wopt, auxMatrix)
	Wopt = Wopt.transpose()

	return (Wopt, featureMatrix, zMatrix)

trainData = [[] for x in xrange(0,10)] 
testData = [[] for x in xrange(0,10)] 
trainDataCentered = [[] for x in xrange(0,10)] 
testDataCentered = [[] for x in xrange(0,10)] 

#values that we use as variables
k = 100
alpha = 0

#reads the data
for x in xrange(0,2000):

	stringInput = f.read(721)

	vect = stringInput.split()
	vect = list(map(int,vect))


	digit = int(math.floor(x/200))
	if x % 200 < 100:
		trainData[digit].append(vect)
	else: testData[digit].append(vect)
		

#finds the center of the Data set, we need it for PCA
centerPoint = center(trainData)

#centers the train data
thisDigit = -1
for sameDigit in trainData:
	thisDigit += 1
	for digit in sameDigit:
		trainDataCentered[thisDigit].append( np.subtract(digit, centerPoint) )

#centers the test data
thisDigit = -1
for sameDigit in testData:
	thisDigit += 1
	for digit in sameDigit:
		testDataCentered[thisDigit].append( np.subtract(digit, centerPoint) )

minMSEval = 100000
minMISSval = 100000
minMSEtrain = 100000
minMISStrain = 100000

for vClass in xrange(0,10):
	#computes the train class and the validation class for cross-validation method
	validationClass = [[] for x in xrange(0,10)]
	trainClass = [[] for x in xrange(0,10)]  

	thisDigit = -1
	for sameDigit in xrange(0, 10):
		thisDigit += 1
		for digit in xrange(0, 100):
			if digit % 10 == vClass:
				validationClass[thisDigit].append(trainDataCentered[thisDigit][digit])
			else: trainClass[sameDigit].append(trainDataCentered[sameDigit][digit])
	
	(W, featureMatrixVal, featZMatrix) = computeWopt(validationClass, k, alpha)
	(Wopt, featureMatrix, zMatrix) = computeWopt(trainClass, k, alpha)

	nr = -1
	MISSval = 0
	MSEval = 0
	for element in xrange(0, 100):
		result = np.matmul( Wopt, featureMatrixVal[element] )
		MSEval += math.pow(np.linalg.norm(np.subtract(featZMatrix[element], result)), 2)
		if np.argmax(result) != np.argmax(featZMatrix[element]):
			MISSval += 1
	MSEval /= 100

	nr = -1
	MSEtrain = 0
	MISStrain = 0
	for element in xrange(0, 900):
		result = np.matmul( Wopt, featureMatrix[element] )
		MSEtrain += math.pow(np.linalg.norm(np.subtract(zMatrix[element], result)), 2)
		if np.argmax(result) != np.argmax(zMatrix[element]):
			MISStrain += 1
	MSEtrain /= 900

	if minMSEtrain > MSEtrain :
		minMSEtrain = MSEval
		WBestOpt = Wopt

	if minMISStrain > MISStrain :
		minMISStrain = MISStrain
		WBestOpt = Wopt

	if minMSEval > MSEval :
		minMSEval = MSEval
		WBestOpt = Wopt

	if minMISSval > MISSval :
		minMISSval = MISSval
		WBestOpt = Wopt


print minMSEval
print minMISSval

print minMSEtrain
print minMISStrain