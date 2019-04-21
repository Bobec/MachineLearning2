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
		for j in xrange(0, 100):
			for sameDigit in xrange(0, 10):
				myMatrix[i][j * 10 + sameDigit] = dataInput[sameDigit][j][i]
	return myMatrix



trainData = [[] for x in xrange(0,10)] 
testData = [[] for x in xrange(0,10)] 
trainDataCentered = [[] for x in xrange(0,10)] 
testDataCentered = [[] for x in xrange(0,10)] 
featureData = [[] for x in xrange(0,10)] 

for x in xrange(0,2000):

	stringInput = f.read(721)

	vect = stringInput.split()
	vect = list(map(int,vect))


	digit = int(math.floor(x/200))
	if x % 200 < 100:
		trainData[digit].append(vect)
	else: testData[digit].append(vect)
		

centerPoint = center(trainData)

thisDigit = -1
for sameDigit in trainData:
	thisDigit += 1
	for digit in sameDigit:
		trainDataCentered[thisDigit].append( np.subtract(digit, centerPoint) )

thisDigit = -1
for sameDigit in testData:
	thisDigit += 1
	for digit in sameDigit:
		testDataCentered[thisDigit].append( np.subtract(digit, centerPoint) )

#plotDigit( trainData[5][5] )


Mat = makeMatrixFromDatapoints( trainDataCentered )
MatT = Mat.transpose();

MatAux = np.matmul(Mat, MatT)

for i in xrange(0,DIM):
	for j in xrange(0,DIM):
		MatAux[i][j] /= N;

#Generate the matrix decompostition. Now the matrix U contains the PCs of out dataset. We can discard as many columns as we want in order to meet the bias-variance dilemma
U, S, VH = np.linalg.svd(MatAux, full_matrices = True)

print U.shape

k = 50

U = U[:, 0:k]
U = U.transpose()

thisDigit = -1
for sameDigit in trainDataCentered:
	thisDigit += 1
	for digit in sameDigit:
		featureData[thisDigit].append( np.matmul(U, digit) )

U = U.transpose()


#plotDigit( trainData[6][9])
#plotDigit( np.add(centerPoint, np.matmul(U, featureData[7][9]) ))

 #Now, we compute Wopt using plain linear regression. We construct the matrices Fi, Fi' and Z

featureMatrix = np.arange(11000).reshape(1000, 11)
s = (N, 10)
zMatrix = np.zeros(s)

thisDigit = -1
digitClass = -1
for sameDigit in featureData:
	digitClass += 1
	for digit in sameDigit:
		thisDigit += 1
		zMatrix[thisDigit][digitClass] = 1
		for x in xrange(0,10):
			featureMatrix[thisDigit][x] = digit[x]
		featureMatrix[thisDigit][10] = 1

featureMatrixT = featureMatrix.transpose()
print featureMatrixT.shape

Wopt = np.linalg.inv( np.true_divide(np.matmul(featureMatrixT, featureMatrix), 0.001))
auxMatrix = np.true_divide( np.matmul(featureMatrixT, zMatrix), 0.001)
Wopt = np.matmul(Wopt, auxMatrix)
Wopt = Wopt.transpose()

result = np.matmul( Wopt, featureMatrix[974] )

print np.argmax(result) == np.argmax(zMatrix[974])