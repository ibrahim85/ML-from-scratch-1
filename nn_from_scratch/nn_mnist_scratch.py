import pandas as pd
import numpy as np

x = pd.read_csv("./train.csv")
X_ = np.array(x)
X = X_[:,1:]
X = (X - X.mean())/X.std()
Y = X_[:,0]
X_train = X[:36000,:]
X_crossval = X[36000:,:]
Y_train = Y[:36000,]
Y_crossval = Y[36000:,]

IMAGE_SIZE = 28*28
H1_SIZE = 256
H2_SIZE = 64
OUPUT_SIZE = 10
BATCH_SIZE = 256
EPOCH = 20
ALPHA = 0.005 #learning rate
REG_LAMBDA = 0.005 #regularization_para
REGULARIZATON = True

def accuracy(pred, y):
	return (100.0 * np.sum(pred == y) / y.shape[0])

def initailize_weights():
	np.random.seed(0)
	model = {}
	model['W1'] = np.random.randn(IMAGE_SIZE, H1_SIZE) / np.sqrt(IMAGE_SIZE)
	model['B1'] = np.zeros((1,H1_SIZE))
	model['W2'] = np.random.randn(H1_SIZE,H2_SIZE) / np.sqrt(H1_SIZE)
	model['B2'] = np.zeros((1,H2_SIZE))
	model['W3'] = np.random.randn(H2_SIZE,OUPUT_SIZE) / np.sqrt(H2_SIZE)
	model['B3'] = np.zeros((1,OUPUT_SIZE))
	return model
	
def forward_prop(model, x):
	z1 = x.dot(model['W1']) + model['B1']
	a1 = np.tanh(z1)
	z2 = a1.dot(model['W2']) + model['B2']
	a2 = np.tanh(z2)
	z3 = a2.dot(model['W3']) + model['B3']
	h_x = np.exp(z3)
	y_out = h_x/ np.sum(h_x, axis=1, keepdims=True)
	return a1, a2, y_out

def backprop(model, x, a1, a2, y, y_out):
	delta4 = y_out
	delta4[range(y.shape[0]), y] -= 1
	dw3 = (a2.T).dot(delta4)
	db3 = np.sum(delta4, axis = 0)
	delta3 = (1 - np.square(a2)) * delta4.dot(model['W3'].T)
	dw2 = (a1.T).dot(delta3)
	db2 = np.sum(delta3, axis = 0)
	delta2 = (1 - np.square(a1)) * delta3.dot(model['W2'].T)
	dw1 = (x.T).dot(delta2)
	db1 = np.sum(delta2, axis = 0)
	
	if REGULARIZATON:
		dw3 += REG_LAMBDA * model['W3']
		dw2 += REG_LAMBDA * model['W2']
		dw1 += REG_LAMBDA * model['W1']

	model['W1'] += -ALPHA * dw1
	model['B1'] += -ALPHA * db1
	model['W2'] += -ALPHA * dw2
	model['B2'] += -ALPHA * db2
	model['W3'] += -ALPHA * dw3
	model['B3'] += -ALPHA * db3

	return model

def loss(model, p , y):
	corect_logprobs = -np.log(p[range(y.shape[0]), y])
	l = np.sum(corect_logprobs)
	if REGULARIZATON :
		l += REG_LAMBDA/2 * (np.sum(np.square(model['W1'])) + np.sum(np.square(model['W2']))+ np.sum(np.square(model['W3'])))
	return 1./y.shape[0] * l

def predict(y_out):
	return np.argmax(y_out, axis=1)

def main():
	model = initailize_weights()	
	for ix in range(EPOCH):
		print ("\nEpoch : %d" % (ix+1))
		count = 0
		while count < Y_train.shape[0]:
			batch_data = X_train[count:(count + BATCH_SIZE),:]
			batch_labels = Y_train[count:(count + BATCH_SIZE),]
			count += BATCH_SIZE
			a1, a2, p = forward_prop(model, batch_data)
			backprop(model, batch_data, a1, a2, batch_labels, p)
		_, _, p = forward_prop(model, X_train)	
		print ("training_loss : %.3f" % (loss(model, p, Y_train)))
		_, _, p = forward_prop(model, X_crossval)
		pred = predict(p)
		print ("Accuracy on validation set: %.2f%% | validation_loss : %.3f" % (accuracy(pred, Y_crossval),loss(model, p, Y_crossval)))
	print ("************************Completed*********************")

if __name__ == "__main__":
	main()