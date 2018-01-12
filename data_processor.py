from keras.datasets import mnist

def get_data(nb_classes = 10):
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = X_train.reshape(X_train.shape[0], -1, 1)
	X_test = X_test.reshape(X_test.shape[0], -1, 1)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_means = np.mean(X_train, axis=0)
	X_stds = np.std(X_train, axis=0)
	X_train = (X_train - X_means)/(X_stds+1e-6)
	X_test = (X_test - X_means)/(X_stds+1e-6)
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train, X_test, y_test)