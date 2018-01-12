from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Activation

#using keras

def keras_model(X_train, Y_train, nb_classes, X_test = None, Y_test = None, hidden_units = 50):
	model = Sequential()
	model.add(LSTM(hidden_units, input_shape=X_train.shape[1:], inner_init='glorot_uniform',
	forget_bias_init='one', activation='tanh', inner_activation='sigmoid'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	rmsprop = RMSprop(clipnorm = 5.0)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
	return model

def train_model(model, X_train, Y_train, X_test = None, Y_test = None, epochs = 100, batch_size = 10, save_model = False):
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test)
	)

if __name__ == '__main__':
	import data_processor as dp
	(X_train, Y_train, X_test, Y_test) = dp.get_data()
	model = keras_model(X_train, Y_train, 10, X_test, Y_test)
	train_model(model, X_train, Y_train, X_test, Y_test)