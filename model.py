import os
from loader import DataLoader
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

CHECKPOINTS_DIR = "checkpoints"

def build_model(input_shape=(6000, 9), num_classes=2):
	print("Building model...")
	model = Sequential()
	model.add(LSTM(16, return_sequences=True, input_shape=input_shape))
	model.add(LSTM(32))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(num_classes, activation='softmax'))
	print(model.summary())
	return model

def compile_model(model):
	print("Compiling model...")
	model.compile(loss='categorical_crossentropy', 
					optimizer=Adam(lr=0.0001),
					metrics=['accuracy'])

def train_model(model, train_generator, train_spe, valid_generator, valid_spe, epochs=50):
	print("Training model...")
	os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
	
	model.fit_generator(generator=train_generator, 
						epochs=epochs, 
						steps_per_epoch=train_spe,
						validation_data=valid_generator,
						validation_steps=valid_spe,
						callbacks=[TensorBoard(), 
						ModelCheckpoint(os.path.join(CHECKPOINTS_DIR, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"), period=1)])

if __name__ == "__main__":
	model = build_model()
	compile_model(model)

	data = DataLoader(path="MobiFallDatasetv2.0")
	train_model(model, data.train_generator(), data.train_spe, data.valid_generator(), data.valid_spe)