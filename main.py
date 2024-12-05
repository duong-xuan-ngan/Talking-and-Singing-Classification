from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
# Adjusting the input shape to match the spectrogram shape
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))  # Adjust input shape based on your spectrogram
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification for talking/singing
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Create the training and validation datasets
train_data = audio_data.shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_audio_data.batch(32)

# Train the model
history = model.fit(train_data, epochs=10, validation_data=val_data)

# Create the training and validation datasets
train_data = audio_data.shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_audio_data.batch(32)

# Train the model
history = model.fit(train_data, epochs=10, validation_data=val_data)
def augment_audio(y):
    # Time stretch example
    y_stretch = librosa.effects.time_stretch(y, rate=1.2)  # Stretch audio 1.2x
    return y_stretch

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[early_stop, model_checkpoint])