# Model Creation
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input , outputs=base_model.layers[-2].output)

sequentialModel = Sequential()
for layers in model.layers:
  sequentialModel.add(layers)

# Freezing the base model layers
for layer in sequentialModel.layers:
  layer.trainable=False

# Adding last Classification Layer
sequentialModel.add(Dense(7, activation='softmax', name='Prediction'))
sequentialModel.summary()

# Compiling Model
sequentialModel.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])