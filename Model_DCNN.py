import numpy as np
from tensorflow.keras import layers, models
from Evaluation import evaluation


# CNN Model
def CNN(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='softmax'))
    return model


# Model DCNN
def Model_DCNN(Train_Data, Train_Target, Test_Data, Test_Target):
    input_shape = Train_Data.shape[1:]
    output_shape = Train_Target.shape[1]

    # Build model
    model = CNN(input_shape, output_shape)
    model.summary()

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(Train_Data, Train_Target,
              batch_size=32,
              epochs=20,
              verbose=1,
              validation_split=0.3)

    # Predict on test data
    Pred = model.predict(Test_Data)
    Pred = (Pred == Pred.max(axis=1, keepdims=True)).astype(int)

    # Evaluate using external function
    Eval = evaluation(Test_Target, Pred)
    return Eval, Pred