from keras.layers import Input, Dense, Add, Activation
from keras.models import Model
import tensorflow as tf
from Evaluation import evaluation


# ==========================================
# FENN Cell (Custom Elman Unit)
# ==========================================
class FENN_Cell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(FENN_Cell, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.Wx = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.Wh = self.add_weight(shape=(self.units, self.units),
                                  initializer='orthogonal',
                                  trainable=True)

        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, states):
        prev_h = states[0]

        h = tf.matmul(inputs, self.Wx) + tf.matmul(prev_h, self.Wh) + self.b
        h = tf.nn.tanh(h)

        return h, [h]


# ==========================================
# Build FENN Model
# ==========================================
def Model_FENN(train_data, train_target,
               test_data, test_target,
               time_steps,
               n_features,
               n_classes=3,
               epochs=50,
               batch_size=32,
               opt='adam'):

    activation = 'softmax'

    # ======================================
    # Input
    # ======================================
    inputs = Input(shape=(time_steps, n_features))

    # ======================================
    # FENN Layer (RNN Wrapper)
    # ======================================
    fenn_cell = FENN_Cell(units=128)
    rnn_layer = tf.keras.layers.RNN(fenn_cell, return_sequences=False)(inputs)

    # ======================================
    # Fully Connected Layers
    # ======================================
    x = Dense(128, activation='relu')(rnn_layer)
    x = Dense(64, activation='relu')(x)

    outputs = Dense(n_classes, activation=activation)(x)

    # ======================================
    # Model
    # ======================================
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # ======================================
    # TRAINING
    # ======================================
    model.fit(train_data, train_target,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1)

    # ======================================
    # TESTING
    # ======================================
    pred = model.predict(test_data)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)

    return Eval, pred