import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     UpSampling2D, Concatenate,
                                     BatchNormalization, Activation, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Evaluation import net_evaluation


# ---------------------------------------------------
# Residual Block
# ---------------------------------------------------
def residual_block(x, filters):
    shortcut = x

    # First Conv
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Conv
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Residual Connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


# ---------------------------------------------------
# ResUNet Model
# ---------------------------------------------------
def build_resunet(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = residual_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = residual_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = residual_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = residual_block(p4, 1024)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = residual_block(u6, 512)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = residual_block(u7, 256)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = residual_block(u8, 128)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = residual_block(u9, 64)

    # Output Layer
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Model Function
def Model_ResUNet(Image, GT):
    resunet_model = build_resunet()
    resunet_model.summary()

    Pred = resunet_model.predict(Image)
    eval = net_evaluation(Pred, GT)

    return eval, Pred
