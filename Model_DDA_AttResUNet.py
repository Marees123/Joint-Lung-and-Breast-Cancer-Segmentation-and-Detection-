import tensorflow as tf
from keras.src.layers import BatchNormalization
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.python.keras.layers import Conv2D, Concatenate, Conv2DTranspose, MaxPooling2D, Multiply, Activation, Add


def res_block(x, filters):
    shortcut = Conv2D(filters, 1, padding='same')(x)

    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def attention_gate(x, g, filters):
    theta_x = Conv2D(filters, 1, padding='same')(x)
    phi_g = Conv2D(filters, 1, padding='same')(g)

    add = Add()([theta_x, phi_g])
    act = Activation('relu')(add)

    psi = Conv2D(1, 1, activation='sigmoid')(act)

    return Multiply()([x, psi])


def encoder_block(x, filters):
    c = res_block(x, filters)
    p = MaxPooling2D((2, 2))(c)
    return c, p


def decoder_block(x, skip, filters):
    x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)

    skip = attention_gate(skip, x, filters)

    x = Concatenate()([x, skip])
    x = res_block(x, filters)

    return x


def build_DD_Attention_ResUNet(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(input_shape)

    # ======================
    # 🔹 ENCODER
    # ======================
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    # ======================
    #  BOTTLENECK
    # ======================
    bn = res_block(p4, 1024)

    # ======================
    # 🔹 DECODER 1 (Coarse)
    # ======================
    d1_1 = decoder_block(bn, c4, 512)
    d1_2 = decoder_block(d1_1, c3, 256)
    d1_3 = decoder_block(d1_2, c2, 128)
    d1_4 = decoder_block(d1_3, c1, 64)

    coarse_output = Conv2D(num_classes, 1, activation='sigmoid')(d1_4)

    # ======================
    # 🔹 DECODER 2 (Refinement)
    # ======================
    d2_1 = decoder_block(bn, c4, 512)
    d2_2 = decoder_block(d2_1, c3, 256)
    d2_3 = decoder_block(d2_2, c2, 128)
    d2_4 = decoder_block(d2_3, c1, 64)

    refined_output = Conv2D(num_classes, 1, activation='sigmoid')(d2_4)

    # ======================
    #  FUSION
    # ======================
    fusion = Concatenate()([coarse_output, refined_output])
    fusion = Conv2D(32, 3, padding='same', activation='relu')(fusion)

    final_output = Conv2D(num_classes, 1, activation='sigmoid')(fusion)

    model = Model(inputs, final_output)

    return model


from Evaluation import net_evaluation


def Model_DD_Attention_ResUNet(Image, GT, lr=1e-4, epochs=50):
    model = build_DD_Attention_ResUNet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(Image, GT, epochs=epochs, batch_size=4, validation_split=0.1)

    Pred = model.predict(Image)
    eval = net_evaluation(Pred, GT)

    return eval, Pred
