from keras.layers import (Input, Conv2D, DepthwiseConv2D, LayerNormalization,
                          Dense, GlobalAveragePooling2D, Add, Activation,
                          Dropout, BatchNormalization, Concatenate,
                          Reshape, Softmax, Multiply)
from keras.models import Model
import tensorflow as tf
from Evaluation import evaluation


# ConvNeXt V2 Block
def convnext_block(x, filters):
    shortcut = x

    x = DepthwiseConv2D(kernel_size=7, padding='same')(x)
    x = LayerNormalization()(x)

    x = Conv2D(4 * filters, kernel_size=1, padding='same')(x)
    x = Activation('gelu')(x)

    x = Conv2D(filters, kernel_size=1, padding='same')(x)

    x = Add()([shortcut, x])
    return x


# Multi-Scale Block
def multi_scale_block(x, filters):
    c3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    c5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(x)
    c7 = Conv2D(filters, (7, 7), padding='same', activation='relu')(x)

    out = Concatenate()([c3, c5, c7])
    out = Conv2D(filters, (1, 1), padding='same')(out)

    return out


# Cross-Attention Fusion
def cross_attention(f1, f2):
    Q = Conv2D(64, 1)(f1)
    K = Conv2D(64, 1)(f2)
    V = Conv2D(64, 1)(f2)

    Q = Reshape((-1, 64))(Q)
    K = Reshape((-1, 64))(K)
    V = Reshape((-1, 64))(V)

    attn = tf.matmul(Q, K, transpose_b=True)
    attn = Softmax()(attn)

    out = tf.matmul(attn, V)

    out = Reshape((f1.shape[1], f1.shape[2], 64))(out)
    return out


# Explainable Attention
def explainable_attention(x):
    att = Conv2D(1, (1, 1), activation='sigmoid', name="Attention_Map")(x)
    out = Multiply()([x, att])
    return out, att


# Feature Extractor Branch (for each modality)
def feature_branch(inputs, filters):
    x = Conv2D(filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = convnext_block(x, filters)
    x = multi_scale_block(x, filters)

    x = convnext_block(x, filters * 2)
    x = multi_scale_block(x, filters * 2)

    return x


def Model_ERMSC_ConvNeXtV2(breast_img, Lung_Cancer, Tar, sol=None, n_classes=3,opt='adam'):
    if sol is None:
        sol = [5, 0.01, 100]
    learnper = round(Lung_Cancer.shape[0] * 0.75)
    train_lung = Lung_Cancer[:learnper, :]
    train_target = Tar[:learnper, :]
    test_lung = Lung_Cancer[learnper:, :]
    test_target = Tar[learnper:, :]
    lung_shape = Lung_Cancer.shape
    breast_shape = breast_img.shape
    test_breast = breast_img

    train_breast = breast_img[:learnper, :]
    train_target = Tar[:learnper, :]
    test_lung = Lung_Cancer[learnper:, :]
    test_target = Tar[learnper:, :]

    activation = 'softmax'

    # Inputs (Two Modalities)
    breast_input = Input(shape=breast_shape, name="Breast_Input")
    lung_input = Input(shape=lung_shape, name="Lung_Input")

    # Separate Feature Extraction
    breast_feat = feature_branch(breast_input, 64)
    lung_feat = feature_branch(lung_input, 64)

    # Cross-Attention Fusion
    attn1 = cross_attention(breast_feat, lung_feat)
    attn2 = cross_attention(lung_feat, breast_feat)

    fused = Concatenate()([breast_feat, lung_feat, attn1, attn2])

    # Explainability
    fused, att_map = explainable_attention(fused)

    # Final Processing
    x = convnext_block(fused, 256)

    x = GlobalAveragePooling2D()(x)

    x = Dense(int(sol[0]), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(n_classes, activation=activation)(x)

    # Model
    model = Model(inputs=[breast_input, lung_input], outputs=output)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        [train_breast, train_lung], train_target,
        validation_data=([test_breast, test_lung], test_target),
        epochs=50,
        steps_per_epochs=int(sol[2]),
        batch_size=16,
        verbose=1
    )

    pred = model.predict([test_breast, test_lung])

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)

    return Eval, pred