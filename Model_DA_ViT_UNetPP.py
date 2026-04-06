import tensorflow as tf
from keras import Input
from keras.src.layers import BatchNormalization, MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers import Conv2D, Add, GlobalAveragePooling1D, Multiply, Reshape, MaxPooling2D, Concatenate, UpSampling2D
from tensorflow.python.keras.models import Model

from Evaluation import net_evaluation


def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def dual_attention_block(x, num_heads=4, key_dim=64):

    # ---- MHSA (Spatial Attention) ----
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    # ---- Feed Forward ----
    ffn = Dense(key_dim * 4, activation='relu')(x)
    ffn = Dense(key_dim)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization()(x)

    # ---- Channel Attention ----
    avg_pool = GlobalAveragePooling1D()(x)
    dense1 = Dense(key_dim // 2, activation='relu')(avg_pool)
    dense2 = Dense(key_dim, activation='sigmoid')(dense1)
    scale = Multiply()([x, tf.expand_dims(dense2, 1)])

    return scale


def patch_embedding(x, embed_dim=256):
    x = Conv2D(embed_dim, kernel_size=1, padding='same')(x)
    shape = tf.shape(x)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]

    x = Reshape((-1, C))(x)  # Flatten to tokens
    return x, H, W, C


def transformer_bottleneck(x, depth=4, embed_dim=256):

    tokens, H, W, C = patch_embedding(x, embed_dim)

    for _ in range(depth):
        tokens = dual_attention_block(tokens, num_heads=4, key_dim=embed_dim)

    # Reshape back to feature map
    x = Reshape((H, W, embed_dim))(tokens)

    return x


def build_DA_ViT_UNetPP(sol, input_shape=(256,256,3), num_classes=1):

    inputs = Input(input_shape)

    # ======================
    # 🔹 ENCODER (CNN)
    # ======================
    x0_0 = conv_block(inputs, 64)
    p0 = MaxPooling2D()(x0_0)

    x1_0 = conv_block(p0, 128)
    p1 = MaxPooling2D()(x1_0)

    x2_0 = conv_block(p1, 256)
    p2 = MaxPooling2D()(x2_0)

    x3_0 = conv_block(p2, 512)
    p3 = MaxPooling2D()(x3_0)

    # ======================
    #  TRANSFORMER BOTTLENECK
    # ======================
    x4_0 = transformer_bottleneck(p3, depth=4, embed_dim=int(sol[0]))

    # ======================
    # 🔹 UNet++ Nested Decoder (FULL)
    # ======================

    # Level 3
    x3_1 = conv_block(Concatenate()([x3_0, UpSampling2D()(x4_0)]), 512)

    # Level 2
    x2_1 = conv_block(Concatenate()([x2_0, UpSampling2D()(x3_0)]), 256)
    x2_2 = conv_block(Concatenate()([x2_0, x2_1, UpSampling2D()(x3_1)]), 256)

    # Level 1
    x1_1 = conv_block(Concatenate()([x1_0, UpSampling2D()(x2_0)]), 128)
    x1_2 = conv_block(Concatenate()([x1_0, x1_1, UpSampling2D()(x2_1)]), 128)
    x1_3 = conv_block(Concatenate()([x1_0, x1_1, x1_2, UpSampling2D()(x2_2)]), 128)

    # Level 0
    x0_1 = conv_block(Concatenate()([x0_0, UpSampling2D()(x1_0)]), 64)
    x0_2 = conv_block(Concatenate()([x0_0, x0_1, UpSampling2D()(x1_1)]), 64)
    x0_3 = conv_block(Concatenate()([x0_0, x0_1, x0_2, UpSampling2D()(x1_2)]), 64)
    x0_4 = conv_block(Concatenate()([x0_0, x0_1, x0_2, x0_3, UpSampling2D()(x1_3)]), 64)

    # ======================
    # OUTPUT
    # ======================
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x0_4)

    model = Model(inputs, outputs)
    return model


def Model_DA_ViT_UNetPP(Image, GT, sol=None):
    if sol is None:
        sol = [5, 0.01, 100]
    model = build_DA_ViT_UNetPP(sol)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(sol[1]),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    Pred = model.predict(Image, GT)
    eval = net_evaluation(Pred, GT)

    return eval, Pred
