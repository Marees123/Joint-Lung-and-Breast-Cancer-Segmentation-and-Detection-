from keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization, Dropout, GRU
from keras.models import Model
from Evaluation import evaluation


def Model_RAN(train_data, test_data, test_target, opt, n_classes=3, sol=None):
    activation = ['Sigmoid', 'relu', 'tanh', 'linear']
    if opt is None:
        opt = 'Adam'#[50, 4, 0, 0, 0, 16]
    # Initialize a Keras Tensor of input_shape
    input_data = Input(shape=train_data)

    # Initial Layers before Attention Module

    # Doing padding because I'm having trouble with img dims that are <= 28
    if train_data.shape[0] <= 28 or train_data.shape[1] <= 28:
        x_dim_inc = (32 - train_data.shape[0]) // 2
        y_dim_inc = (32 - train_data.shape[1]) // 2

        # Pad the input data to 32x32
        padded_input_data = ZeroPadding2D((x_dim_inc, y_dim_inc))(train_data)
        conv_layer_1 = Conv2D(filters=32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same')(padded_input_data)
    else:
        conv_layer_1 = Conv2D(filters=32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same')(train_data)

    max_pool_layer_1 = MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='same')(conv_layer_1)

    # Residual Unit then Attention Module #1
    res_unit_1 = residual_unit(max_pool_layer_1, filters=[32, 64, 128])
    att_mod_1 = attention_module(res_unit_1, filters=[32, 64, 128])

    # Residual Unit then Attention Module #2
    res_unit_2 = residual_unit(att_mod_1, filters=[32, 64, 128])
    att_mod_2 = attention_module(res_unit_2, filters=[32, 64, 128])

    # Residual Unit then Attention Module #3
    res_unit_3 = residual_unit(att_mod_2, filters=[32, 64, 128])
    att_mod_3 = attention_module(res_unit_3, filters=[32, 64, 128])

    # Ending it all
    res_unit_end_1 = residual_unit(att_mod_3, filters=[32, 32, 64])
    res_unit_end_2 = residual_unit(res_unit_end_1, filters=[32, 32, 64])
    res_unit_end_3 = residual_unit(res_unit_end_2, filters=[32, 32, 64])
    res_unit_end_4 = residual_unit(res_unit_end_3, filters=[32, 32, 64])

    # Avg Pooling
    avg_pool_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(res_unit_end_4)

    # Flatten the data
    flatten_op = Flatten()(avg_pool_layer)

    # FC Layers for prediction
    fully_connected_layer_1 = Dense(4, activation='relu')(flatten_op)
    dropout_layer_1 = Dropout(0.5)(fully_connected_layer_1)
    fully_connected_layer_2 = Dense(256, activation='relu')(dropout_layer_1)
    dropout_layer_2 = Dropout(0.5)(fully_connected_layer_2)
    fully_connected_layer_3 = Dense(256, activation='relu')(dropout_layer_2)
    dropout_layer_3 = Dropout(0.5)(fully_connected_layer_3)
    fully_connected_layer_last = Dense(n_classes, activation=activation[int(sol)])(dropout_layer_3)

    # Fully constructed model
    model = Model(inputs=test_data, outputs=fully_connected_layer_last)
    pred = model.predict(test_target)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred


# Pre-Activation Identity ResUnit Bottleneck Architecture
def residual_unit(residual_input_data, filters):
    # Hold input_x here for later processing
    identity_x = residual_input_data

    filter1, filter2, filter3 = filters

    # Layer 1
    batch_norm_op_1 = BatchNormalization()(residual_input_data)
    activation_op_1 = Activation('relu')(batch_norm_op_1)
    conv_op_1 = Conv2D(filters=filter1,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same')(activation_op_1)

    # Layer 2
    batch_norm_op_2 = BatchNormalization()(conv_op_1)
    activation_op_2 = Activation('relu')(batch_norm_op_2)
    conv_op_2 = Conv2D(filters=filter2,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       padding='same')(activation_op_2)

    # Layer 3
    batch_norm_op_3 = BatchNormalization()(conv_op_2)
    activation_op_3 = Activation('relu')(batch_norm_op_3)
    conv_op_3 = Conv2D(filters=filter3,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same')(activation_op_3)

    # Element-wise Addition
    if identity_x.shape[-1].value != conv_op_3.shape[-1].value:
        filter_n = conv_op_3.shape[-1].value

        identity_x = Conv2D(filters=filter_n,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same')(identity_x)

    output = Add()([identity_x, conv_op_3])

    return output


def attention_module(attention_input_data, filters, p=1):
    # Send input_x through #p residual_units
    p_res_unit_op_1 = attention_input_data
    for _ in range(p):
        p_res_unit_op_1 = residual_unit(p_res_unit_op_1, filters=filters)

    # Perform Trunk Branch Operation
    trunk_branch_op = trunk_branch(trunk_input_data=p_res_unit_op_1, filters=filters)

    # Perform Mask Branch Operation
    mask_branch_op = mask_branch(mask_input_data=p_res_unit_op_1, filters=filters)

    # Perform Attention Residual Learning: Combine Trunk and Mask branch results
    ar_learning_op = attention_residual_learning(mask_input=mask_branch_op, trunk_input=trunk_branch_op)

    # Send branch results through #p residual_units
    p_res_unit_op_2 = ar_learning_op
    for _ in range(p):
        p_res_unit_op_2 = residual_unit(p_res_unit_op_2, filters=filters)

    return p_res_unit_op_2


def trunk_branch(trunk_input_data, filters, t=1):
    # sequence of residual units, default=2
    t_res_unit_op = trunk_input_data
    for _ in range(t):
        t_res_unit_op = residual_unit(t_res_unit_op, filters=filters)

    return t_res_unit_op


def mask_branch(mask_input_data, filters, m=3, r=1):
    # r = num of residual units between adjacent pooling layers, default=1
    # m = num max pooling / linear interpolations to do

    # Downsampling Step Initialization - Top
    downsampling = MaxPool2D(pool_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(mask_input_data)

    for _ in range(m):
        # Perform residual units ops r times between adjacent pooling layers
        for j in range(r):
            downsampling = residual_unit(residual_input_data=downsampling, filters=filters)

        # Last pooling step before middle step - Bottom
        downsampling = MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='same')(downsampling)

    # ===================================================================================================

    # Middle Residuals - Perform 2*r residual units steps before upsampling
    middleware = downsampling
    for _ in range(2 * r):
        middleware = residual_unit(residual_input_data=middleware, filters=filters)

    # ===================================================================================================

    # Upsampling Step Initialization - Top
    upsampling = UpSampling2D(size=(2, 2))(middleware)

    for _ in range(m):
        # Perform residual units ops r times between adjacent pooling layers
        for j in range(r):
            upsampling = residual_unit(residual_input_data=upsampling, filters=filters)

        # Last interpolation step - Bottom
        upsampling = UpSampling2D(size=(2, 2))(upsampling)

    conv_filter = upsampling.shape[-1].value

    conv1 = Conv2D(filters=conv_filter,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='same')(upsampling)

    conv2 = Conv2D(filters=conv_filter,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='same')(conv1)

    sigmoid = Activation('sigmoid')(conv2)

    return sigmoid


def attention_residual_learning(mask_input, trunk_input):
    # https://stackoverflow.com/a/53361303/9221241
    Mx = Lambda(lambda x: 1 + x)(mask_input)  # 1 + mask
    return Multiply()([Mx, trunk_input])  # M(x) * T(x)
