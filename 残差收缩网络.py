import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, \
    Dense, Add, Multiply, Concatenate, Reshape
from tensorflow.keras.models import Model


def residual_shrinkage_block(x, filters, strides=1):
    """残差收缩模块（支持通道独立阈值）"""
    # 主路径
    shortcut = x
    x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # 收缩路径（自动学习阈值）
    gap = GlobalAveragePooling2D()(x)  # [B, C]
    gap = Dense(filters // 16, activation='relu')(gap)
    gap = Dense(filters, activation='sigmoid')(gap)  # 通道独立阈值
    gap = Reshape((1, 1, filters))(gap)  # [B, 1, 1, C]
    x = Multiply()([x, gap])  # 软阈值化

    # 调整shortcut维度
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # 残差连接
    x = Add()([x, shortcut])
    return Activation('relu')(x)


def build_drsn(input_shape=(224, 224, 3), num_classes=23, pretrained_weights=None):
    """构建完整DRSN网络，支持迁移学习"""
    # 输入层
    inputs = Input(shape=input_shape)

    # -------------------- 初始下采样层 --------------------
    # Convolution_1 (224x224x3 -> 112x112x64)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # MaxPooling (112x112x64 -> 56x56x64)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # -------------------- 多尺度特征提取层 --------------------
    # Convolution_2 (56x56x64 -> 56x56x64)
    branch1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    branch1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(branch1)

    branch2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)

    branch3 = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    branch3 = Conv2D(64, kernel_size=1, strides=1, padding='same')(branch3)

    branch4 = Conv2D(64, kernel_size=1, strides=1, padding='same')(x)

    x = Concatenate()([branch1, branch2, branch3, branch4])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # -------------------- 残差收缩模块堆叠 --------------------
    # Convolution_3 (56x56x64 -> 56x56x64) - 2个模块
    for _ in range(2):
        x = residual_shrinkage_block(x, filters=64, strides=1)

    # Convolution_4 (56x56x64 -> 28x28x128) - 4个模块（第一个步长=2）
    x = residual_shrinkage_block(x, filters=128, strides=2)
    for _ in range(3):
        x = residual_shrinkage_block(x, filters=128, strides=1)

    # Convolution_5 (28x28x128 -> 14x14x256) - 6个模块（第一个步长=2）
    x = residual_shrinkage_block(x, filters=256, strides=2)
    for _ in range(5):
        x = residual_shrinkage_block(x, filters=256, strides=1)

    # Convolution_6 (14x14x256 -> 7x7x512) - 3个模块（第一个步长=2）
    x = residual_shrinkage_block(x, filters=512, strides=2)
    for _ in range(2):
        x = residual_shrinkage_block(x, filters=512, strides=1)

    # -------------------- 分类层 --------------------
    x = GlobalAveragePooling2D()(x)  # [B, 512]
    outputs = Dense(num_classes, activation='softmax')(x)

    # 构建模型
    model = Model(inputs, outputs)

    # 加载预训练权重（迁移学习）
    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)

    return model


# ----------------------------------------------
# 实例化网络（输入尺寸224x224x3，输出23类）
model = build_drsn(input_shape=(224, 224, 3), num_classes=23)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
test