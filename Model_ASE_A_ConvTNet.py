from keras import layers, models
from keras.src.optimizers import Adam
import numpy as np
from Evaluation_nrml import evaluation


def Model_ASE_A_ConvTNet(train_data, train_target, test_data, test_target, SPE=None, EP=None, HN=None, BS=None, sol=None):
    if sol is None:
        sol = [5, 0.01, 100]
    if SPE is None:
        SPE = int(sol[2])  # 5
    if HN is None:
        HN = int(sol[0])  # 128
    if BS is None:
        BS = 32
    if EP is None:
        EP = 30

    IMG_SIZE = 32

    # -------- Data Reshaping --------
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    num_classes = train_target.shape[-1]

    # -------- Model Definition --------
    inputs = layers.Input(shape=(32, 32, 3))

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # SE Attention 1
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(ch // 16, activation='relu')(se)
    se = layers.Dense(ch, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, ch))(se)
    x = layers.multiply([x, se])

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # SE Attention 2
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(ch // 16, activation='relu')(se)
    se = layers.Dense(ch, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, ch))(se)
    x = layers.multiply([x, se])

    # -------- Conv → Transformer Conversion --------
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((h * w, c))(x)

    # Transformer Block 1
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(128, activation='relu')(x)
    ffn = layers.Dense(c)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)

    # Transformer Block 2
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(128, activation='relu')(x)
    ffn = layers.Dense(c)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)

    # -------- Classification Head --------
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(HN, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.summary()

    # -------- Compile --------
    model.compile(
        optimizer=Adam(learning_rate=sol[1]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )  # 0.001

    # -------- Train --------
    model.fit(
        Train_X, train_target,
        epochs=EP,
        batch_size=BS,
        steps_per_epoch=SPE,
        validation_data=(Test_X, test_target),
        verbose=1
    )

    # -------- Predict & Evaluate --------
    pred = model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred
