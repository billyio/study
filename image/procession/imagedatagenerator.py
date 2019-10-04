keras
リアルタイムにデータ拡張しながら，テンソル画像データのバッチを生成します
また，このジェネレータは，データを無限にループするので，無限にバッチを生成します

.flow(x, y)の使用例:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,  # 整数．画像をランダムに回転する回転範囲
    width_shift_range=0.2,  # ランダムに水平シフトする範囲
    height_shift_range=0.2, # ランダムに水平シフトする範囲
    horizontal_flip=True) # 水平方向に入力をランダムに反転

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break