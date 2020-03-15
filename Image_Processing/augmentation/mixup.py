参照
・@yu4uのqiita
# https://qiita.com/yu4u/items/70aa007346ec73b7ff05
・原著論文
# https://arxiv.org/abs/1710.09412

mixupは、2つの訓練サンプルのペアを混合して新たな訓練サンプルを作成するdata augmentation手法の1つ
具体的には、データとラベルのペア(X1,y1)(X1,y1), (X2,y2)(X2,y2)から、下記の式により新たな訓練サンプル(X,y)(X,y)を作成
ラベルy1,y2y1,y2はone-hot表現のベクトル

X1,X2X1,X2は任意のベクトルやテンソル

X=λX1+(1−λ)X2y=λy1+(1−λ)y2
X=λX1+(1−λ)X2y=λy1+(1−λ)y2

λ∈[0,1]λ∈[0,1]は、ベータ分布Be(α,α)Be(α,α)からのサンプリングにより取得し、ααはハイパーパラメータとなる
特徴的なのは、データX1,X2X1,X2だけではなく、ラベルy1,y2y1,y2も混合してしまう

# 
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        _, class_num = self.y_train.shape
        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])

        return X, y


model.fit_generator(generator=training_generator,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=epochs, verbose=1,
                    callbacks=callbacks)

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

training_generator = MixupGenerator(x_train, y_train, datagen=datagen)()

# 
from mixup_generator import MixupGenerator

generator1 = MixupGenerator(x_train, y_train, batch_size=batch_size)()
x, y = next(generator1)
x_train = np.vstack((x_train,x))
y_train = np.vstack((y_train,y))

