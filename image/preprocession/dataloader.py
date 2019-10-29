データセットを使ってバッチサイズ分のデータを生成する。またデータのシャッフル機能も持つ。

dataloader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=True)

epochs = 4
for epoch in epochs:
    for i in dataloader:
        # 学習処理


train_ = torch.utils.data.TensorDataset(torch.from_numpy(trX).float(), torch.from_numpy(trY.astype(np.int64)))
train_iter = torch.utils.data.DataLoader(train_, batch_size=64, shuffle=True)

以下でバッチ化されたサンプルを確認できる。
train_ = next(iter(train_iter))
print(train_)