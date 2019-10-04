データセットを使ってバッチサイズ分のデータを生成する。またデータのシャッフル機能も持つ。

dataloader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=True)

epochs = 4
for epoch in epochs:
    for i in dataloader:
        # 学習処理