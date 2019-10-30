TTA (Test Time Augmentation)

学習時ではなく推論時にもAugmentationを行い、推論の精度を上げる手法

tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(train_datagen.flow(x_val, batch_size=bs, shuffle=False), steps = len(x_val)/bs)
    predictions.append(preds)

pred = np.mean(predictions, axis=0)

np.mean(np.equal(np.argmax(y_val, axis=-1), np.argmax(pred, axis=-1)))