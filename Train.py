import numpy as np
from sklearn.model_selection import train_test_split
from Models import ResUNet, MNet, UNet, DiceCoeff
from keras.callbacks import ModelCheckpoint, TensorBoard

EPOCHS = 20
BATCH_SIZE = 1

NAME = "unet"
DETAILS = "-Epoch {epoch:02d}-Dice {DiceCoeff:.4f}"

#model = MNet()
#model = ResUNet()
model = UNet()

x = np.load('./training/images.npy')
y = np.load('./training/masks.npy')

x = x[..., np.newaxis]
for i in range(x.shape[0]):
  x[i,:, :] = (x[i,:, :] - np.mean(x[i,:, :]))/ np.std(x[i,:, :])
print("Loaded {} images. Array shape: {}".format(len(x),x.shape))
y = y[..., np.newaxis]
print("Loaded {} masks. Array shape: {}".format(len(y),y.shape))

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=1)

checkpoint = ModelCheckpoint("./models/{}.model".format((NAME+DETAILS), monitor=[DiceCoeff], verbose=1, save_best_only=True, mode='max'))

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,validation_data=(x_val,y_val), callbacks=[checkpoint])

#model.save("./models/FINAL-{}.model".format(NAME))