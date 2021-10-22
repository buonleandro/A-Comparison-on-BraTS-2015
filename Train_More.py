import numpy as np
from sklearn.model_selection import train_test_split
from Models import BceDiceLoss, DiceCoeff
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

EPOCHS = 10
BATCH_SIZE = 2

NAME = "resunet0910"
METRICS = "-E{epoch:02d}-Dice{DiceCoeff:.4f}"

dependencies = {
    'DiceCoeff': DiceCoeff,
    'BceDiceLoss' : BceDiceLoss
}
model = load_model('./models/BEST-resunet-Epoch 14-Dice 0.4488.model', custom_objects=dependencies)

x = np.load('./training/images.npy')
y = np.load('./training/masks.npy')

x = x[..., np.newaxis]
y = y[..., np.newaxis]

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=1)

checkpoint = ModelCheckpoint("./models/{}.model".format((NAME+METRICS), monitor=[DiceCoeff], verbose=1, save_best_only=True, mode='max'))

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,validation_data=(x_val,y_val), callbacks=[checkpoint])

#model.save("./models/FINAL-{}.model".format(NAME))