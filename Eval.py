import numpy as np
from sklearn.model_selection import train_test_split
from Models import BceDiceLoss, DiceCoeff
from keras.models import load_model

dependencies = {
    'DiceCoeff': DiceCoeff,
    'BceDiceLoss' : BceDiceLoss
}

model = load_model('./models/BEST-resunet-Epoch 14-Dice 0.4488.model', custom_objects=dependencies)

x = np.load('./training/images.npy')
y = np.load('./training/masks.npy')

x = x[..., np.newaxis]
y = y[..., np.newaxis]

_, x_val, _, y_val = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=1)

results=model.evaluate(x_val,y_val,batch_size=2)
print(results)