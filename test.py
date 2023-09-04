import tensorflow as tf
# import tensorflow_datasets as tfds

import sklearn.datasets as skd
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split

X, Y = skd.fetch_california_housing(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ds = tfds.load('radon', split='train')
# Y = ds['activity']

def evaluate(Y_test, preds):
    print(f"R2 score: {skm.r2_score(Y_test, preds)}")
    print(f"MAE: {skm.mean_absolute_error(Y_test, preds)}")

def create_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs
    x = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mean_absolute_error")
    return model


model = create_model(X_train[0].shape)
model.fit(X_train, Y_train)
evaluate(Y_test, model.predict(X_test))