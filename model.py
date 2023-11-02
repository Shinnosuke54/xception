import keras
import tensorflow as tf
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input

# base model(Xception)を読み込み
# include_top : 出力層側を含むかどうか
base_model =  tf.keras.applications.xception.Xception(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))

# base modelに出力層を追加
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation="relu")(x)
prediction=Dense(10,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=prediction)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer="adam",
    metrics=["accuracy"]
)
