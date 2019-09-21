
import numpy as np
from keras.applications.resnet import ResNet50
from keras.layers import Input, Flatten
from keras.models import Model

def extract_feature(X, cnn_arch="resnet50"):
    if cnn_arch == "resnet50":
        from keras.applications.resnet import preprocess_input
        cnn = ResNet50(include_top=False, weights='imagenet')
    else:
        raise ValueError(f"Not supported cnn_arch {cnn_arch}")

    input = Input(shape=X.shape[1:],name = 'image_input')
    x = cnn(input)
    x = Flatten()(x)
    model = Model(inputs=input, outputs=x)

    return model.predict(preprocess_input(X), batch_size=64)
