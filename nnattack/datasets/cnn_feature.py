
import numpy as np
from keras.applications.resnet import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Flatten
from keras.models import Model
from skimage.transform import resize



def extract_feature(X, cnn_arch="resnet50"):
    if cnn_arch == "resnet50":
        from keras.applications.resnet import preprocess_input
        cnn = ResNet50(include_top=False, weights='imagenet')

    elif cnn_arch == "resnet50v2":
        from keras.applications.resnet_v2 import preprocess_input
        cnn = ResNet50V2(include_top=False, weights='imagenet')
    elif cnn_arch == "inceptionv3":
        from keras.applications.inception_v3 import preprocess_input
        cnn = InceptionV3(include_top=False, weights='imagenet')
        X = np.array([resize(X, (299, 299, 3)) for x in X])
        import ipdb; ipdb.set_trace()
    else:
        raise ValueError(f"Not supported cnn_arch {cnn_arch}")

    input = Input(shape=X.shape[1:], name = 'image_input')
    x = cnn(input)
    x = Flatten()(x)
    model = Model(inputs=input, outputs=x)

    X = preprocess_input(X)

    return model.predict(X, batch_size=64)
