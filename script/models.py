from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Convolution2D, Dense, merge, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.models import Model


class ScaledSigmoid(Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
        super(ScaledSigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaledSigmoid, self).build(input_shape)

    def call(self, x, mask=None):
        return self.alpha / (1 + K.exp(-x / self.beta))

    def get_output_shape_for(self, input_shape):
        return input_shape

def foamNet(rows, cols):
    # canale
    ch = 1

    # dimensioni
    rows = rows
    cols = cols

    input = Input(shape=(ch, rows, cols))

    x = Convolution2D(96, 11, 11, activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 5, 5, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(384, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(384, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, 1, 1, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    # superiore
    sup = Dense(1, activation='linear')(x)
    sup = ScaledSigmoid(alpha=1024, beta=1)(sup)

    # inferiore
    inf = Dense(1, activation='linear')(x)
    inf = ScaledSigmoid(alpha=1024, beta=1)(inf)

    foam = merge([sup, inf], mode='concat', concat_axis=1)

    return Model(input=input, output=foam)

def foamNet2(rows, cols):
    # canale
    ch = 1

    # dimensioni
    rows = rows
    cols = cols

    # attivazione
    activation = 'relu'

    input = Input(shape=(ch, rows, cols))

    x = Convolution2D(30, 5, 5, init='normal', subsample=(1, 1), activation=activation)(input)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(30, 5, 5, init='normal', subsample=(1, 1), activation=activation)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(30, 4, 4, init='normal', subsample=(1, 1), activation=activation)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(30, 3, 3, init='normal', subsample=(1, 1), activation=activation)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(120, 3, 3, init='normal', subsample=(1, 1), activation=activation)(x)

    x = Convolution2D(256, 3, 3, init='normal', subsample=(1, 1), activation=activation)(x)

    x = Convolution2D(256, 3, 3, init='normal', subsample=(1, 1), activation=activation)(x)

    x = Flatten()(x)
    x = Dense(256, init='normal', activation=activation)(x)
    x = Dropout(0.5)(x)

    x = Dense(256, init='normal', activation=activation)(x)
    x = Dropout(0.5)(x)

    foam = Dense(2, init='normal', activation=activation)(x)

    # # superiore
    # sup = Dense(1, activation='linear')(x)
    # sup = ScaledSigmoid(alpha=1024, beta=1)(sup)
    #
    # # inferiore
    # inf = Dense(1, activation='linear')(x)
    # inf = ScaledSigmoid(alpha=1024, beta=1)(inf)
    #
    # foam = merge([sup, inf], mode='concat', concat_axis=1)

    return Model(input=input, output=foam)


def get_eye_encoder(small):
    input_ratio = 1 if not small else 2
    eye_img = Input(shape=(3, 224/input_ratio, 224/input_ratio))

    h = Convolution2D(96, 11, 11, activation='relu')(eye_img)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Convolution2D(256, 5, 5, activation='relu')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Convolution2D(384, 3, 3, activation='relu')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    out = Convolution2D(64, 1, 1, activation='linear')(h)

    model = Model(input=eye_img, output=out)
    return model


def get_face_encoder(small):
    input_ratio = 1 if not small else 2
    face_img = Input(shape=(3, 224/input_ratio, 224/input_ratio))

    h = Convolution2D(96, 11, 11, activation='relu')(face_img)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Convolution2D(256, 5, 5, activation='relu')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Convolution2D(384, 3, 3, activation='relu')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    out = Convolution2D(64, 1, 1, activation='linear')(h)

    model = Model(input=face_img, output=out)
    return model


def get_eye_tracking_for_everyone_model(small):

    input_ratio = 1 if not small else 2
    dense_ratio = 1 if not small else 2

    eye_encoder = get_eye_encoder(small)
    face_encoder = get_face_encoder(small)

    r_eye = Input(shape=(3, 224/input_ratio, 224/input_ratio))
    l_eye = Input(shape=(3, 224/input_ratio, 224/input_ratio))
    face = Input(shape=(3, 224/input_ratio, 224/input_ratio))
    face_grid = Input(shape=(1, 25, 25))

    l_eye_code = eye_encoder(l_eye)
    r_eye_code = eye_encoder(r_eye)
    face_code = face_encoder(face)

    # Dense subnets
    e = merge([l_eye_code, r_eye_code], mode='concat', concat_axis=1)
    e = Flatten()(e)
    fc_e1 = Dense(128 / dense_ratio, activation='relu')(e)

    face_code = Flatten()(face_code)
    fc_f1 = Dense(128 / dense_ratio, activation='relu')(face_code)
    fc_f2 = Dense(64 / dense_ratio, activation='relu')(fc_f1)

    face_grid_code = Flatten()(face_grid)
    fc_fg1 = Dense(256 / dense_ratio, activation='relu')(face_grid_code)
    fc_fg2 = Dense(128 / dense_ratio, activation='relu')(fc_fg1)

    h = merge([fc_e1, fc_f2, fc_fg2], mode='concat', concat_axis=1)
    h = Dense(128 / dense_ratio, activation='relu')(h)

    x = Dense(1, activation='linear')(h)
    x = ScaledSigmoid(alpha=1920, beta=1)(x)

    y = Dense(1, activation='linear')(h)
    y = ScaledSigmoid(alpha=1080, beta=1)(y)

    gaze = merge([x, y], mode='concat', concat_axis=1)
    return Model(input=[r_eye, l_eye, face, face_grid], output=gaze)
