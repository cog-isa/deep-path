import keras, logging, itertools, numpy
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Reshape, \
    Convolution2D, MaxPooling2D, AveragePooling2D, \
    BatchNormalization, Activation, Lambda, Reshape, \
    Permute, TimeDistributed, merge, Input
from keras.applications.vgg16 import VGG16

from ..keras_utils import get_backend
from ..base_utils import load_object_from_dict
from .training_data_gen import scale_tensors


logger = logging.getLogger()


class OneLayer(object):
    def _build_inner_model(self, input_layer):
        h = Flatten()(input_layer)
        return h


class TwoLayer(object):
    def __init__(self,
                 hidden_size = 10,
                 hidden_activation = 'relu',
                 dropout = 0.8,
                 *args, **kwargs):
        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation
        self.dropout = dropout
        super(TwoLayer, self).__init__(*args, **kwargs)

    def _build_inner_model(self, input_layer):
        h = Flatten()(input_layer)
        h = Dense(self.hidden_size,
                  activation = self.hidden_activation)(h)
        h = Dropout(self.dropout)(h)
        return h


class ConvAndDense(object):
    def __init__(self,
                 hidden_sizes = [10],
                 hidden_activations = ['relu'],
                 hidden_dropouts = [0.2],
                 hidden_batchnorm = [False],
                 conv_cores = [10],
                 conv_core_sizes = [4],
                 conv_strides = [2],
                 conv_activations = ['relu'],
                 conv_dropouts = [0.2],
                 conv_pooling = ['max'],
                 conv_batchnorm = [False],
                 *args, **kwargs):
        self.hidden_sizes = hidden_sizes
        self.hidden_activations = hidden_activations
        self.hidden_dropouts = hidden_dropouts
        self.hidden_batchnorm = hidden_batchnorm
        self.conv_cores = conv_cores
        self.conv_core_sizes = conv_core_sizes
        self.conv_strides = conv_strides
        self.conv_activations = conv_activations
        self.conv_dropouts = conv_dropouts
        self.conv_pooling = conv_pooling
        self.conv_batchnorm = conv_batchnorm
        super(ConvAndDense, self).__init__(*args, **kwargs)

    def _build_inner_model(self, input_layer):
        h = input_layer

        is_theano = K.image_dim_ordering() == 'th'
        if len(self.input_shape) == 2:
            if is_theano: # theano is (layers, rows, cols)
                reshape_input_to = (1,) + self.input_shape
            else:
                reshape_input_to = self.input_shape + (1,) # tensorflow is (rows, cols, layers)
            h = Reshape(reshape_input_to)(h)

        bn_axis = 0 if is_theano else 2
        for cores, core_size, stride, act, dropout, pool, bn in itertools.izip_longest(self.conv_cores,
                                                                                       self.conv_core_sizes,
                                                                                       self.conv_strides,
                                                                                       self.conv_activations,
                                                                                       self.conv_dropouts,
                                                                                       self.conv_pooling,
                                                                                       self.conv_batchnorm):
            h = Convolution2D(cores,
                              core_size,
                              core_size,
                              subsample = (stride, stride))(h)
            if not dropout is None and 0 < dropout < 1:
                h = Dropout(dropout)(h)
            if bn:
                h = BatchNormalization(axis = bn_axis)(h)
            if pool in ('max', 'min'):
                h = (MaxPooling2D if pool == 'max' else AveragePooling2D)()(h)
            h = Activation(act)(h)
        h = Flatten()(h)

        for size, act, dropout, bn in itertools.izip_longest(self.hidden_sizes,
                                                             self.hidden_activations,
                                                             self.hidden_dropouts,
                                                             self.hidden_batchnorm):
            h = Dense(size)(h)
            if not dropout is None and 0 < dropout < 1:
                h = Dropout(dropout)(h)
            if bn:
                h = BatchNormalization()(h)
            h = Activation(act)(h)

        return h


DEFAULT_DEEP_NESTED_MODEL = { 'ctor' : 'keras.applications.vgg16.VGG16',
                             'kwargs' : { 'include_top' : False,
                                         'weights' : 'imagenet' } }
DEFAULT_SCALE_TARGET_SHAPE = (224, 224)
class DeepPreproc(object):
    def __init__(self,
                 nested_model = DEFAULT_DEEP_NESTED_MODEL,
                 scale_target_shape = DEFAULT_SCALE_TARGET_SHAPE,
                 *args, **kwargs):
        self.nested_model = nested_model
        self.scale_target_shape = scale_target_shape
        self.scale_factor = None
        super(DeepPreproc, self).__init__(*args, **kwargs)

    def _build_model(self):
        self.src_input_shape = self.input_shape
        if len(self.input_shape) == 2:
            self.input_shape = self.scale_target_shape
        elif len(self.input_shape) == 3:
            if get_backend() == 'tf':
                self.input_shape = self.scale_target_shape + (self.src_input_shape[-1],)
            else:
                self.input_shape = (self.src_input_shape[0],) + self.scale_target_shape
        else:
            raise NotImplementedError()

        self.scale_factor = (1, ) + tuple(numpy.array(self.input_shape, dtype = 'float') / numpy.array(self.src_input_shape))

        super(DeepPreproc, self)._build_model()

    def _build_inner_model(self, h):
        dim_order = get_backend()
        if len(self.input_shape) == 2:
            if dim_order == 'tf':
                h = Reshape(self.input_shape + (1,))(h)
                h = Lambda(lambda t: K.repeat_elements(t, 3, 3))(h)
            else:
                h = Reshape((1, ) + self.input_shape)(h)
                h = Lambda(lambda t: K.repeat_elements(t, 3, 1))(h)
            h = load_object_from_dict(self.nested_model)(h)

        elif len(self.input_shape) == 3:
            if dim_order == 'tf':
                images_number = self.input_shape[-1]
                image_shape = self.input_shape[:-1]

                h = Permute((3, 1, 2))(h)
                h = Reshape((images_number,) + image_shape + (1,))(h)
                h = Lambda(lambda t: K.repeat_elements(t, 3, 4))(h)
            else:
                images_number = self.input_shape[0]
                image_shape = self.input_shape[1:]

                h = Reshape((images_number, 1) + image_shape)(h)
                h = Lambda(lambda t: K.repeat_elements(t, 3, 2))(h)
            h = TimeDistributed(load_object_from_dict(self.nested_model))(h)
        else:
            raise NotSupportedError()

        h = Flatten()(h)
        h = Dense(4096, activation = 'relu')(h)
        h = Dense(4096, activation = 'relu')(h)

        return h

    def _gen_train_val_data_from_memory(self):
        train_gen, val_gen = super(DeepPreproc, self)._gen_train_val_data_from_memory()
        return scale_tensors(train_gen, self.scale_factor, 3), scale_tensors(val_gen, self.scale_factor, 3)


def inception_conv(input_tensor, n, filters1, filters2, filters31, filters32, filters41, filters42, filters43):
    path1 = Convolution2D(filters1, 1, 1, border_mode = 'same')(input_tensor)
    path1 = BatchNormalization()(path1)
    path1 = Activation('relu')(path1)

    path2 = MaxPooling2D(strides = (1, 1), border_mode = 'same')(input_tensor)
    path2 = Convolution2D(filters2, 1, 1, border_mode = 'same')(path2)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)

    path3 = Convolution2D(filters31, 1, 1, border_mode = 'same')(input_tensor)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)
    path3 = Convolution2D(filters32, n, 1, border_mode = 'same')(path3)
    path3 = BatchNormalization()(path3)
    path3 = Convolution2D(filters32, 1, n, border_mode = 'same')(path3)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)

    path4 = Convolution2D(filters41, 1, 1, border_mode = 'same')(input_tensor)
    path4 = BatchNormalization()(path4)
    path4 = Activation('relu')(path4)
    path4 = Convolution2D(filters42, n, 1, border_mode = 'same')(path4)
    path4 = BatchNormalization()(path4)
    path4 = Convolution2D(filters42, 1, n, border_mode = 'same')(path4)
    path4 = BatchNormalization()(path4)
    path4 = Activation('relu')(path4)
    path4 = Convolution2D(filters43, n, 1, border_mode = 'same')(path4)
    path4 = BatchNormalization()(path4)
    path4 = Convolution2D(filters43, 1, n, border_mode = 'same')(path4)
    path4 = BatchNormalization()(path4)
    path4 = Activation('relu')(path4)
    
    return merge([path1, path2, path3, path4], mode = 'concat')


def inception_subsample(input_tensor, n, filters11, filters12, filters31, filters32, filters33):
    path1 = MaxPooling2D(strides = (2, 2), border_mode = 'same')(input_tensor)

    path2 = Convolution2D(filters11, 1, 1, border_mode = 'same')(input_tensor)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)
    path2 = Convolution2D(filters12, n, 1, subsample = (2, 1), border_mode = 'same')(path2)
    path2 = BatchNormalization()(path2)
    path2 = Convolution2D(filters12, 1, n, subsample = (1, 2), border_mode = 'same')(path2)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)

    path3 = Convolution2D(filters31, 1, 1, border_mode = 'same')(input_tensor)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)
    path3 = Convolution2D(filters32, n, 1, border_mode = 'same')(path3)
    path3 = BatchNormalization()(path3)
    path3 = Convolution2D(filters32, 1, n, border_mode = 'same')(path3)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)
    path3 = Convolution2D(filters33, n, 1, subsample = (2, 1), border_mode = 'same')(path3)
    path3 = BatchNormalization()(path3)
    path3 = Convolution2D(filters33, 1, n, subsample = (1, 2), border_mode = 'same')(path3)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)
    
    return merge([path1, path2, path3], mode = 'concat')


DEFAULT_INCEPTION_STRUCTURE = [
    dict(type = 'conv', n = 3, filters1 = 64, filters2 = 64, filters31 = 64, filters32 = 64, filters41 = 64, filters42 = 64, filters43 = 64),
    dict(type = 'conv', n = 3, filters1 = 64, filters2 = 64, filters31 = 64, filters32 = 64, filters41 = 64, filters42 = 64, filters43 = 64),
    dict(type = 'subs', n = 3, filters11 = 64, filters12 = 64, filters31 = 64, filters32 = 64, filters33 = 64),
    dict(type = 'subs', n = 3, filters11 = 64, filters12 = 64, filters31 = 64, filters32 = 64, filters33 = 64)
]

class Inception(object):
    def __init__(self,
                 structure = DEFAULT_INCEPTION_STRUCTURE,
                 *args, **kwargs):
        self.structure = structure
        super(Inception, self).__init__(*args, **kwargs)

    def _build_inner_model(self, h):
        dim_order = get_backend()
        if len(self.input_shape) == 2:
            if dim_order == 'tf':
                h = Reshape(self.input_shape + (1,))(h)
            else:
                h = Reshape((1, ) + self.input_shape)(h)
            h = self._build_inception_model()(h)
        elif len(self.input_shape) == 3:
            if dim_order == 'tf':
                images_number = self.input_shape[-1]
                image_shape = self.input_shape[:-1]

                h = Permute((3, 1, 2))(h)
                h = Reshape((images_number,) + image_shape + (1,))(h)
            else:
                images_number = self.input_shape[0]
                image_shape = self.input_shape[1:]

                h = Reshape((images_number, 1) + image_shape)(h)
            h = TimeDistributed(self._build_inception_model())(h)
        else:
            raise NotSupportedError()

        return h

    def _build_inception_model(self):
        dim_order = get_backend()
        if len(self.input_shape) == 2:
            if dim_order == 'tf':
                input_layer_shape = self.input_shape + (1,)
            else:
                input_layer_shape = (1,) + self.input_shape
        elif len(self.input_shape) == 3:
            if dim_order == 'tf':
                input_layer_shape = self.input_shape[:-1] + (1,)
            else:
                input_layer_shape = (1,) + self.input_shape[1:]
        else:
            raise NotSupportedError()

        input_layer = Input(shape = input_layer_shape)
        h = input_layer
        for block_info in self.structure:
            block_type = block_info['type']
            args = dict(block_info)
            del args['type']

            if block_type == 'conv':
                h = inception_conv(h, **args)
            elif block_type == 'subs':
                h = inception_subsample(h, **args)

        return Model([input_layer], [h])
