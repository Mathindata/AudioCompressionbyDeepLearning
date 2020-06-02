
import logging
import scipy.signal
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.collections import LineCollection
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from keras import backend as K, initializers
from keras.engine.topology import Layer, InputSpec
from keras.layers.convolutional import Conv2DTranspose, Conv1D
from keras.engine.training import Model
from keras.engine.input_layer import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import RNN
from keras.utils import conv_utils

logger = logging.getLogger(__name__)


def findNextDivisible(n, d):
    return int(d * np.ceil(n / float(d)))


# TODO: maybe use tf.fake_quant_with_min_max_args which implements similar idea and handles gradient
# see: https://www.tensorflow.org/api_docs/python/tf/fake_quant_with_min_max_args
def floatAsInt(x, nbits=16):
    x = K.clip(x, -1.0, 1.0)
    x *= 2**(nbits - 1) - 1
    x = K.round(x)
    return x


def floatAsIntToFloat(x, nbits=16):
    x /= 2**(nbits - 1) - 1
    x = K.clip(x, -1.0, 1.0)
    return x


def categorical_crossentropy(y_true, y_pred):
    # NOTE: take the mean so that the scale of the gradient does not depend on the length of the inputs, nor the batch size
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


# FIXME: need a DPCM architecture
#  see: https://web.stanford.edu/class/ee398a/handouts/lectures/06-Prediction.pdf
#       http://iphome.hhi.de/wiegand/assets/pdfs/DIC_predictive_coding_07.pdf

# see: 'Speech Coding Algorithms: Foundation and Evolution of Standardized Coders'
#     DPCM with MA Prediction
#     http://dsp-book.narod.ru/ChuSCA.pdf


class PredictionError(Layer):

    class PredictionCell(Layer):
        def __init__(self, pred, **kwargs):
            self.pred = pred
            self.state_size = tuple([pred.kernel_size] * pred.nbChannels)
            Layer.__init__(self, **kwargs)

        def call(self, inputs, states):
            # Shift history buffer and append newest input at the end
            history = tf.stack(states, axis=-1)
            x = self.pred._predict(history, padding='valid')
            x = tf.squeeze(x, axis=1)

            # Add error signal to prediction
            x = floatAsIntToFloat(floatAsInt(x) + inputs)

            # Store new output sample in history
            history = tf.manip.roll(history, shift=-1, axis=1)
            history = tf.concat([history[:, :-1, :], tf.expand_dims(x, axis=1)], axis=1)

            return x, tf.unstack(history, axis=-1)

    class PredictionErrorRNN(RNN):

        def __init__(self, pred, **kwargs):
            self.pred = pred
            RNN.__init__(self, PredictionError.PredictionCell(pred), return_sequences=True, return_state=False, **kwargs)

        def call(self, inputs, mask=None, training=None, initial_state=None):
            self.cell._dropout_mask = None
            self.cell._recurrent_dropout_mask = None
            return RNN.call(self,
                            inputs,
                            mask=mask,
                            training=training,
                            initial_state=initial_state)

    def __init__(self, nbChannels, kernel_size, **kwargs):
        self.__dict__.update(nbChannels=nbChannels, kernel_size=kernel_size)
        super(PredictionError, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(self.kernel_size, input_dim, input_dim),
                                      initializer=initializers.get('zeros'),
                                      name='kernel')
        self.bias = self.add_weight(shape=(input_dim,),
                                    initializer=initializers.get('zeros'),
                                    name='bias')

        Layer.build(self, input_shape)

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size}
        base_config = super(PredictionError, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _predict(self, x, padding='causal'):
        x = K.conv1d(x, self.kernel, strides=1, padding=padding)
        x = K.bias_add(x, self.bias)
        return x

    def call(self, x):
        xq = floatAsIntToFloat(floatAsInt(x))

        # Predict the input at time t based on past samples at time t-1, t-2, ... t-N
        xqs = tf.pad(xq[:, :-1, :], paddings=[(0, 0), (1, 0), (0, 0)], mode='CONSTANT', constant_values=0.0)
        prediction = self._predict(K.stop_gradient(xqs), padding='causal')

        # Compute the prediction error
        error = floatAsInt(xq) - floatAsInt(prediction)

        self.add_loss(K.mean(K.square(x - prediction), axis=-1), x)

        return error

    def getReconstructionLayer(self):
        return PredictionError.PredictionErrorRNN(self)


class SmoothEmbedding(Model):

    def __init__(self, nbChannels, sampleRate, nbFilters=32, kernelSize=8, stride=2, embeddingRate=30.0, nbEmbeddingDim=32, **kwargs):
        self.__dict__.update(nbChannels=nbChannels, sampleRate=sampleRate,
                             nbFilters=nbFilters, kernelSize=kernelSize, stride=stride,
                             embeddingRate=embeddingRate, nbEmbeddingDim=nbEmbeddingDim)

        x = Input(shape=(None, nbChannels))
        dsConvs, usConvs, usConvOut = self._initInternalLayers()
        self.nbStackedLayers = len(dsConvs)
        xr, z = self._reconstruct(x, dsConvs, usConvs, usConvOut)

        Model.__init__(self, x, [xr, z], **kwargs)

        self.nbStackedLayers = len(dsConvs)
        self.dsConvs = dsConvs
        self.usConvs = usConvs
        self.usConvOut = usConvOut

    def _initInternalLayers(self):
        i = 0
        fs = self.sampleRate
        dsConvs = []
        usConvs = []
        usConvOut = Conv1DTranspose(self.nbChannels, self.kernelSize, stride=1, activation='linear', padding='same', name='embedding-conv-out')
        while fs > self.embeddingRate:
            # Downsampling with strided convolutions
            fs = int(np.ceil(fs / self.stride))
            dsConv = Conv1D(self.nbFilters, kernel_size=self.kernelSize, strides=self.stride, activation='linear', padding='same', name='embedding-conv-ds-%d' % (i + 1))
            dsConvs.append(dsConv)

            # Upsampling with strided convolutions
            usConv = Conv1DTranspose(self.nbFilters, kernel_size=self.kernelSize, stride=self.stride, activation='linear', padding='same', name='embedding-conv-us-%d' % (i + 1))
            usConvs.append(usConv)

            i += 1

        return dsConvs, usConvs, usConvOut

    def asEncodingOnly(self, n=None):
        if n is None:
            n = len(self.dsConvs) - 1
        x = Input(shape=(None, self.nbChannels))
        z = x
        for dsConv in self.dsConvs:
            z = dsConv(z)
            z = BatchNormalization()(z)
            z = LeakyReLU()(z)
        return Model(x, z)

    def asDecodingOnly(self, n=None):
        if n is None:
            n = len(self.dsConvs) - 1
        z = Input(shape=(None, self.nbEmbeddingDim))
        xr = z
        for usConv in reversed(self.usConvs):
            xr = usConv(xr)
            xr = BatchNormalization()(xr)
            xr = LeakyReLU()(xr)
        xr = self.usConvOut(xr)
        return Model(z, xr)

    def _reconstruct(self, x, dsConvs, usConvs, usConvOut, n=None):
        if n is None:
            n = len(dsConvs) - 1
        z = x
        for dsConv in dsConvs[:n + 1]:
            z = dsConv(z)
            z = BatchNormalization()(z)
            z = LeakyReLU()(z)

        # Upsampling
        xr = z
        for usConv in reversed(usConvs[:n + 1]):
            xr = usConv(xr)
            xr = BatchNormalization()(xr)
            xr = LeakyReLU()(xr)
        xr = usConvOut(xr)

        return xr, z

    def asLimitedToLayerPretraining(self, n=None):
        if n is None:
            n = len(self.dsConvs) - 1
        assert n < len(self.dsConvs)
        x = Input(shape=(None, self.nbChannels))
        xr, _ = self._reconstruct(x, self.dsConvs, self.usConvs, self.usConvOut, n)
        return Model(x, xr)

    def get_config(self):
        config = {'nbChannels': self.nbChannels,
                  'sampleRate': self.sampleRate,
                  'nbFilters': self.nbFilters,
                  'kernelSize': self.kernelSize,
                  'stride': self.stride,
                  'embeddingRate': self.embeddingRate,
                  'nbEmbeddingDim': self.nbEmbeddingDim}
        base_config = super(SmoothEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SmoothEnvelope(Layer):

    def __init__(self, fs, envelopeFs, stride=2, upsample=True, **kwargs):
        self.__dict__.update(fs=fs, envelopeFs=envelopeFs, stride=stride, upsample=upsample)
        super(SmoothEnvelope, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'fs': self.fs,
                  'envelopeFs': self.envelopeFs,
                  'stride': self.stride,
                  'upsample': self.upsample}
        base_config = super(SmoothEnvelope, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _getEnvelope(self, x):
        length = tf.shape(x)[1]

        nbStackedLayers = 0
        fs = self.fs
        while fs > self.envelopeFs:
            fs = int(np.ceil(fs / self.stride))
            nbStackedLayers += 1
        self.nbStackedLayers = nbStackedLayers

        # NOTE: make sure the data can be downsampled correctly
        factor = tf.constant(self.stride ** nbStackedLayers, dtype=tf.float32)
        maximumLength = tf.cast(tf.cast(length, tf.float32) * tf.ceil(factor / tf.cast(length, tf.float32)), dtype=tf.int32)
        x = tf.pad(x, paddings=[(0, 0), (0, maximumLength - length), (0, 0)], mode='CONSTANT', constant_values=0.0)

        # Downsampling using strided maxpooling to keep peak amplitudes information
        for _ in range(nbStackedLayers):
            x = MaxPooling1D(pool_size=3, strides=self.stride, padding='same')(x)

        if self.upsample:
            # X must have shape: [batch, height, width, channels]
            x = tf.squeeze(tf.image.resize_images(tf.expand_dims(x, 2), size=tf.stack([maximumLength, 1]), method=ResizeMethod.BILINEAR, align_corners=True), axis=2)

            # Apply correction for the phase shift introduced by maxpooling
            dr = int((self.stride ** nbStackedLayers) / 2)
            x = tf.pad(x[:, :-dr, :], paddings=[(0, 0), (dr, 0), (0, 0)], mode='CONSTANT', constant_values=0.0)

            # Remove the padding
            x = x[:, :length, :]

        return x

    def call(self, x):

        # Upper envelope
        xu = self._getEnvelope(tf.clip_by_value(x, 0.0, np.inf))

        # Lower envelope
        xl = -self._getEnvelope(tf.clip_by_value(-x, 0.0, np.inf))

        return [xu, xl]

    def compute_output_shape(self, input_shape):
        if self.upsample:
            output_shape = input_shape
        else:
            fs = self.fs
            length = input_shape[1]
            while fs > self.envelopeFs:
                fs = int(np.ceil(fs / self.stride))
                length = conv_utils.conv_output_length(length, filter_size=3, padding='same', stride=self.stride)
            output_shape = (input_shape[0], length, input_shape[2])

        return [output_shape, output_shape]


class SmoothLogEnergyProfile(Layer):

    def __init__(self, fs, logEnergyFs, stride=2, offset=False, upsample=True, **kwargs):
        self.__dict__.update(fs=fs, logEnergyFs=logEnergyFs, stride=stride, offset=offset, upsample=upsample)
        super(SmoothLogEnergyProfile, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'fs': self.fs,
                  'logEnergyFs': self.logEnergyFs,
                  'stride': self.stride,
                  'offset': self.offset,
                  'upsample': self.upsample}
        base_config = super(SmoothLogEnergyProfile, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _getLogEnergy(self, x):
        length = tf.shape(x)[1]

        nbStackedLayers = 0
        fs = self.fs
        while fs > self.logEnergyFs:
            fs = int(np.ceil(fs / self.stride))
            nbStackedLayers += 1
        self.nbStackedLayers = nbStackedLayers

        # NOTE: make sure the data can be downsampled correctly
        factor = tf.constant(self.stride ** nbStackedLayers, dtype=tf.float32)
        maximumLength = tf.cast(tf.cast(length, tf.float32) * tf.ceil(factor / tf.cast(length, tf.float32)), dtype=tf.int32)
        x = tf.pad(x, paddings=[(0, 0), (0, maximumLength - length), (0, 0)], mode='CONSTANT', constant_values=0.0)

        # Calculate energy sample-wise
        x = K.square(x)

        # Downsampling using strided maxpooling to keep peak amplitudes information
        for _ in range(nbStackedLayers):
            x = AveragePooling1D(pool_size=3, strides=self.stride, padding='same')(x)

        x = K.log(x + K.epsilon())

        if self.offset:
            x -= K.log(K.epsilon())

        if self.upsample:
            # X must have shape: [batch, height, width, channels]
            x = tf.squeeze(tf.image.resize_images(tf.expand_dims(x, 2), size=tf.stack([maximumLength, 1]), method=ResizeMethod.BILINEAR, align_corners=True), axis=2)

            # Apply correction for the phase shift introduced by maxpooling
            dr = int((self.stride ** nbStackedLayers) / 2)
            x = tf.pad(x[:, :-dr, :], paddings=[(0, 0), (dr, 0), (0, 0)], mode='CONSTANT', constant_values=0.0)

            # Remove the padding
            x = x[:, :length, :]

        return x

    def call(self, x):
        x = self._getLogEnergy(x)
        return x

    def compute_output_shape(self, input_shape):
        if self.upsample:
            output_shape = input_shape
        else:
            fs = self.fs
            length = input_shape[1]
            while fs > self.logEnergyFs:
                fs = int(np.ceil(fs / self.stride))
                length = conv_utils.conv_output_length(length, filter_size=3, padding='same', stride=self.stride)
            output_shape = (input_shape[0], length, input_shape[2])

        return output_shape


class SmoothMeanProfile(Layer):

    def __init__(self, fs, meanFs, stride=2, upsample=True, **kwargs):
        self.__dict__.update(fs=fs, meanFs=meanFs, stride=stride, upsample=upsample)
        super(SmoothMeanProfile, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'fs': self.fs,
                  'meanFs': self.meanFs,
                  'stride': self.stride,
                  'upsample': self.upsample}
        base_config = super(SmoothMeanProfile, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _getMean(self, x):
        length = tf.shape(x)[1]

        nbStackedLayers = 0
        fs = self.fs
        while fs > self.meanFs:
            fs = int(np.ceil(fs / self.stride))
            nbStackedLayers += 1
        self.nbStackedLayers = nbStackedLayers

        # NOTE: make sure the data can be downsampled correctly
        factor = tf.constant(self.stride ** nbStackedLayers, dtype=tf.float32)
        maximumLength = tf.cast(tf.cast(length, tf.float32) * tf.ceil(factor / tf.cast(length, tf.float32)), dtype=tf.int32)
        x = tf.pad(x, paddings=[(0, 0), (0, maximumLength - length), (0, 0)], mode='CONSTANT', constant_values=0.0)

        # Downsampling using strided maxpooling to keep peak amplitudes information
        for _ in range(nbStackedLayers):
            x = AveragePooling1D(pool_size=3, strides=self.stride, padding='same')(x)

        if self.upsample:
            # X must have shape: [batch, height, width, channels]
            x = tf.squeeze(tf.image.resize_images(tf.expand_dims(x, 2), size=tf.stack([maximumLength, 1]), method=ResizeMethod.BILINEAR, align_corners=True), axis=2)

            # Apply correction for the phase shift introduced by maxpooling
            dr = int((self.stride ** nbStackedLayers) / 2)
            x = tf.pad(x[:, :-dr, :], paddings=[(0, 0), (dr, 0), (0, 0)], mode='CONSTANT', constant_values=0.0)

            # Remove the padding
            x = x[:, :length, :]

        return x

    def call(self, x):
        x = self._getMean(x)
        return x

    def compute_output_shape(self, input_shape):
        if self.upsample:
            output_shape = input_shape
        else:
            fs = self.fs
            length = input_shape[1]
            while fs > self.meanFs:
                fs = int(np.ceil(fs / self.stride))
                length = conv_utils.conv_output_length(length, filter_size=3, padding='same', stride=self.stride)
            output_shape = (input_shape[0], length, input_shape[2])

        return output_shape


class Preemphasis(Layer):

    def __init__(self, coefficient=0.97, **kargs):
        self.__dict__.update(coefficient=coefficient)
        super(Preemphasis, self).__init__(**kargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'coefficient': self.coefficient}
        base_config = super(Preemphasis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        return K.concatenate([x[:, 0:1, :], x[:, 1:, :] - self.coefficient * x[:, :-1, :]], axis=1)


class OuterMiddleEarFilter(Layer):

    '''
    Based on Huber et al. Ann.Otol.Rhinol.Laryngol. 110 31-35 (2001)
    <gain>  <lower cutoff>  <upper cutoff>  <filter order>
    '''
    human_data_Huber = np.array([
        [0, 1300, 3100, 1],
        [-13, 4000, 6000, 1],
    ])

    '''
    Based on Ruggero
    <gain>  <lower cutoff>  <upper cutoff>  <filter order>
    '''
    human_data_Ruggero = np.array([
        [-2, 1900, 4200, 1],
        [-3, 4500, 6300, 1],
        [-19, 8000, 12000, 1]
    ])

    human_data_generic = np.array([
        [1, 700, 1200, 1],
    ])

    def __init__(self, fs, filterSize=257, mode='ruggero', **kargs):
        if filterSize is not None:
            if filterSize % 2 == 0:
                filterSize += 1
                logger.warning(
                    'Length of the FIR filter adjusted to the next odd number to ensure symmetry: %d' % (filterSize))

        self.__dict__.update(fs=fs, filterSize=filterSize, mode=mode)
        super(OuterMiddleEarFilter, self).__init__(**kargs)
        self.input_spec = InputSpec(ndim=3)

        if mode == 'ruggero':
            filterData = OuterMiddleEarFilter.human_data_Ruggero
        elif mode == 'huber':
            filterData = OuterMiddleEarFilter.human_data_Huber
        elif mode == 'generic':
            filterData = OuterMiddleEarFilter.human_data_generic
        else:
            raise Exception('Unsupported mode: %s' % (mode))

        self.filters = self._getFilterCoefficients(filterData)

    def get_config(self):
        config = {'fs': self.fs,
                  'filterSize': self.filterSize,
                  'mode': self.mode}
        base_config = super(OuterMiddleEarFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _getFilterCoefficients(self, filterData):
        filters = []
        for i in range(len(filterData)):
            # Band-pass filter
            gaindB, cutoff_low, cutoff_high, order = filterData[i]
            assert order == 1

            cutoff_low = cutoff_low / (self.fs / 2)
            cutoff_high = cutoff_high / (self.fs / 2)
            if cutoff_low >= 1.0 or cutoff_high >= 1.0:
                logger.warning('Ignoring bandpass filter because sampling frequency (%0.3f Hz) is too low: %f dB, cutoff low = %0.3f Hz, cutoff high = %0.3f Hz' % (self.fs, gaindB, cutoff_low * (self.fs / 2), cutoff_high * (self.fs / 2)))
                continue

            gainLinear = 10.0 ** (gaindB / 20.0)
            b = scipy.signal.firwin(self.filterSize, [cutoff_low, cutoff_high], pass_zero=False)
            filters.append(b * gainLinear)
        return filters

    def call(self, x):
        # Filters can be grouped together
        # NOTE: kernel has shape [length, dim_in, dim_out]
        kernel = K.constant(np.transpose(np.array(self.filters, dtype=np.float32))[:, np.newaxis, :])

        # TODO: replicate the kernel on the channel axis to handle multichannel cases?
        # kernel = tf.tile(kernel, tf.stack([1, tf.shape(x)[-1], 1]))
        x = K.conv1d(x, kernel, strides=1, padding='same', data_format='channels_last')
        x = K.sum(x, axis=-1, keepdims=True)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)


def erbspace(low, high, N, earQ=9.26449, minBW=24.7):
    low = float(low)
    high = float(high)
    cf = -(earQ * minBW) + np.exp((np.arange(N)) * (-np.log(high + earQ * minBW) +
                                                    np.log(low + earQ * minBW)) / (N - 1)) * (high + earQ * minBW)
    cf = cf[::-1]
    return cf


def melspace(low, high, N):
    warpingFunc = (lambda freq: 2595.0 * np.log10(1.0 + freq / 700.0))
    unwarpingFunc = (lambda m: 700.0 * ((10 ** (m / 2595.0)) - 1.0))

    # Perform uniform sampling in the mel-scale
    melLow = warpingFunc(float(low))
    melHigh = warpingFunc(float(high))
    mels = np.linspace(melLow, melHigh, N)

    return unwarpingFunc(mels)


def universalWarpingFunctionSpace(low, high, N):

    _region_data = np.array([
        [1, 240, 6.0],
        [240, 550, 4.3869],
        [550, 1280, 2.4629],
        [1280, 3000, 1.4616],
        [3000, 8001, 1.0]
    ])

    _curve_data = np.array([
        [20, 0],
        [90.5753, 171.167],
        [104.405, 197.227],
        [120.341, 232.189],
        [141.19, 275.972],
        [162.74, 310.934],
        [181.035, 337.155],
        [208.656, 381.019],
        [236.277, 407.16],
        [267.527, 451.105],
        [302.909, 495.05],
        [349.106, 547.816],
        [431.986, 609.161],
        [472.055, 644.366],
        [534.431, 706.116],
        [615.807, 794.491],
        [709.575, 882.867],
        [832.245, 980.064],
        [893.316, 1033.15],
        [1066.44, 1148.08],
        [1185.89, 1236.61],
        [1295.47, 1325.23],
        [1440.57, 1413.77],
        [1659.49, 1546.66],
        [1780.88, 1635.36],
        [2015.67, 1741.62],
        [2241.33, 1839.06],
        [2491.98, 1954.3],
        [2770.37, 2087.35],
        [2972.4, 2211.66],
        [3189.51, 2318.16],
        [3362.33, 2415.85],
        [3737.75, 2557.8],
        [3871.44, 2628.85],
        [4154.87, 2708.65],
        [4458.58, 2806.25],
        [4784.49, 2903.85],
        [4955.36, 2983.81],
        [5411.84, 3116.94],
        [6342.08, 3356.58],
        [6566.49, 3489.95],
        [7000.00, 3500.00],
        [8000.00, 3700.00]
    ])

    x = _curve_data[:, 0]
    y = _curve_data[:, 1]
    warpingFunc = scipy.interpolate.interp1d(x, y, kind='linear')
    unwarpingFunc = scipy.interpolate.interp1d(y, x, kind='linear')

    # Perform uniform sampling in the warped-scale
    vLow = warpingFunc(low)
    vHigh = warpingFunc(high)
    vs = np.linspace(vLow, vHigh, N)

    return unwarpingFunc(vs)


class FilterBank(Layer):

    def __init__(self, nbFilters=64, filterSize=None, minFilterSize=257, fs=16000.0, cfmin=20.0, cfmax=8000.0, warping='linear', **kwargs):

        if filterSize is not None:
            if filterSize % 2 == 0:
                filterSize += 1
                logger.warning(
                    'Length of the FIR filter adjusted to the next odd number to ensure symmetry: %d' % (filterSize))

        self.__dict__.update(nbFilters=nbFilters, filterSize=filterSize, minFilterSize=minFilterSize, fs=fs,
                             cfmin=cfmin, cfmax=cfmax, warping=warping)
        super(FilterBank, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

        cf = self._computeCentralFrequencies()
        assert (cf >= 0.0).all() and (cf <= self.fs / 2).all()
        self.cf = cf

        bw = self._computeBandwiths(cf)
        assert len(bw) == len(cf)
        assert (bw >= 0.0).all()
        self.bw = bw

        if self.filterSize is None:
            # Calculate the number of taps for each filter
            # The filter should cover at least 16 periods of the signal, or 257 samples
            n = np.array(16 * self.fs / cf, dtype=np.int)
            n[n < self.minFilterSize] = self.minFilterSize
            n[n % 2 == 0] += 1
        else:
            n = np.tile(self.filterSize, len(cf)).astype(np.int)
        self.n = n

        self.filters = self._getFilterCoefficients(cf, bw, n)

    def get_config(self):
        config = {'nbFilters': self.nbFilters,
                  'filterSize': self.filterSize,
                  'minFilterSize': self.minFilterSize,
                  'fs': self.fs,
                  'cfmin': self.cfmin,
                  'cfmax': self.cfmax,
                  'warping': self.warping}
        base_config = super(FilterBank, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _computeCentralFrequencies(self):
        # Compute the central frequencies of the filters
        if self.warping is None or self.warping == 'linear':
            cf = np.linspace(self.cfmin, self.cfmax, self.nbFilters)
        elif self.warping == 'erbspace':
            cf = erbspace(self.cfmin, self.cfmax, self.nbFilters)
        elif self.warping == 'melspace':
            cf = melspace(self.cfmin, self.cfmax, self.nbFilters)
        elif self.warping == 'universal':
            cf = universalWarpingFunctionSpace(self.cfmin, self.cfmax, self.nbFilters)
        else:
            raise Exception('Unsupported warping: %s' % (self.warping))
        cf = np.array(cf, np.float32)
        return cf

    def _computeBandwiths(self, cf):
        bw = np.diff(cf) / 2
        bw = np.concatenate((bw, [bw[-1]]), axis=0)
        return bw

    def _getFilterCoefficients(self, cf, bw, n):

        filters = []
        for i in range(len(cf)):
            if i == 0:
                # Low-pass filter
                cutoff = (cf[i] + bw[i]) / (self.fs / 2)
                b = scipy.signal.firwin(n[i], cutoff=cutoff, window='hamming')
            elif i == len(cf) - 1:
                # High-pass filter
                cutoff = (cf[i - 1] + bw[i - 1]) / (self.fs / 2)
                b = scipy.signal.firwin(
                    n[i], cutoff=cutoff, window='hamming', pass_zero=False)
            else:
                # Band-pass filter
                cutoff_low = (cf[i - 1] + bw[i - 1]) / (self.fs / 2)
                cutoff_high = (cf[i] + bw[i]) / (self.fs / 2)
                b = scipy.signal.firwin(
                    n[i], [cutoff_low, cutoff_high], pass_zero=False)
            filters.append(b)

        return filters

    def call(self, x):

        if self.filterSize is None:
            # Multiple independent filters
            for b in self.filters:
                b = np.array(b, dtype=np.float32)
        else:
            # Filters can be grouped together
            # NOTE: kernel has shape [length, dim_in, dim_out]
            kernel = K.constant(np.transpose(np.array(self.filters, dtype=np.float32))[:, np.newaxis, :])

            # TODO: replicate the kernel on the channel axis to handle multichannel cases?
            # kernel = tf.tile(kernel, tf.stack([1, tf.shape(x)[-1], 1]))
            x = K.conv1d(x, kernel, strides=1, padding='same', data_format='channels_last')

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.nbFilters)

    def display(self, merged=False):
        # Adapted from: http://mpastell.com/2010/01/18/fir-with-scipy/

        if not merged:
            fig = plt.figure(figsize=(8, 6), facecolor='white', frameon=True)
            for b in self.filters:
                w, h = scipy.signal.freqz(b, 1)
                h_dB = 20 * np.log10(abs(h))
                plt.subplot(211)
                plt.plot(w / max(w), h_dB)
                plt.ylim(-150, 5)
                plt.ylabel('Magnitude (db)')
                plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
                plt.title(r'Frequency response')
                plt.subplot(212)
                h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
                plt.plot(w / max(w), h_Phase)
                plt.ylabel('Phase (radians)')
                plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
                plt.title(r'Phase response')
                plt.subplots_adjust(hspace=0.5)
        else:
            fig = plt.figure(figsize=(8, 6), facecolor='white', frameon=True)
            hs = []
            for b in self.filters:
                w, h = scipy.signal.freqz(b, 1)
                hs.append(h)
            h = np.sum(np.array(hs), axis=0)
            h_dB = 20 * np.log10(abs(h))
            plt.plot(w / max(w), h_dB)
            plt.ylim(-150, 5)
            plt.ylabel('Magnitude (db)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Frequency response')

        return fig


class GammatoneFilterbank(FilterBank):
    ''' Gammatone filterbank implemented as FIR filter, with phase compensation across channels. Non-causal filter. '''

    def __init__(self, nbFilters=64, filterSize=None, minFilterSize=257, fs=16000.0, cfmin=20.0, cfmax=8000.0, warping='melspace', order=3):
        self.__dict__.update(order=order)
        super(GammatoneFilterbank, self).__init__(nbFilters, filterSize, minFilterSize, fs, cfmin, cfmax, warping)

    def get_config(self):
        config = {'order': self.order}
        base_config = super(GammatoneFilterbank, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _computeBandwiths(self, cf):
        # Compute the bandwidth of the filters
        bw = 10 ** (0.03728 + 0.78563 * np.log10(cf))
        return bw

    def _getFilterCoefficients(self, cf, bw, n):
        filters = []
        dt = 1.0 / self.fs
        for i in range(len(cf)):
            # Phase compensated impulse response
            tc = (self.order - 1) / (2 * np.pi * bw[i])
            t = np.linspace(-tc, -tc + n[i] * dt, n[i])
            b = (t + tc) ** (self.order - 1) * np.exp(-2 * np.pi * bw[i] * (t + tc)) * np.cos(2 * np.pi * cf[i] * t)

            # Normalize impulse response
            b /= np.max(np.abs(b))

            # Trim the impulse response to avoid near-zero coefficients
            if self.filterSize is None:
                cofThreshold = 1e-04
                b = np.flipud(b)
                idx = np.where(np.abs(b) > cofThreshold)[0]
                b = np.flipud(b[idx[0]::])

            # Normalize filter
            b /= np.sum(np.square(b))

            # FIXME: the filterbank current do not sum to 0 dB over all the spectrum!
            filters.append(b)

        return filters


def visualizeFilterbankData(data, fs, cf=None):
    fig = plt.figure()

    nbSamples, nbChannels = data.shape
    t = np.arange(nbSamples) / fs

    ticklocs = []
    ax = fig.add_subplot(1, 1, 1)
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.7
    y0 = dmin
    y1 = (nbChannels - 1) * dr + dmax
    ax.set_ylim(y0, y1)

    segs = []
    for i in range(nbChannels):
        segs.append(np.hstack((t[:, np.newaxis], data[:, i, np.newaxis])))
        ticklocs.append(i * dr)

    offsets = np.zeros((nbChannels, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    ax.add_collection(lines)

    ax.set_yticks(ticklocs)
    if cf is not None:
        assert len(cf) == nbChannels
        ax.set_yticklabels(['%d Hz' % (int(f)) for f in cf])
    else:
        ax.set_yticklabels(['CH%d' % (d + 1) for d in range(nbChannels)])

    ax.set_xlabel('Time (s)')
    return fig


class STFT(Layer):

    def __init__(self, fs, frameLength=1024, frameStep=256, fftLength=None, **kargs):
        if fftLength is None:
            fftLength = frameLength

        self.__dict__.update(fs=fs, frameLength=frameLength, frameStep=frameStep, fftLength=fftLength)
        super(STFT, self).__init__(**kargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'fs': self.fs,
                  'frameLength': self.frameLength,
                  'frameStep': self.frameStep,
                  'fftLength': self.fftLength}
        base_config = super(STFT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):

        if K.backend() != 'tensorflow':
            raise Exception('Unsupported backend: %s' % (K.backend()))

        # NOTE: tf.contrib.signal.stft expects time to be the last dimension, so reorder temporarily
        x = tf.transpose(x, perm=[0, 2, 1])
        stfts = tf.contrib.signal.stft(x, frame_length=self.frameLength, frame_step=self.frameStep,
                                       fft_length=self.fftLength, pad_end=True)
        stfts = tf.transpose(stfts, perm=[0, 2, 1, 3])
        spectrograms = tf.abs(stfts)
        return spectrograms

    def compute_output_shape(self, input_shape):
        nbSpectrogramBins = self.fftLength // 2 + 1
        return (input_shape[0], None, nbSpectrogramBins)


class MFCC(Layer):

    def __init__(self, fs, frameLength=1024, frameStep=256, fftLength=None,
                 melSpaceLowFreq=80, melSpaceHighFreq=None, melSpaceNbBins=80,
                 cepstralLiftering=22, nbMfccs=20,
                 deltaMfccs=True, doubleDeltaMfccs=True, **kargs):

        if fftLength is None:
            fftLength = frameLength
        if melSpaceHighFreq is None:
            melSpaceHighFreq = int(fs / 2)

        self.__dict__.update(fs=fs, frameLength=frameLength, frameStep=frameStep, fftLength=fftLength,
                             melSpaceLowFreq=melSpaceLowFreq, melSpaceHighFreq=melSpaceHighFreq, melSpaceNbBins=melSpaceNbBins,
                             cepstralLiftering=cepstralLiftering, nbMfccs=nbMfccs, deltaMfccs=deltaMfccs, doubleDeltaMfccs=doubleDeltaMfccs)
        super(MFCC, self).__init__(**kargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'fs': self.fs,
                  'frameLength': self.frameLength,
                  'frameStep': self.frameStep,
                  'fftLength': self.fftLength,
                  'melSpaceLowFreq': self.melSpaceLowFreq,
                  'melSpaceHighFreq': self.melSpaceHighFreq,
                  'melSpaceNbBins': self.melSpaceNbBins,
                  'cepstralLiftering': self.cepstralLiftering,
                  'nbMfccs': self.nbMfccs,
                  'deltaMfccs': self.deltaMfccs,
                  'doubleDeltaMfccs': self.doubleDeltaMfccs}
        base_config = super(MFCC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):

        if K.backend() != 'tensorflow':
            raise Exception('Unsupported backend: %s' % (K.backend()))

        # NOTE: tf.contrib.signal.stft expects time to be the last dimension, so reorder temporarily
        x = tf.transpose(x, perm=[0, 2, 1])
        stfts = tf.contrib.signal.stft(x, frame_length=self.frameLength, frame_step=self.frameStep,
                                       fft_length=self.fftLength, pad_end=True)
        stfts = tf.transpose(stfts, perm=[0, 2, 1, 3])
        num_spectrogram_bins = stfts.shape[-1].value
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            self.melSpaceNbBins, num_spectrogram_bins, self.fs, self.melSpaceLowFreq, self.melSpaceHighFreq)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms)[..., :self.nbMfccs]
        mfccs = tf.cast(mfccs, tf.float32)

        # TODO: allow an option to compute the log energy of each frame and put the value as the 0-th MFCC coefficient

        # TODO: add support for Cepstral Mean Normalisation (CMS)

        # Sinusoidal cepstral liftering
        if self.cepstralLiftering is not None:
            n = tf.range(self.nbMfccs, dtype=tf.float32)
            L = self.cepstralLiftering
            lift = 1 + (L / 2.0) * tf.sin(np.pi * n / L)
            mfccs = lift * mfccs

        # Compute delta and double-delta cepstral features
        padding = tf.constant([[0, 0, ], [1, 0, ], [0, 0, ], [0, 0, ]])
        deltas = tf.pad(mfccs[:, 1:, :, :] - mfccs[:, :-1, :, :],
                        padding, "CONSTANT")
        deltadeltas = tf.pad(deltas[:, 1:, :, :] - deltas[:, :-1, :, :],
                             padding, "CONSTANT")
        if self.deltaMfccs and self.doubleDeltaMfccs:
            features = tf.concat([mfccs, deltas, deltadeltas], axis=-1)
        elif self.deltaMfccs:
            features = tf.concat([mfccs, deltas], axis=-1)
        elif self.doubleDeltaMfccs:
            features = tf.concat([mfccs, deltadeltas], axis=-1)
        else:
            features = mfccs

        return features

    def compute_output_shape(self, input_shape):
        nbFeatures = self.nbMfccs
        if self.deltaMfccs:
            nbFeatures += self.nbMfccs
        if self.doubleDeltaMfccs:
            nbFeatures += self.nbMfccs
        return (input_shape[0], None, nbFeatures)


class SmoothCepstral(Layer):

    def __init__(self, fs, filterLength=1024,
                 melSpaceLowFreq=80, melSpaceHighFreq=None, melSpaceNbBins=80, nbCepstralFeatures=None, **kargs):

        if melSpaceHighFreq is None:
            melSpaceHighFreq = int(fs / 2)
        if nbCepstralFeatures is None:
            nbCepstralFeatures = melSpaceNbBins

        self.__dict__.update(fs=fs, filterLength=filterLength,
                             melSpaceLowFreq=melSpaceLowFreq, melSpaceHighFreq=melSpaceHighFreq, melSpaceNbBins=melSpaceNbBins,
                             nbCepstralFeatures=nbCepstralFeatures)
        super(SmoothCepstral, self).__init__(**kargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'fs': self.fs,
                  'filterLength': self.filterLength,
                  'melSpaceLowFreq': self.melSpaceLowFreq,
                  'melSpaceHighFreq': self.melSpaceHighFreq,
                  'melSpaceNbBins': self.melSpaceNbBins,
                  'nbCepstralFeatures': self.nbCepstralFeatures}
        base_config = super(SmoothCepstral, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):

        if K.backend() != 'tensorflow':
            raise Exception('Unsupported backend: %s' % (K.backend()))

        # Mel-scale FIR filterbank
        # Replaces the STFT and warping of linear scale spectrograms into the mel-scale.
        mel_spectrograms = FilterBank(nbFilters=self.melSpaceNbBins, filterSize=self.filterLength, fs=self.fs,
                                      cfmin=self.melSpaceLowFreq, cfmax=self.melSpaceHighFreq, warping='melspace')(x)

        # Compute a stabilized log of the the energy of each sub-band, to get log-magnitude
        log_mel_spectrograms = tf.log(tf.abs(mel_spectrograms) + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first ones.
        mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms)[..., :self.nbCepstralFeatures]
        mfccs = tf.cast(mfccs, tf.float32)

        features = mfccs

        return features

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.nbCepstralFeatures)


class InstantaneousAmplitude(Layer):
    """ Instantaneous amplitude of an analytic signal given by the Hilbert transform """

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(InstantaneousAmplitude, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=1)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(InstantaneousAmplitude, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        ndim = len(x.shape)
        if self.axis == -1:
            axis = ndim - 1
        else:
            axis = self.axis

        N = K.shape(x)[axis]

        axes = list(range(ndim))
        axes.remove(axis)
        x = K.permute_dimensions(x, axes + [axis])

        # Remove the mean
        xmean = K.mean(x, axis=-1, keepdims=True)
        x -= xmean

        if K.backend() == 'tensorflow':
            Xf = tf.fft(tf.cast(x, dtype=tf.complex64))

            # Even and odd length signal
            h_even = tf.concat([[1.0], tf.fill([(N // 2) - 1], 2.0), [1.0], tf.fill([(N // 2) - 1], 0.0)], axis=0)
            h_odd = tf.concat([[1.0], tf.fill([((N + 1) // 2) - 1], 2.0), [0.0], tf.fill([(N // 2) - 1], 0.0)], axis=0)
            h = tf.cond(tf.equal(tf.mod(N, 2), 0), lambda: h_even, lambda: h_odd)

            y = tf.ifft(Xf * tf.cast(h, dtype=tf.complex64))
            y = tf.abs(y)
        else:
            raise Exception('Unsupported backend: %s' % (K.backend()))

        # Add back the mean
        y += xmean

        axes = list(range(ndim))
        axes.remove(axis)
        y = K.permute_dimensions(y, axes + [axis])

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class Conv1DTranspose(Conv2DTranspose):

    def __init__(self, filters,
                 kernel_size,
                 stride=1,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(
            filters,
            kernel_size=(kernel_size, 1),
            strides=(stride, 1),
            **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'stride': self.stride}
        base_config = super(Conv1DTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(Conv1DTranspose, self).build((input_shape[0], input_shape[1], 1, input_shape[2]))

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = input_shape[channel_axis]
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})

    def call(self, inputs):
        return K.squeeze(super(Conv1DTranspose, self).call(K.expand_dims(inputs, axis=2)), axis=2)

    def compute_output_shape(self, input_shape):
        output_shape = super(Conv1DTranspose, self).compute_output_shape((input_shape[0], input_shape[1], 1, input_shape[2]))
        return (output_shape[0], output_shape[1], output_shape[3])


class Resample1D(Layer):

    methods = {'linear': ResizeMethod.BILINEAR,
               'nearest': ResizeMethod.NEAREST_NEIGHBOR,
               'cubic': ResizeMethod.BICUBIC}

    def __init__(self, scale, method='linear', **kwargs):
        if method not in Resample1D.methods:
            raise Exception('Unsupported resampling method: %s' % (method))

        self.__dict__.update(scale=scale, method=method)
        super(Resample1D, self).__init__(**kwargs)

    def get_config(self):
        config = {'scale': self.scale,
                  'method': self.method}
        base_config = super(Resample1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x = inputs
        resampledLen = tf.cast(self.scale * tf.cast(tf.shape(x)[1], tf.float32), tf.int32)

        # X must have shape: [batch, height, width, channels]
        x = tf.squeeze(tf.image.resize_images(tf.expand_dims(x, 2), size=tf.stack([resampledLen, 1]), method=Resample1D.methods[self.method], align_corners=True), axis=2)
        return x

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            length = int(self.scale * input_shape[1])
        else:
            length = None
        return (input_shape[0], length, input_shape[2])


class LayerNorm1D(Layer):
    # Adapted from: https://github.com/keras-team/keras/issues/3878
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=(input_shape[2],),
                                     initializer=initializers.Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=(input_shape[2],),
                                    initializer=initializers.Zeros(),
                                    trainable=True,)
        self.built = True

    def call(self, x):
        mean = K.mean(x, axis=[0, 1], keepdims=True)
        std = K.std(x, axis=[0, 1], keepdims=True)
        return self.gamma * (x - mean) / (std + K.epsilon()) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
