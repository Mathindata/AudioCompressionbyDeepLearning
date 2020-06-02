
import os
import sys
import logging
import subprocess
import six
import functools
import numpy as np

from keras import backend as K
from keras.layers import Input, Conv1D
from keras.models import Model
from keras.engine.base_layer import InputSpec, Layer
from keras.initializers import Constant
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Lambda
from keras.layers.merge import Add, Multiply

from .signal import findNextDivisible, Resample1D
from .core import Codec
from .utils import argsToString
from .core import AudioDataGenerator
from functools import partial
from codec.signal import Preemphasis


logger = logging.getLogger(__name__)


def ndArraytoBytes(x):
    # Get byte data in the proper order (little endian)
    return x.newbyteorder('<').tobytes('C')


class FFmpegCodec(Codec):

    supportedByteFormats = {np.dtype(np.float32): 'f32le',  # PCM 32-bit floating-point little-endian
                            np.dtype(np.float64): 'f64le',  # PCM 64-bit floating-point little-endian
                            np.dtype(np.int16): 's16le',    # PCM signed 16-bit little-endian
                            np.dtype(np.int32): 's32le',    # PCM signed 32-bit little-endian
                            np.dtype(np.int8): 's8',        # PCM signed 8-bit
                            np.dtype(np.uint16): 'u16le',   # PCM unsigned 16-bit little-endian
                            np.dtype(np.uint32): 'u32le',   # PCM unsigned 32-bit little-endian
                            np.dtype(np.uint8): 'u8',       # PCM unsigned 8-bit
                            }

    supportedContainers = ['ac3',    # raw AC-3
                           'aiff',   # Audio IFF
                           'alaw',   # PCM A-law
                           'amr',    # 3GPP AMR
                           'ast',    # AST (Audio Stream)
                           'au',     # Sun AU
                           'data',   # raw data
                           'eac3',   # raw E-AC-3
                           'f32le',  # PCM 32-bit floating-point little-endian
                           'f64le',  # PCM 64-bit floating-point little-endian
                           'flac',   # raw FLAC
                           'g722',   # raw G.722
                           'g723_1',  # raw G.723.1
                           'gsm',    # raw GSM
                           'ircam',  # Berkeley/IRCAM/CARL Sound Format
                           'mp3',    # MP3 (MPEG audio layer 3)
                           'mulaw',  # PCM mu-law
                           'ogg',    # Ogg
                           'oma',    # Sony OpenMG audio
                           's16le',  # PCM signed 16-bit little-endian
                           's24le',  # PCM signed 24-bit little-endian
                           's32le',  # PCM signed 32-bit little-endian
                           's8',     # PCM signed 8-bit
                           'tta',    # TTA (True Audio)
                           'u16le',  # PCM unsigned 16-bit little-endian
                           'u24le',  # PCM unsigned 24-bit little-endian
                           'u32le',  # PCM unsigned 32-bit little-endian
                           'u8',     # PCM unsigned 8-bit
                           'w64',    # Sony Wave64
                           'wav',    # WAV / WAVE (Waveform Audio)
                           'wv',     # raw WavPack
                           ]

    supportedCodecs = ['aac',                  # AAC (Advanced Audio Coding) (decoders: aac aac_fixed )
                       'ac3',                  # ATSC A/52A (AC-3) (decoders: ac3 ac3_fixed ) (encoders: ac3 ac3_fixed )
                       'adpcm_adx',            # SEGA CRI ADX ADPCM
                       'adpcm_g722',           # G.722 ADPCM (decoders: g722 ) (encoders: g722 )
                       'adpcm_g726',           # G.726 ADPCM (decoders: g726 ) (encoders: g726 )
                       'adpcm_ima_qt',         # ADPCM IMA QuickTime
                       'adpcm_ima_wav',        # ADPCM IMA WAV
                       'adpcm_ms',             # ADPCM Microsoft
                       'adpcm_swf',            # ADPCM Shockwave Flash
                       'adpcm_yamaha',         # ADPCM Yamaha
                       'alac',                 # ALAC (Apple Lossless Audio Codec)
                       'amr_nb',               # AMR-NB (Adaptive Multi-Rate NarrowBand) (decoders: amrnb libopencore_amrnb ) (encoders: libopencore_amrnb )
                       'amr_wb',               # AMR-WB (Adaptive Multi-Rate WideBand) (decoders: amrwb libopencore_amrwb ) (encoders: libvo_amrwbenc )
                       'comfortnoise',         # RFC 3389 Comfort Noise
                       'dts',                  # DCA (DTS Coherent Acoustics) (decoders: dca ) (encoders: dca )
                       'eac3',                 # ATSC A/52B (AC-3, E-AC-3)
                       'flac',                 # FLAC (Free Lossless Audio Codec)
                       'g723_1',               # G.723.1
                       'gsm',                  # GSM (decoders: gsm libgsm ) (encoders: libgsm )
                       'gsm_ms',               # GSM Microsoft variant (decoders: gsm_ms libgsm_ms ) (encoders: libgsm_ms )
                       'mlp',                  # MLP (Meridian Lossless Packing)
                       'mp2',                  # MP2 (MPEG audio layer 2) (decoders: mp2 mp2float ) (encoders: mp2 mp2fixed libtwolame )
                       'libmp3lame',           # MP3 (MPEG audio layer 3) (decoders: mp3 mp3float ) (encoders: libmp3lame libshine )
                       'nellymoser',           # Nellymoser Asao
                       'libopus',              # Opus (Opus Interactive Audio Codec) (decoders: opus libopus ) (encoders: opus libopus )
                       'pcm_alaw',             # PCM A-law / G.711 A-law
                       'pcm_f32be',            # PCM 32-bit floating point big-endian
                       'pcm_f32le',            # PCM 32-bit floating point little-endian
                       'pcm_f64be',            # PCM 64-bit floating point big-endian
                       'pcm_f64le',            # PCM 64-bit floating point little-endian
                       'pcm_mulaw',            # PCM mu-law / G.711 mu-law
                       'pcm_s16be',            # PCM signed 16-bit big-endian
                       'pcm_s16be_planar',     # PCM signed 16-bit big-endian planar
                       'pcm_s16le',            # PCM signed 16-bit little-endian
                       'pcm_s16le_planar',     # PCM signed 16-bit little-endian planar
                       'pcm_s24be',            # PCM signed 24-bit big-endian
                       'pcm_s24daud',          # PCM D-Cinema audio signed 24-bit
                       'pcm_s24le',            # PCM signed 24-bit little-endian
                       'pcm_s24le_planar',     # PCM signed 24-bit little-endian planar
                       'pcm_s32be',            # PCM signed 32-bit big-endian
                       'pcm_s32le',            # PCM signed 32-bit little-endian
                       'pcm_s32le_planar',     # PCM signed 32-bit little-endian planar
                       'pcm_s64be',            # PCM signed 64-bit big-endian
                       'pcm_s64le',            # PCM signed 64-bit little-endian
                       'pcm_s8',               # PCM signed 8-bit
                       'pcm_s8_planar',        # PCM signed 8-bit planar
                       'pcm_u16be',            # PCM unsigned 16-bit big-endian
                       'pcm_u16le',            # PCM unsigned 16-bit little-endian
                       'pcm_u24be',            # PCM unsigned 24-bit big-endian
                       'pcm_u24le',            # PCM unsigned 24-bit little-endian
                       'pcm_u32be',            # PCM unsigned 32-bit big-endian
                       'pcm_u32le',            # PCM unsigned 32-bit little-endian
                       'pcm_u8',               # PCM unsigned 8-bit
                       'ra_144',               # RealAudio 1.0 (14.4K) (decoders: real_144 ) (encoders: real_144 )
                       'roq_dpcm',             # DPCM id RoQ
                       'sonic',                # Sonic
                       'speex',                # Speex (decoders: libspeex ) (encoders: libspeex )
                       'truehd',               # TrueHD
                       'tta',                  # TTA (True Audio)
                       'libvorbis',            # Vorbis (decoders: vorbis libvorbis ) (encoders: vorbis libvorbis )
                       'wavpack',              # WavPack (encoders: wavpack libwavpack )
                       'wmav1',                # Windows Media Audio 1
                       'wmav2',                # Windows Media Audio 2
                       ]

    supportedExperimentalCodecs = ['dts',                  # DCA (DTS Coherent Acoustics) (decoders: dca ) (encoders: dca )
                                   'mlp',                  # MLP (Meridian Lossless Packing)
                                   's302m',                # SMPTE 302M
                                   'sonic',                # Sonic
                                   'truehd',               # TrueHD
                                   ]

    supportedSampleRates = {'adpcm_g726': [8000],
                            'adpcm_swf': [11025, 22050, 44100],
                            'amr_nb': [8000],
                            'eac3': [44100, 48000],
                            'g723_1': [8000],
                            'gsm': [8000],
                            'gsm_ms': [8000],
                            'roq_dpcm': [22050],
                            }

    supportedBitrates = {'g723_1': [6300],
                         'gsm': [13000],
                         'gsm_ms': [13000]}

    encoders = {'ac3': 'ac3',
                'adpcm_g722': 'g722',
                'adpcm_g726': 'g726',
                'amr_nb': 'libopencore_amrnb',
                'amr_wb': 'libvo_amrwbenc',
                'dts': 'dca',
                'gsm': 'libgsm',
                'gsm_ms': 'libgsm_ms',
                'mp2': 'mp2',
                'mp3': 'libmp3lame',
                'libopus': 'libopus',
                'ra_144': 'real_144',
                'speex': 'libspeex',
                'libvorbis': 'libvorbis',
                'wavpack': 'libwavpack',
                }

    decoders = {'aac': 'aac',
                'ac3': 'ac3',
                'adpcm_g722': 'g722',
                'adpcm_g726': 'g726',
                'amr_nb': 'libopencore_amrnb',
                'amr_wb': 'libopencore_amrwb',
                'dts': 'dca',
                'gsm': 'libgsm',
                'gsm_ms': 'libgsm_ms',
                'mp2': 'mp2',
                'mp3': 'mp3',
                'libopus': 'libopus',
                'ra_144': 'real_144',
                'speex': 'libspeex',
                'libvorbis': 'libvorbis',
                }

    def __init__(self, codecFormat, container='data', bitrate=None):
        self.__dict__.update(codecFormat=codecFormat, container=container, bitrate=bitrate)
        Codec.__init__(self, name=codecFormat + '-ffmpeg')

    def encode(self, data, fs, codecParams=None, extraArgs=None):

        defaultCodecParams = {'format': self.codecFormat,
                              'container': self.container,
                              'bitrate': self.bitrate}
        if codecParams is None:
            codecParams = defaultCodecParams
        else:
            newCodecParams = dict(defaultCodecParams)
            for key, value in six.iteritems(codecParams):
                newCodecParams[key] = value
            codecParams = newCodecParams

        if data.dtype not in FFmpegCodec.supportedByteFormats:
            raise Exception('Unsupported byte format: %s' % (str(data.dtype)))
        inputFormatPCM = FFmpegCodec.supportedByteFormats[data.dtype]

        if data.ndim == 1:
            nbChannels = 1
        elif data.ndim == 2:
            nbChannels = data.shape[1]
        else:
            raise Exception('Unsupported data shape: %s' % (str(data.shape)))

        # Open a pipe to ffmpeg for reading and writing
        args = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']

        args += ['-ar', str(int(fs)), '-ac', str(int(nbChannels)), '-f', inputFormatPCM, '-i', 'pipe:0']

        codecFormat = codecParams['format']
        container = codecParams['container']
        if codecFormat is not None:
            if codecFormat in FFmpegCodec.encoders:
                codec = FFmpegCodec.encoders[codecFormat]
            else:
                codec = codecFormat
            args += ['-acodec', codec]
        if codecFormat in FFmpegCodec.supportedExperimentalCodecs:
            logger.warning('Using experimental codec \'%s\'' % (codecFormat))
            args += ['-strict', '-2']
        args += ['-f', container]

        bitrate = codecParams['bitrate']
        if bitrate is not None:
            if codecFormat in FFmpegCodec.supportedBitrates and int(bitrate) not in FFmpegCodec.supportedBitrates[codecFormat]:
                bestBitrate = min(FFmpegCodec.supportedBitrates[codecFormat], key=lambda x: abs(x - bitrate))
                logger.warning('Need to set bitrate from %d b/s to %d b/s for codec \'%s\'' % (bitrate, bestBitrate, codecFormat))
                bitrate = bestBitrate
            args += ['-b:a', str(bitrate)]

        if codecFormat in FFmpegCodec.supportedSampleRates and int(fs) not in FFmpegCodec.supportedSampleRates[codecFormat]:
            # Find the nearest sample rate
            bestFs = min(FFmpegCodec.supportedSampleRates[codecFormat], key=lambda x: abs(x - fs))
            logger.warning('Need to resample audio data from %d Hz to %d Hz for codec \'%s\'' % (fs, bestFs, self.codec))
            fs = bestFs
        args += ['-ar', str(int(fs))]

        if extraArgs is not None:
            args += extraArgs

        args += ['-map', '0', 'pipe:1']

        logger.debug('Invoking FFmpeg for encoding: ' + argsToString(args))
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())

        stdoutBytes, stderrBytes = p.communicate(input=ndArraytoBytes(data))

        encoding = sys.stdout.encoding
        if encoding is None:
            encoding = 'UTF-8'

        if stderrBytes is not None and len(stderrBytes) > 0:
            msg = stderrBytes.decode(encoding)
            raise Exception('Error returned from FFmpeg during encoding: ' + msg)
        if stdoutBytes is None or len(stdoutBytes) == 0:
            raise Exception('No bytes received from FFmpeg during encoding')
        encodedBytes = np.frombuffer(stdoutBytes, dtype=np.uint8)

        logger.debug('Original size: %d bytes, encoded size: %d bytes' % (data.nbytes, encodedBytes.nbytes))

        return encodedBytes, codecParams

    def decode(self, data, fs, nbChannels, dtype, codecParams):
        assert data.dtype == np.uint8
        assert data.ndim == 1

        dtype = np.dtype(dtype)
        if dtype not in FFmpegCodec.supportedByteFormats:
            raise Exception('Unsupported byte format: %s' % (str(dtype)))
        outputFormatPCM = FFmpegCodec.supportedByteFormats[dtype]

        # Open a pipe to ffmpeg for reading and writing
        args = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
        args += ['-acodec', str(codecParams['format']), '-ac', str(int(nbChannels)), '-f', codecParams['container'], '-i', 'pipe:0']
        args += ['-map', '0', '-f', outputFormatPCM, 'pipe:1']
        logger.debug('Invoking FFmpeg for decoding: ' + argsToString(args))
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())

        stdoutBytes, stderrBytes = p.communicate(input=ndArraytoBytes(data))

        encoding = sys.stdout.encoding
        if encoding is None:
            encoding = 'UTF-8'

        if stderrBytes is not None and len(stderrBytes) > 0:
            msg = stderrBytes.decode(encoding)
            raise Exception('Error returned from FFmpeg during decoding: ' + msg)
        if stdoutBytes is None or len(stdoutBytes) == 0:
            raise Exception('No bytes received from FFmpeg during decoding')
        decodedBytes = np.frombuffer(stdoutBytes, dtype=dtype)

        if nbChannels > 1:
            decodedBytes = np.reshape(decodedBytes, (-1, nbChannels), order='C')

        if str(codecParams['format']) == 'libopus':
            # NOTE: FFMpeg will decode Opus always at 48 kHz
            import resampy
            decodedBytes = resampy.resample(decodedBytes, 48000, fs)

        logger.debug('Encoded size: %d bytes, decoded size: %d bytes' % (data.nbytes, decodedBytes.nbytes))

        return decodedBytes


class SoftmaxQuantization(Layer):

    def __init__(self, nbBins, beta=1.0, **kwargs):
        self.__dict__.update(nbBins=nbBins, beta=beta)
        super(SoftmaxQuantization, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=1)
        self.active = True
        self.activeSwitch = None

    def get_config(self):
        config = {'nbBins': self.nbBins,
                  'beta': self.beta}
        base_config = super(SoftmaxQuantization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def setActive(self, active):
        if self.activeSwitch is not None:
            K.set_value(self.activeSwitch, active)
        self.active = active

    def build(self, _):

        # Flag to activate or not quantization
        self.activeSwitch = K.variable(bool(self.active), name='quantization-active')

        # Stiffness control parameter of the softmax distribution
        # TODO: is regularization actually needed here?
        self.log_beta = self.add_weight(shape=(1,),
                                        initializer=Constant(value=np.log(self.beta)),
                                        regularizer=None,  # regularizers.l1(1e-3)
                                        trainable=True,
                                        name='log-beta')

        # Quantization bins uniformly distributed on the [-1, 1] interval
        self.bins = K.constant(np.linspace(-1.0, 1.0, self.nbBins).astype(np.float32), name='quantization-bins')

        self.built = True

    def call(self, x):

        # Force the input in the interval [-1.0, 1.0] of the quantization bins
        x = K.tanh(x)

        # Soft assignments from L1 distance to bins
        distances = K.abs(K.expand_dims(x, axis=-1) - K.expand_dims(self.bins, axis=0))
        qs = K.softmax(distances / -K.exp(self.log_beta), axis=-1)

        # Hard assignments using masking
        indices = K.argmax(qs, axis=-1)
        mask = K.one_hot(indices, self.nbBins)
        qh = qs * K.stop_gradient(mask)

        # Select between hard and soft assignments depending if quantization is active
        q = K.switch(self.activeSwitch, qh, qs)

        return q

    def compute_output_shape(self, input_shape):
        return input_shape + (self.nbBins,)


class SoftmaxDequantization(Layer):

    def __init__(self, nbBins, **kwargs):
        self.__dict__.update(nbBins=nbBins)
        super(SoftmaxDequantization, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=1)

    def get_config(self):
        config = {'nbBins': self.nbBins}
        base_config = super(SoftmaxDequantization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, _):
        # Quantization bins uniformly distributed on the [-1, 1] interval
        self.bins = K.constant(np.linspace(-1.0, 1.0, self.nbBins).astype(np.float32), name='quantization-bins')

        self.built = True

    def call(self, x):
        # Dequantization from assignments
        x = K.sum(x * self.bins, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


def snr(y_true, y_pred):
    energySignal = K.sum(K.square(y_true))
    energyNoise = K.sum(K.square(y_true - y_pred))

    snr = K.switch(K.greater(energyNoise, 0.0),
                   10.0 * K.log(energySignal / energyNoise) / np.log(10.0),
                   np.inf * K.ones_like(energySignal))
    return snr


def crossentropy(p, q):
    return -K.mean((p + K.epsilon()) * K.log(q + K.epsilon()) / np.log(2.0))


def kl_divergence(p, q):
    return K.mean((p + K.epsilon()) * K.log((p + K.epsilon()) / (q + K.epsilon())) / np.log(2.0))


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def js_distance(p, q):
    return K.sqrt(js_divergence(p, q))


class EndToEndCodec(object):

    @staticmethod
    def model(nbChannels, fs, stride=2, embeddingRate=30.0, nbEmbeddingDim=32, nbQuantizationBins=16, nbResidualLayers=2, **kwargs):

        inputs = Input(shape=(None, nbChannels))
        x = inputs

        nbFilters = 64
        kernelSize = 3
        with K.name_scope('encoder'):
            i = 0
            fs = fs
            while fs > embeddingRate:
                fs = int(np.ceil(fs / stride))

                # Projection shortcut using resampling
                xs = Resample1D(scale=1.0 / stride)(x)

                # Residual blocks
                for _ in range(nbResidualLayers):
                    fx = Conv1D(nbFilters, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(x)
                    fx = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(fx)
                    fx = Conv1D(nbFilters, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(fx)
                    x = Add()([fx, x])
                    x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

                # Non-linear convolution followed by linear downsampling
                x = Conv1D(nbFilters, kernel_size=kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-ds-%d' % (i + 1))(x)
                x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)
                x = Resample1D(scale=1.0 / stride, method='linear')(x)

                x = Add()([x, xs])

                i += 1
            nbDownsamplingLayers = i

            x = Conv1D(nbEmbeddingDim, kernel_size=3, strides=1, activation='linear', padding='same', name='codec-conv-embedding')(x)
            x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

        # Quantization
        if nbQuantizationBins is not None:
            quantization = SoftmaxQuantization(nbQuantizationBins, name='quantization')
            x = quantization(x)

        # Dequantization
        if nbQuantizationBins is not None:
            x = SoftmaxDequantization(nbQuantizationBins)(x)

        with K.name_scope('decoder'):

            x = Conv1D(nbFilters, kernel_size=1, strides=1, activation='linear', padding='same')(x)
            x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

            # Upsampling with resize convolutions
            for i in range(nbDownsamplingLayers):

                # Projection shortcut
                xs = Resample1D(stride, method='linear')(x)

                # Linear upsampling followed by non-linear convolution
                x = Resample1D(stride, method='linear')(x)
                x = Conv1D(nbFilters, kernel_size=kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-us-%d' % (i + 1))(x)
                x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

                # Residual blocks
                for _ in range(nbResidualLayers):
                    fx = Conv1D(nbFilters, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(x)
                    fx = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(fx)
                    fx = Conv1D(nbFilters, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(fx)
                    x = Add()([fx, x])
                    x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

                x = Add()([x, xs])

            xr = Conv1D(nbChannels, kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-out')(x)
            outputs = xr

        return Model(inputs, outputs, **kwargs)

    @staticmethod
    def modelv2(nbChannels, stride=2, nbFilters=[64, 64, 128, 128, 256], nbEmbeddingDim=32, nbQuantizationBins=16, nbResidualLayers=1, **kwargs):

        inputs = Input(shape=(None, nbChannels))
        x = inputs

        kernelSize = 5
        with K.name_scope('encoder'):
            for i, n in enumerate(nbFilters):
                # Non-linear convolution followed by linear downsampling
                x = Conv1D(n, kernel_size=kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-ds-%d' % (i + 1))(x)
                x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

                # Residual blocks
                for _ in range(nbResidualLayers):
                    fx = Conv1D(n, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(x)
                    fx = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(fx)
                    fx = Conv1D(n, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(fx)
                    x = Add()([fx, x])
                    x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

                x = Resample1D(scale=1.0 / stride, method='linear')(x)

            x = Conv1D(nbEmbeddingDim, kernel_size=kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-embedding')(x)
            x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

        # Quantization
        if nbQuantizationBins is not None:
            quantization = SoftmaxQuantization(nbQuantizationBins, name='quantization')
            x = quantization(x)

        # Dequantization
        if nbQuantizationBins is not None:
            x = SoftmaxDequantization(nbQuantizationBins)(x)

        kernelSize = 5
        with K.name_scope('decoder'):
            # Upsampling with resize convolutions
            for i, n in enumerate(reversed(nbFilters)):
                # Linear upsampling followed by non-linear convolution
                x = Resample1D(stride, method='linear')(x)

                x = Conv1D(n, kernel_size=kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-us-%d' % (i + 1))(x)
                x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

                # Residual blocks
                for _ in range(nbResidualLayers):
                    fx = Conv1D(n, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(x)
                    fx = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(fx)
                    fx = Conv1D(n, kernel_size=kernelSize, strides=1, activation='linear', padding='same')(fx)
                    x = Add()([fx, x])
                    x = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(x)

            xr = Conv1D(nbChannels, kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-out')(x)
            outputs = xr

        return Model(inputs, outputs, **kwargs)

    @staticmethod
    def modelv3(nbChannels, stride=2, nbFilters=[64, 64, 128, 128, 256], nbEmbeddingDim=32, nbQuantizationBins=16, nbResidualLayers=1, **kwargs):

        inputs = Input(shape=(None, nbChannels))
        x = inputs

        def _gated_block(x, kernelSize, nbFilters, noisy=False, name=None):

            # Define the gated activation unit for the input
            f = Conv1D(nbFilters, kernelSize, padding='same', activation='linear', name=name)(x)    # filter
            f = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(f)

            g = Conv1D(nbFilters, kernelSize, padding='same', activation='sigmoid')(x)              # gate
            zi = Multiply()([f, g])

            if noisy:
                # Define the gated activation unit for the noise
                n = Lambda(lambda x: K.random_normal(shape=K.shape(x), mean=0.0, stddev=1.0))(x)
                f = Conv1D(nbFilters, kernelSize, padding='same', activation='linear')(n)     # filter
                f = PReLU(alpha_initializer=Constant(0.25), shared_axes=[1])(f)
                g = Conv1D(nbFilters, kernelSize, padding='same', activation='sigmoid')(x)  # gate
                zn = Multiply()([f, g])

                # Add signal and noise
                y = Add()([zi, zn])
            else:
                y = zi

            # Define the parametrized skip connection
            skip = Conv1D(nbFilters, 1, padding='same')(y)

            return y, skip

        kernelSize = 5
        with K.name_scope('encoder'):
            for i, n in enumerate(nbFilters):
                # Non-linear convolution followed by linear downsampling
                skips = []
                for _ in range(nbResidualLayers):
                    x, skip = _gated_block(x, kernelSize, n, noisy=False, name='codec-conv-ds-%d' % (i + 1))
                    skips.append(skip)

                if len(skips) > 1:
                    x = Add()(skips)
                else:
                    x = skips[0]

                x = Resample1D(scale=1.0 / stride, method='linear')(x)

            x = Conv1D(nbEmbeddingDim, kernel_size=kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-embedding')(x)

        # Quantization
        if nbQuantizationBins is not None:
            quantization = SoftmaxQuantization(nbQuantizationBins, name='quantization')
            x = quantization(x)

        # Dequantization
        if nbQuantizationBins is not None:
            x = SoftmaxDequantization(nbQuantizationBins)(x)

        kernelSize = 5
        with K.name_scope('decoder'):
            # Upsampling with resize convolutions
            for i, n in enumerate(reversed(nbFilters)):
                # Linear upsampling followed by non-linear convolution
                x = Resample1D(stride, method='linear')(x)

                skips = []
                for _ in range(nbResidualLayers):
                    x, skip = _gated_block(x, kernelSize, n, noisy=False, name='codec-conv-us-%d' % (i + 1))
                    skips.append(skip)

                if len(skips) > 1:
                    x = Add()(skips)
                else:
                    x = skips[0]

            xr = Conv1D(nbChannels, kernelSize, strides=1, activation='linear', padding='same', name='codec-conv-out')(x)
            outputs = xr

        return Model(inputs, outputs, **kwargs)

    class DataGenerator(AudioDataGenerator):

        def __init__(self, model, *args, **kwargs):
            self.__dict__.update(model=model)
            AudioDataGenerator.__init__(self, *args, **kwargs)

        def _maximum_length(self, data):
            maximumLength = max([len(audioData.data) for audioData in data])
            stride = 2  # self.model.get_layer('codec-conv-ds-1').strides[0]
            nbDownsamplingLayers = len([layer for layer in self.model.layers if 'codec-conv-ds-' in layer.name])
            return findNextDivisible(maximumLength, stride ** nbDownsamplingLayers)

        def _data_generation_outputs(self, data):
            return self._data_generation_inputs(data)

    @staticmethod
    def generator(*args, **kwargs):
        return EndToEndCodec.DataGenerator(*args, **kwargs)

    @staticmethod
    def setQuantizationActive(model, active):
        try:
            quantization = model.get_layer('quantization')
            quantization.setActive(active)
        except ValueError:
            pass


def makeNotTrainable(model):
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False


def makeTrainable(model):
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True


def binary_crossentropy_logits(x, y):
    max_val = K.clip(-x, 0.0, np.inf)
    loss = x - x * y + max_val + K.log(K.exp(-max_val) + K.exp(-x - max_val))
    return K.mean(loss)


class PerceptualTrainingHelper(object):

    def __init__(self, codec):
        if not isinstance(codec, Model):
            raise Exception('Unsupported codec instance: %s' % (str(codec.__class__.__name__)))
        self.__dict__.update(codec=codec)

        self.loss = K.constant(0.0)
        self.priors = dict()
        self.constraints = dict()
        self.adversaries = dict()

        self._override_functions()

    def addPrior(self, model, name=None, weight=1.0, distribution='multinomial', metric='kl_divergence', unreferenced=False):
        if name is None:
            name = model.name
        if name in self.priors:
            raise Exception('Prior model already exists: %s' % (name))

        if unreferenced:
            # Apply the model on the reconstructed audio only
            makeNotTrainable(model)
            likelihood = model(self.codec.output)

            # Calculate the negative log-likelihood under this model
            nll = -K.mean(K.log(likelihood))
            loss = weight * nll
            self.loss += loss

        else:
            # Apply the model on the original, and reconstructed audio
            makeNotTrainable(model)
            c = model(self.codec.input)
            cr = model(self.codec.output)

            # Calculate the mean KL divergence and add as loss function to model
            if distribution == 'multinomial':
                if metric == 'kl_divergence':
                    loss = kl_divergence(c, cr)
                elif metric == 'reverse_kl_divergence':
                    loss = kl_divergence(cr, c)
                elif metric == 'crossentropy':
                    loss = crossentropy(c, cr)
                elif metric == 'js_distance':
                    loss = js_distance(c, cr)
                elif metric == 'mse' or metric == 'mean_squared_error':
                    loss = K.mean(K.square(c - cr))
                else:
                    raise Exception('Unsupported metric for the multinomial distribution: %s' % (metric))
            else:
                raise Exception('Unsupported distribution: %s' % (distribution))
            loss *= weight
            self.loss += loss

        self.priors[name] = [model, loss]

    def addConstraint(self, model, name=None, weight=1.0, unreferenced=False):
        if name is None:
            name = model.name
        if name in self.constraints:
            raise Exception('Constraint model already exists: %s' % (name))

        if unreferenced:
            makeNotTrainable(model)
            cr = model(self.codec.output)
            loss = weight * K.mean(cr)
            self.loss += loss
        else:
            # Apply the model on the original, and reconstructed audio
            makeNotTrainable(model)
            c = model(self.codec.input)
            cr = model(self.codec.output)

            # Calculate the mean squared error and add as loss function to model
            if isinstance(c, (list, tuple)):
                loss = 0.0
                for i in range(len(c)):
                    mse = K.mean(K.square(c[i] - cr[i]))
                    loss += weight * mse
            else:
                mse = K.mean(K.square(c - cr))
                loss = weight * mse
            self.loss += loss

        self.constraints[name] = [model, loss]

    def addAdversary(self, model, optimizer, name=None, mode='non-saturating', weight=1.0):
        if name is None:
            name = model.name
        if name in self.adversaries:
            raise Exception('Adversary model already exists: %s' % (name))

        # TODO: would it be possible to do simultaneous updates and avoid redundant computation?

        if mode == 'relativistic-average':
            # Loss function to improve the adversarial model (discriminator)
            # NOTE: use the unweight loss as the output of the wrapped model to take into account
            #       trainable weights in the original model.
            makeNotTrainable(self.codec)
            makeTrainable(model)
            D_xr = model(self.codec.input)
            D_xf = model(self.codec.output)
            bce = Lambda(lambda x: 0.5 * binary_crossentropy_logits(x[0] - K.mean(x[1]), 1.0) + 0.5 * binary_crossentropy_logits(x[1] - K.mean(x[0]), 0.0))
            D_bce = bce([D_xr, D_xf])
            loss_d = D_bce
            adversary_model = Model(self.codec.input, D_bce)
            adversary_model.compile(loss=lambda *_: loss_d,
                                    optimizer=optimizer)
            assert len(adversary_model.trainable_weights) > 0

            # Loss function to improve the codec model (generator)
            makeTrainable(self.codec)
            makeNotTrainable(model)
            D_xr = model(self.codec.input)
            D_xf = model(self.codec.output)
            D_bce = 0.5 * binary_crossentropy_logits(D_xr - K.mean(D_xf), 1.0) + 0.5 * binary_crossentropy_logits(D_xf - K.mean(D_xr), 0.0)
            G_bce = 0.5 * binary_crossentropy_logits(D_xr - K.mean(D_xf), 0.0) + 0.5 * binary_crossentropy_logits(D_xf - K.mean(D_xr), 1.0)
            loss_g = G_bce
            self.loss += weight * loss_g
            assert len(self.codec.trainable_weights) > 0

        elif mode == 'non-saturating':
            # Loss function to improve the adversarial model (discriminator)
            # NOTE: use the unweight loss as the output of the wrapped model to take into account
            #       trainable weights in the original model.
            makeNotTrainable(self.codec)
            makeTrainable(model)
            D_xr = model(self.codec.input)
            D_xf = model(self.codec.output)
            bce = Lambda(lambda x: binary_crossentropy_logits(x[0], 1.0) + binary_crossentropy_logits(x[1], 0.0))
            D_bce = bce([D_xr, D_xf])
            loss_d = D_bce
            adversary_model = Model(self.codec.input, D_bce)
            adversary_model.compile(loss=lambda *_: loss_d,
                                    optimizer=optimizer)
            assert len(adversary_model.trainable_weights) > 0

            # Loss function to improve the codec model (generator)
            makeTrainable(self.codec)
            makeNotTrainable(model)
            D_xf = model(self.codec.output)
            G_bce = binary_crossentropy_logits(D_xf, 1.0)
            loss_g = G_bce
            self.loss += weight * loss_g
            assert len(self.codec.trainable_weights) > 0
        else:
            raise Exception('Unsupported mode: %s' % (mode))

        self.adversaries[name] = [adversary_model, (loss_g, loss_d)]

    def getPriorloss(self, name):
        _, loss = self.priors[name]
        return loss

    def getConstraintloss(self, name):
        _, loss = self.constraints[name]
        return loss

    def getAdversarialLoss(self, name):
        _, loss = self.adversaries[name]
        return loss

    def getMetricFunctions(self):
        metrics = []
        for lossName in six.iterkeys(self.priors):
            def loss(y_true, y_pred, ln):
                return self.priors[ln][1]
            func = functools.partial(loss, ln=lossName)
            func.__name__ = lossName
            metrics.append(func)

        for lossName in six.iterkeys(self.constraints):
            def loss(y_true, y_pred, ln):
                return self.constraints[ln][1]
            func = functools.partial(loss, ln=lossName)
            func.__name__ = lossName
            metrics.append(func)

        for lossName in six.iterkeys(self.adversaries):
            # Generator loss
            def loss(y_true, y_pred, ln):
                return self.adversaries[ln][1][0]
            func = functools.partial(loss, ln=lossName)
            func.__name__ = lossName + '_g'
            metrics.append(func)

            # Discriminator loss
            def loss(y_true, y_pred, ln):
                return self.adversaries[ln][1][1]
            func = functools.partial(loss, ln=lossName)
            func.__name__ = lossName + '_d'
            metrics.append(func)

        return ['mse', snr] + metrics

    def _override_functions(self):
        adversaries = self.adversaries

        def _make_train_function(self):
            if not hasattr(self, 'train_function'):
                raise RuntimeError('You must compile your model before using it.')
            self._check_trainable_weights_consistency()
            if self.train_function is None:

                with K.name_scope('training'):
                    # Gets loss and metrics for the codec. Updates weights at each call.
                    with K.name_scope(self.optimizer.__class__.__name__):
                        training_updates = self.optimizer.get_updates(
                            params=self._collected_trainable_weights,
                            loss=self.total_loss)
                    updates = (self.updates +
                               training_updates +
                               self.metrics_updates)

                    inputs = (self._feed_inputs +
                              self._feed_targets +
                              self._feed_sample_weights)
                    if self._uses_dynamic_learning_phase():
                        inputs += [K.learning_phase()]

                    base_train_function = K.function(
                        inputs,
                        [self.total_loss] + self.metrics_tensors,
                        updates=updates,
                        name='train_function_codec',
                        **self._function_kwargs)

                    # Gets losses for the adversary models. Updates weights at each call.
                    adversaries_train_functions = []
                    for name, [model, _] in six.iteritems(adversaries):
                        with K.name_scope(model.optimizer.__class__.__name__):
                            training_updates = model.optimizer.get_updates(
                                params=model._collected_trainable_weights,
                                loss=model.total_loss)
                        updates = (model.updates +
                                   training_updates +
                                   model.metrics_updates)

                        inputs = (model._feed_inputs +
                                  model._feed_targets +
                                  model._feed_sample_weights)
                        if model._uses_dynamic_learning_phase():
                            inputs += [K.learning_phase()]

                        adversary_train_function = K.function(
                            inputs,
                            [model.total_loss] + model.metrics_tensors,
                            updates=updates,
                            name='train_function_adversary_' + name,
                            **model._function_kwargs)

                        adversaries_train_functions.append(adversary_train_function)

                    # Alternating optimization of adversary models, then the codec
                    def alternating_train_function(_inputs):
                        for func in adversaries_train_functions:
                            func(_inputs)
                        return base_train_function(_inputs)

                    self.train_function = alternating_train_function

        self.codec._make_train_function = partial(_make_train_function, self.codec)

    def getLossFunction(self):
        # NOTE: use a dummy loss to be able to show the metrics
        def func(*_):
            return self.loss
        return func
