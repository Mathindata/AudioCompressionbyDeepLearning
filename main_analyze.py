
import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

from codec import getModuleObjects
from codec.utils import setReproducible, saveAudioToFile, configureRootLogger
from codec.corpus import TIMIT
from codec.codec import EndToEndCodec, makeNotTrainable


logger = logging.getLogger(__name__)

CDIR = os.path.dirname(os.path.realpath(__file__))

try:
    TIMIT_DATA_DIR = os.environ["TIMIT_DATA_DIR"]
except KeyError:
    raise Exception("Please set the environment variable TIMIT_DATA_DIR")


def main():

    parser = argparse.ArgumentParser(description='Codec training.')
    parser.add_argument("--input-name", help="Name of the codec", type=str)
    args = parser.parse_args()

    corpus = TIMIT(dirpath=TIMIT_DATA_DIR)

    # Load the trained codec from file
    modelPath = os.path.join(CDIR, 'models')
    modelFilename = os.path.join(modelPath, args.input_name + '.h5')
    if not os.path.exists(modelFilename):
        raise Exception('You must train the codec model first!')
    codec = load_model(modelFilename, custom_objects=getModuleObjects(), compile=False)
    makeNotTrainable(codec)
    EndToEndCodec.setQuantizationActive(codec, False)

    testData = corpus.testData()
    testGenerator = EndToEndCodec.generator(codec, testData, corpus.fs, batchSize=1, shuffle=True)

    outputPath = os.path.join(CDIR, 'outputs', args.input_name)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    nbSamples = 10
    for i in range(nbSamples):
        x, _ = testGenerator[i]
        xr = codec.predict(x)

        outputRefFilename = os.path.join(outputPath, 'output-%d-orig.wav' % (i + 1))
        saveAudioToFile(np.squeeze(x), outputRefFilename, corpus.fs)

        outputReconsFilename = os.path.join(outputPath, 'output-%d-recons.wav' % (i + 1))
        saveAudioToFile(np.squeeze(xr), outputReconsFilename, corpus.fs)

        t = np.arange(x.shape[1]) / corpus.fs
        fig = plt.figure()
        plt.plot(t, np.squeeze(x), label='original')
        plt.plot(t, np.squeeze(xr), label='reconstruction')
        plt.axis('tight')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('Codec reconstruction')
        plt.legend()
        plt.ylim(-2.5, 2.5)
        plt.savefig(os.path.join(outputPath, 'output-%d-waveform.png' % (i + 1)))
        plt.close(fig)

    logger.info('All done.')


if __name__ == '__main__':
    configureRootLogger(logging.INFO, logFile=os.path.join(CDIR, 'logs', 'analyze.log'))
    setReproducible(disableGpuMemPrealloc=True)
    main()
