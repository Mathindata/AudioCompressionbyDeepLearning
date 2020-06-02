
import os
import logging
import datetime
import numpy as np

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN, EarlyStopping, TensorBoard
from keras.metrics import top_k_categorical_accuracy

from codec.utils import setReproducible, configureRootLogger
from codec.corpus import TIMIT
from codec.gm import AutoregressiveGenerativeModel, AutoregressiveGenerativeModelDataGenerator
from codec.quantization import MuLawQuantizer

logger = logging.getLogger(__name__)

CDIR = os.path.dirname(os.path.realpath(__file__))


try:
    TIMIT_DATA_DIR = os.environ["TIMIT_DATA_DIR"]
except KeyError:
    raise Exception("Please set the environment variable TIMIT_DATA_DIR")


def top_10_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(K.reshape(y_true, shape=(-1, K.shape(y_true)[-1])),
                                      K.reshape(y_pred, shape=(-1, K.shape(y_pred)[-1])),
                                      k=10)


def top_5_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(K.reshape(y_true, shape=(-1, K.shape(y_true)[-1])),
                                      K.reshape(y_pred, shape=(-1, K.shape(y_pred)[-1])),
                                      k=5)


def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(K.reshape(y_true, shape=(-1, K.shape(y_true)[-1])),
                                      K.reshape(y_pred, shape=(-1, K.shape(y_pred)[-1])),
                                      k=3)


def main():

    corpus = TIMIT(dirpath=TIMIT_DATA_DIR)
    quantizer = MuLawQuantizer(k=256)
    gm = AutoregressiveGenerativeModel(quantizer, nbChannels=corpus.nbChannels,
                                       nbFilters=64,
                                       name='gm')

    modelPath = os.path.join(CDIR, 'models')
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    modelFilename = os.path.join(modelPath, 'gm.h5')

    logPath = os.path.join(CDIR, 'logs', 'gm', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    gm.compile(optimizer=Adam(1e-2), loss='categorical_crossentropy', metrics=['categorical_accuracy',])
#                                                                                top_3_categorical_accuracy,
#                                                                                top_5_categorical_accuracy,
#                                                                                top_10_categorical_accuracy])
    print(gm.summary())

    # NOTE: memory requirement for processing speech on the raw waveform is too high to have a large batch size
    batchSize = 8
    trainData = corpus.trainData()
    trainGenerator = AutoregressiveGenerativeModelDataGenerator(gm, trainData, corpus.fs, batchSize)
    logger.info('Number of audio samples in training set: %d' % (len(trainData)))

    testData = corpus.testData()
    testGenerator = AutoregressiveGenerativeModelDataGenerator(gm, testData, corpus.fs, batchSize)
    logger.info('Number of audio samples in test set: %d' % (len(testData)))

    callbacks = []

    tensorboard = TensorBoard(logPath, histogram_freq=1, write_graph=False, batch_size=batchSize, write_grads=True)
    tensorboardGenerator = AutoregressiveGenerativeModelDataGenerator(gm, testData[:10], corpus.fs, batchSize)
    x, y = tensorboardGenerator[0]
    tensorboard.validation_data = [x,  # X
                                   y,  # y
                                   np.ones(len(x)),  # sample weights
                                   ]
    callbacks.append(tensorboard)

    checkpointer = ModelCheckpoint(modelFilename,
                                   monitor='val_loss',
                                   save_best_only=True,
                                   save_weights_only=True)
    callbacks.append(checkpointer)

    callbacks.append(ReduceLROnPlateau(monitor='loss',
                                       factor=0.2,
                                       patience=5,
                                       min_lr=1e-5))

    callbacks.append(TerminateOnNaN())
    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=50, mode='auto'))

    try:
        gm.fit_generator(trainGenerator,
                         epochs=200,
                         validation_data=testGenerator,
                         use_multiprocessing=False,
                         shuffle=True,
                         verbose=1,
                         callbacks=callbacks)
    except KeyboardInterrupt:
        logger.info("Training interrupted by the user")
        gm.save_weights(modelFilename)

    gm.save_weights(modelFilename)

    logger.info('All done.')


if __name__ == '__main__':
    configureRootLogger(logging.INFO, logFile=os.path.join(CDIR, 'logs', 'gm.log'))
    setReproducible()
    main()
