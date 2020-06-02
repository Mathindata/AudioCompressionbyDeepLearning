
import os
import logging
import datetime
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN

from codec.utils import setReproducible, configureRootLogger
from codec.asr import EndToEndContinuousASR, Vocabulary
from codec.corpus import TIMIT
from codec.visualization import TensorBoard, draw_multinomial_prior

logger = logging.getLogger(__name__)

CDIR = os.path.dirname(os.path.realpath(__file__))


try:
    TIMIT_DATA_DIR = os.environ["TIMIT_DATA_DIR"]
except KeyError:
    raise Exception("Please set the environment variable TIMIT_DATA_DIR")


def main():

    corpus = TIMIT(dirpath=TIMIT_DATA_DIR)
    vocabulary = Vocabulary(corpus.phoneCodes)
#     asr = EndToEndContinuousASR.model(vocabulary.getMaxVocabularySize(),
#                                       nbChannels=corpus.nbChannels, sampleRate=corpus.fs,
#                                       predictionRate=1000, name='asr')
    asr = EndToEndContinuousASR.modelv2(vocabulary.getMaxVocabularySize(),
                                        nbChannels=corpus.nbChannels, sampleRate=corpus.fs,
                                        predictionRate=1000, name='asr')
#     asr = EndToEndContinuousASR.modelv3(vocabulary.getMaxVocabularySize(),
#                                         nbChannels=corpus.nbChannels, fs=corpus.fs,
#                                         nbFilters=128, filterLength=256, nbCepstralFeatures=None, predictionRate=1000,
#                                         name='asr')

    modelPath = os.path.join(CDIR, 'models')
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    modelFilename = os.path.join(modelPath, 'asr.h5')

    logPath = os.path.join(CDIR, 'logs', 'asr', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(logPath):
        os.makedirs(logPath)

    asr.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print(asr.summary())

    callbacks = []
    checkpointer = ModelCheckpoint(modelFilename,
                                   monitor='val_loss',
                                   save_best_only=True,
                                   save_weights_only=False)
    callbacks.append(checkpointer)

    callbacks.append(ReduceLROnPlateau(monitor='loss',
                                       factor=0.2,
                                       patience=25,
                                       min_delta=1e-5,
                                       min_lr=1e-5))

    callbacks.append(TerminateOnNaN())
    # callbacks.append(EarlyStopping(monitor='loss', min_delta=1e-6, patience=50, mode='auto'))

    # NOTE: memory requirement for processing speech on the raw waveform is too high to have a large batch size
    batchSize = 8
    trainData = corpus.trainData()
    trainGenerator = EndToEndContinuousASR.generator(asr, vocabulary, silenceToken='h#', data=trainData, fs=corpus.fs, batchSize=batchSize)
    logger.info('Number of audio samples in training set: %d' % (len(trainData)))

    batchSize = 1
    testData = corpus.testData()
    testGenerator = EndToEndContinuousASR.generator(asr, vocabulary, silenceToken='h#', data=testData, fs=corpus.fs, batchSize=batchSize)
    logger.info('Number of audio samples in test set: %d' % (len(testData)))

    nbShownClasses = 32
    indices = np.arange(nbShownClasses)
    tf.summary.image('prior/asr', draw_multinomial_prior(asr.output, None, corpus.fs, nbShownClasses, indices))

    batchSize = 8
    tensorboardGenerator = EndToEndContinuousASR.generator(asr, vocabulary, silenceToken='h#', data=testData[:batchSize], fs=corpus.fs, batchSize=batchSize)
    tensorboard = TensorBoard(logPath, tensorboardGenerator, fs=corpus.fs, histogram_freq=1, write_graph=True, batch_size=batchSize, write_audio=False, write_grads=True)
    callbacks.append(tensorboard)

    try:
        asr.fit_generator(trainGenerator,
                          epochs=1000,
                          validation_data=testGenerator,
                          use_multiprocessing=False,
                          shuffle=True,
                          verbose=1,
                          callbacks=callbacks)
    except KeyboardInterrupt:
        logger.info("Training interrupted by the user")

    asr.save(modelFilename)

    logger.info('All done.')


if __name__ == '__main__':
    configureRootLogger(logging.INFO, logFile=os.path.join(CDIR, 'logs', 'asr.log'))
    setReproducible()
    main()
