# The main pupose of this repository is to serve as a code sample. Codes here are a few samples of the main project I did for my PhD at UdeS. For patent reasons, the original repository is private yet.

# End-to-end speech compression with perceptual feedback from an automatic speech recognizer

## Dependencies

Main requirements:
- Python 2.7+ or 3.4+
- [Keras 2.2.0+](https://keras.io/)
- [Tensorflow 1.9.0+](https://www.tensorflow.org/)
- [FFmpeg 3.3.4+](https://www.ffmpeg.org/)

To install dependencies:
```
sudo pip install --upgrade pip numpy scipy matplotlib keras tensorflow tensorboard six joblib nose coverage librosa
```

To setup an environment with Anaconda:
```
conda create -n speech-codec pip python=3.6
source activate speech-codec
conda install -c anaconda numpy scipy matplotlib scikit-learn keras tensorflow-gpu tensorboard six joblib nose coverage cython
conda install -c conda-forge librosa
pip install tfmpl
```

### PESQ

To install the reference PESQ implementation:
```
mkdir -p $HOME/build
cd $HOME/build
git clone https://github.com/dennisguse/ITU-T_pesq.git
cd ITU-T_pesq
make
sudo cp bin/itu-t-pesq2005 /usr/bin
```

### Datasets


```
export TI46_DATA_DIR=/scratch/Datasets/TI46
export RAVDESS_DATA_DIR=/scratch/Datasets/RAVDESS
export TIMIT_DATA_DIR=/scratch/Datasets/TIMIT
export NOISEX92_DATA_DIR=/scratch/Datasets/NOISEX-92
```

## Installing the library

Download the source code from the git repository:
```
mkdir -p $HOME/work
cd $HOME/work
git clone https://github.com/sbrodeur/speech-compression-asr.git
```

Note that the library must be in the PYTHONPATH environment variable for Python to be able to find it:
```
export PYTHONPATH=$HOME/work/speech-compression-asr:$PYTHONPATH 
```
This can also be added at the end of the configuration file $HOME/.bashrc

## Running unit tests

To ensure all dependencies were correctly installed, it is advised to run the test suite:
```
cd $HOME/work/speech-compression-asr/tests
./run_tests.sh
```
Note that this can take some time.
