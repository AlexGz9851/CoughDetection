from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
import itertools
import os
import glob
import os.path as path

sampling_freq, audio = wavfile.read("dataset1/data/cough/18448__zippi1__sound-cough1.wav")
mfcc_features = mfcc(audio, sampling_freq)
filterbank_features = logfbank(audio, sampling_freq)

"""
print ('\nMFCC:\nNumber of windows =', mfcc_features.shape[0])
print ('Length of each feature =', mfcc_features.shape[1])
print ('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print ('Length of each feature =', filterbank_features.shape[1])
"""

figure = plt.figure(figsize=(20, 3))
example_data_path = 'dataset1/data/cough/'
file_paths = glob.glob(path.join(example_data_path, "*.wav"))
for idx in range(0):
    sampling_freq, audio = wavfile.read(file_paths[idx])
    mfcc_features = mfcc(audio, sampling_freq, nfft=4096)
    # print(file_paths[idx], mfcc_features.shape[0])
    plt.yscale('linear')
    plt.matshow((mfcc_features.T)[:, :300])
    plt.text(150, -10, idx, horizontalalignment='center', fontsize = 20)

    plt.show()


class HMMTrainer(object):
  def __init__(self, model_name='GaussianHMM', n_components=5 , cov_type='diag', n_iter=1000):
    """test component values"""
    self.model_name = model_name
    self.n_components = n_components
    self.cov_type = cov_type
    self.n_iter = n_iter
    self.models = []
    if self.model_name == 'GaussianHMM':
      self.model = hmm.GaussianHMM(n_components=self.n_components,  covariance_type=self.cov_type,n_iter=self.n_iter)
    else:
      raise TypeError('Invalid model type')

  def train(self, X):
    np.seterr(all='ignore')
    self.models.append(self.model.fit(X))
    # Run the model on input data

  def get_score(self, input_data):
    return self.model.score(input_data)

hmm_models = []
input_folder = 'dataset1/data/'
# Parse the input directory
for dirname in os.listdir(input_folder):
    # Get the name of the subfolder
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder):
        continue
    # Extract the label
    label = subfolder[subfolder.rfind('/') + 1:]
    # Initialize variables
    X = np.array([])
    y_words = []
    # Iterate through the audio files (leaving 1 file for testing in each class)
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)
            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)

            # Append the label
            y_words.append(label)
            # print('X.shape =', X.shape)
    # Train and save HMM model
    hmm_trainer = HMMTrainer(n_components=10)
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, label))
    hmm_trainer = None

input_folder = 'dataset1/test/'
real_labels = []
pred_labels = []
for dirname in os.listdir(input_folder):

  subfolder = os.path.join(input_folder, dirname)
  if not os.path.isdir(subfolder):
    continue
  # Extract the label
  label_real = subfolder[subfolder.rfind('/') + 1:]

  for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
    real_labels.append(label_real)
    filepath = os.path.join(subfolder, filename)
    sampling_freq, audio = wavfile.read(filepath)
    mfcc_features = mfcc(audio, sampling_freq)
    max_score = -9999999999999999999
    output_label = None
    for item in hmm_models:
       hmm_model, label = item
       score = hmm_model.get_score(mfcc_features)
       if score > max_score:
          max_score = score
          output_label = label
    pred_labels.append(output_label)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print("here")
print(real_labels, pred_labels)
cm = confusion_matrix(real_labels, pred_labels)
np.set_printoptions(precision=2)
classes = ["cough","no_cough"]
plt.figure()
plot_confusion_matrix(cm, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()