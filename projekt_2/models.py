import os
import librosa
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib
import random
import itertools
import IPython.display as ipd
import librosa.display
import torch.nn as nn

TRAIN_PATH = '/content/gdrive/My Drive/ptaki/ptaki/train'
TEST_PATH = '/content/gdrive/My Drive/ptaki/ptaki/test'
MAIN_PATH = '/content/gdrive/My Drive/ptaki/ptaki'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_log_mel(file_name, start=0, stop=None, n_mels=60):
    samples, sample_rate = librosa.core.load(file_name, sr = None)
    samples = samples[int(start * sample_rate):int(stop * sample_rate) if stop else None]
    spectrogram = librosa.feature.melspectrogram(y = samples, sr = sample_rate,
                                                 n_mels = n_mels, fmin = 4000, fmax = 9500)
    
    log_spec = librosa.core.power_to_db(spectrogram, ref=np.median)
    return log_spec


def load_spec(file_name, start=0, stop=None):
    sample_rate, samples = wavfile.read(file_name)
    samples = samples[int(start * sample_rate):int(stop * sample_rate) if stop else None]
    _, _, spectrogram = signal.spectrogram(samples, sample_rate)
    return spectrogram


def load_test(load_repr=load_log_mel):
    with open('sampleSubmission.csv', 'r') as file:
        lines = file.read().split()[1:]
        sample_ids = [line.split(',')[0] for line in lines]
        samples = np.array([s.split('/') for s in sample_ids])
    
    X_test = []
    rec_files = sorted([file_name for file_name in os.listdir('test') 
                        if file_name.endswith('.wav')], key=lambda x: int(x.split('.')[0][3:]))
    for file_name in rec_files:
        recording_id = file_name.split('.')[0][3:]
        time_markers = samples[samples[:, 0] == recording_id, 1].astype(np.int)
        for t in time_markers:
            representation = load_repr(os.path.join('test', file_name), start = t, stop = t + 1)
            X_test.append(representation)
    return np.array(X_test)


def read_labels():
    labels = []
    with open(os.path.join('train', 'labels.txt'), 'r') as file:
        text = file.read()
        for line in text.split('\n')[1:]:
            if len(line) > 1:
                rec, start, stop = line.split(',')
                rec, start, stop = int(rec[3:]), float(start), float(stop)
                labels.append([rec, start, stop])
    return np.array(labels)


def check_voices(second, labels, tol=0.):
    return (labels[1] >= second and labels[1] < second + 1 - tol) or \
           (labels[2] < second + 1 and labels[2] > second + tol) or \
           (labels[1] < second and labels[2] > second + 1)


def map_seconds_to_y(labels):
    y = [0] * 10
    y_restrictive = [0] * 10
    for s in range(10):
        for l in labels:
            if check_voices(s, l):
                y[s] = 1
            if check_voices(s, l, 0.02):
                y_restrictive[s] = 1
        if y[s] != y_restrictive[s]:
            y[s] = -1
    return y


def load_train(load_repr=load_log_mel):
    labels = read_labels()
    X_train, y_train = [], []
    rec_files = [file_name for file_name in os.listdir('train') if file_name.endswith('.wav')]
    for file_name in rec_files:
        recording_id = int(file_name.split('.')[0][3:])
        recording_labels = labels[labels[:, 0] == recording_id]
        y_binary = map_seconds_to_y(recording_labels)
        for i, y in enumerate(y_binary):
            if y != -1:
                try:
                    representation = load_repr(os.path.join('train', file_name), start = i, stop = i + 1)
                    X_train.append(representation)
                    y_train.append(y)
                except ValueError:
                    print('Error reading file', file_name)
                except TypeError:
                    print('Unsupported type', file_name)
    return np.array(X_train), np.array(y_train)



def save_predictions(preds):
    with open('sampleSubmission.csv', 'r') as file:
        submission_text = file.read().split()
        header = submission_text[0]
        lines = submission_text[1:]

    output_lines = [header]
    for pred, line in zip(preds, lines):
        output_lines.append("{},{}".format(line.split(',')[0], pred))
    
    with open('mySubmission.csv', 'w') as file:
        file.write('\n'.join(output_lines) + '\n')

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 3 * 5, 64),
            nn.ReLU()
        )
        self.fc2 = torch.nn.Linear(64, 2)
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        #16, 10, 15
        x = x.view(-1, 32 * 3 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        return(x)

class CNN_Wrapper:
    
    def __init__(self, idx, pretrained, path):
        # Should I set another seed in each model? It is needed? Probably not
        self.idx = idx
        self.clf = SimpleCNN()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.clf.parameters())
        self.epochs = 25
        self.serialize_path = path if path else f'saved_model/{self.idx}.pkl'
        self.batch_size = 64
        self.pretrained = pretrained
    
    def build_loaders(self, X, y):
        split_point = int(len(X) * 0.8)

        X_train = torch.Tensor(X[:split_point])
        y_train = torch.LongTensor(y[:split_point])

        X_valid = torch.Tensor(X[split_point:])
        y_valid = torch.LongTensor(y[split_point:])

        dataset = TensorDataset(X_train, y_train)
        data_loader = DataLoader(dataset, self.batch_size, shuffle = True)

        valid_dataset = TensorDataset(X_valid, y_valid)
        valid_data_loader = DataLoader(valid_dataset, self.batch_size)
        
        return data_loader, valid_data_loader, y_valid
    
    def fit(self, XX=None, yy=None):
        if self.pretrained:
            print("This model has already pretrained")
            return
        data_loader, valid_data_loader, y_valid = self.build_loaders(XX, yy)
        
        best_preds, best_score = None, 0.
        losses, scores = [], []
        best_auc = 0.
        for epoch in trange(self.epochs):
            running_loss = 0
            self.clf.train()
            for X, y in data_loader:
                self.optimizer.zero_grad()

                outputs = self.clf(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            losses.append(running_loss)

            self.clf.eval()
            preds = []
            for X, _ in valid_data_loader:
                out = self.clf(X)
                preds.append(torch.softmax(out, dim = 1)[:, 1].detach().numpy())
            preds = np.concatenate(preds, axis = 0)

            # Metryką testującą jest ROC AUC
            score = roc_auc_score(y_valid.numpy(), preds)
            scores.append(score)
            #print(f"AUC score = {score}")
            best_auc = max(best_auc, score)
            if score > best_score:
                best_score = score
                best_preds = preds
                #np.save('tmp_preds', best_preds)

                # Model dający najlepszy wynik powinien być zapisany
                torch.save(self.clf.state_dict(), self.serialize_path)
        
        print(f"Model {self.idx} had AUC = {best_auc}")
        self.plot(scores, losses)
        
    
    def plot(self, scores, losses):
        print(f"Results for {self.idx} model")
        plt.plot(scores)
        plt.show()

        plt.plot(losses)
        plt.show()
    
    def predict(self, X_test):
        self.clf.load_state_dict(torch.load(self.serialize_path))

        X_test_tensor = torch.Tensor(X_test)

        test_dataset = TensorDataset(X_test_tensor)
        test_data_loader = DataLoader(test_dataset, batch_size = self.batch_size)

        self.clf.eval()
        preds = []
        zeros, ones = 0, 0
        for X in test_data_loader:
            out = self.clf(X[0])

            bools_0 = torch.nonzero(out[:, 0] > out[:, 1]).size(0)
            bools_1 = torch.nonzero(out[:, 0] < out[:, 1]).size(0)
            zeros, ones = zeros + bools_0, ones + bools_1

            preds.append(torch.softmax(out, dim = 1)[:, 1].detach().numpy())

        preds = np.concatenate(preds, axis = 0)
        print(zeros, ones)

        #save_predictions(preds)
        return preds
      
    def fitting_on_validation(valid_data_loader):
        pass


class VotingClassifier:
    def __init__(self, models):
        self.models = models
        self.models_cnt = len(models)
        
    def fit(self, X=None, y=None):
        for i, model in enumerate(self.models):
            print(f"Starting Model {i+1}")
            model.fit()
            
    def predict(self, X_test):
        all_labels = []
        for i, model in enumerate(self.models):
            print(f"Predicting Model {i}")
            predictions = model.predict(X_test)
            all_labels.append(predictions)
        return all_labels
    
    def voting(self, labels):
        full_mean_result = torch.zeros(labels[0].shape)
        for i, lab in enumerate(labels):
            full_mean_result = full_mean_result + torch.FloatTensor(lab)
        
        final_labels = full_mean_result/len(labels)
        save_predictions(final_labels)
        print(final_labels)
        return final_labels