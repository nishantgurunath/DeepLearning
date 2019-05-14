import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
#from ctcdecode import CTCBeamDecoder
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from warpctc_pytorch import CTCLoss
import torch.nn.utils.rnn as rnnUtils
from operator import itemgetter
import dataset.phoneme_list as phon
#import Levenshtein as L

class SpeechModelDataLoaderTest(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        # concatenate your articles and build into batches

        i = 0
        num_utt = len(self.dataset)
        print (num_utt)
        while(i<num_utt):
            out_data = []
            if torch.cuda.is_available():
                out_data.append(torch.cuda.FloatTensor(self.dataset[i]))
            else:
                out_data.append(torch.FloatTensor(self.dataset[j]))
            yield (out_data)
            i = i + 1
 

class SpeechModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # concatenate your articles and build into batches
        index = np.arange(len(self.dataset[0]))
        if(self.shuffle==True):
            np.random.shuffle(index)
        data = self.dataset[0][index]   
        labels = self.dataset[1][index]                                      ## Shuffle
        num_batches = len(data)//self.batch_size
        print (num_batches)
        num_res = len(data)%self.batch_size
        num_utt = num_batches*self.batch_size

        i = 0
        while(i<num_batches):
            out_data = []
            out_labels = []
            target_lengths = []
            s = i*self.batch_size
            e = (i+1)*self.batch_size
            index1 = np.arange(s,e)
            lengths = [len(data[j]) for j in range(s,e)] 
            l1, l2 = [list(x) for x in zip(*sorted(zip(lengths, index1), key=itemgetter(0), reverse = True))]


            if torch.cuda.is_available():
                out_data.append([torch.cuda.FloatTensor(data[l2[j]]) for j in range(0,e-s)])
                for j in l2:
                    target_lengths.append(len(labels[j]))
                    for k in range(0,len(labels[j])):
                        out_labels.append(labels[j][k])
            else:
                out_data.append([torch.FloatTensor(data[l2[j]]) for j in range(0,e-s)])
                for j in l2:
                    target_lengths.append(len(labels[j]))
                    for k in range(0,len(labels[j])):
                        out_labels.append(labels[j][k])
            yield (out_data[-1], torch.cuda.LongTensor(out_labels), torch.cuda.IntTensor(target_lengths))
            i = i + 1
            

        ## Residual Data
        #L = len(data)
        #if torch.cuda.is_available():
        #        yield ([torch.cuda.FloatTensor(data[j]) for j in range(L-num_res,L)], [torch.cuda.LongTensor(labels[j]) for j in range(L-num_res,L)])
        #else:
        #        yield ([torch.FloatTensor(data[j]) for j in range(L-num_res,L)], [torch.LongTensor(labels[j]) for j in range(L-num_res,L)])





class DigitsModel(nn.Module):

    def __init__(self):
        super(DigitsModel, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.ELU()
            #nn.Conv2d(8, 16, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            #nn.ELU(),
        )
        #self.rnns = nn.ModuleList()
        self.nchannels = 1
        embed_size = 40*self.nchannels
        self.hidden_size = 512
        self.nlayers = 5
        directions = 2
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers = self.nlayers, bidirectional=True)
        self.linear_layer = nn.Linear(in_features=self.hidden_size*directions, out_features=self.hidden_size)
        nn.init.uniform_(self.linear_layer.weight.data,-0.1,0.1)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(in_features=self.hidden_size, out_features=47)
        nn.init.uniform_(self.output_layer.weight.data,-0.1,0.1)
        bias = np.load('logprob.npy')
        self.output_layer.bias.data = torch.FloatTensor(bias)      

    def forward(self, features):
        # features: n, t(variable), f
        n = len(features)
        f = len(features[0][0])
        #features = rnnUtils.pack_sequence(features)
        #features,lengths = rnnUtils.pad_packed_sequence(features, batch_first=False, padding_value=0.0)
        #features = features.unsqueeze(1)
        #embedding = self.embed(features)
#        print (embedding.shape)
        #embedding = embedding.view(-1,n,self.nchannels*f)
#        print (embedding.shape)
        ## PACKING
        hidden = (torch.zeros(self.nlayers,n,self.hidden_size).cuda(),torch.zeros(self.nlayers,n,self.hidden_size).cuda())
        h = rnnUtils.pack_sequence(features,hidden)
        h, _ = self.rnn(h)
#        print (h.shape)

        ## PADDING
        h,lengths = rnnUtils.pad_packed_sequence(h, batch_first=False, padding_value=0.0)
        h = self.linear_layer(h)
        h = self.relu(h)
        logits = self.output_layer(h)

        return logits, lengths

class CTCCriterion(CTCLoss):
    def forward(self, prediction, target):
        acts = prediction[0]
        act_lens = prediction[1].int()
        label_lens = prediction[2].int()
        labels = (target + 1).view(-1).int()
        return super(CTCCriterion, self).forward(
            acts=acts,
            labels=labels.cpu(),
            act_lens=act_lens.cpu(),
            label_lens=label_lens.cpu()
        )


class ER:

    def __init__(self):
        self.label_map = [' '] + phon.PHONEME_MAP
        self.decoder = CTCBeamDecoder(
            labels=self.label_map,
            blank_id=0
        )

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        logits = prediction[0]
        feature_lengths = prediction[1].int()
        labels = target[0] + 1
        target_lengths = target[1]
        logits = torch.transpose(logits, 0, 1)
        logits = logits.cpu()
        probs = F.softmax(logits, dim=2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=feature_lengths)
        pos = 0
        ls = 0.

        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            true = "".join(self.label_map[l] for l in labels[pos:pos + int(target_lengths[i])])
            #print("Pred: {}, True: {}".format(pred, true))
            pos += target_lengths[i]
            ls += L.distance(pred, true)
        assert pos == labels.size(0)
        return ls / output.size(0)

def run_eval(model, test_dataset):
    model.eval()
    error_rate_op = ER()
    loader = SpeechModelDataLoader(test_dataset, shuffle=False, batch_size=16)
    predictions = []
    feature_lengths = []
    labels = []
    target_lengths = []
    i = 0
    for data_batch, labels_batch, target_lengths_batch in loader:
        predictions_batch, feature_lengths_batch = model(data_batch)
        predictions.append(predictions_batch.to("cpu"))
        feature_lengths.append(feature_lengths_batch.to("cpu"))
        labels.append(labels_batch.cpu())
        target_lengths.append(target_lengths_batch.to("cpu"))
        i = i + 1
    predictions = rnnUtils.pad_sequence(predictions) 
    predictions = predictions.view(predictions.size(0),-1,predictions.size(3))
    labels = torch.cat(labels, dim=0)
    feature_lengths = torch.cat(feature_lengths, dim=0)
    target_lengths = torch.cat(target_lengths, dim=0)
    error = error_rate_op((predictions, feature_lengths), (labels.view(-1), target_lengths))
    return error


def run_test(model, test_dataset):
    label_map = [' '] + phon.PHONEME_MAP
    model.eval()
    decoder = CTCBeamDecoder(
            labels=label_map,
            blank_id=0
        )
    model.eval()
    loader = SpeechModelDataLoaderTest(test_dataset)
    predictions = []
    feature_lengths = []
    for data_batch in loader:
        predictions_batch, feature_lengths_batch = model(data_batch)
        predictions.append(predictions_batch.to("cpu"))
        feature_lengths.append(feature_lengths_batch.to("cpu"))

    predictions = rnnUtils.pad_sequence(predictions) 
    predictions = predictions.view(predictions.size(0),-1,predictions.size(3))
    feature_lengths = torch.cat(feature_lengths, dim=0)
    logits = torch.transpose(predictions, 0, 1)
    logits = logits.cpu()
    probs = F.softmax(logits, dim=2)
    output, scores, timesteps, out_seq_len = decoder.decode(probs=probs, seq_lens=feature_lengths)
    pos = 0
    File = open("submission.csv", "w")
    File.write('ID,Predicted\n')
    k = 0
    for i in range(output.size(0)):
        pred = "".join(label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
        File.write('%d,' % k)
        File.write('%s\n' % pred)
        k += 1
        pos += len(output[i, 0, 0:out_seq_len[i, 0]])
    File.close()
    return None





def run():
    best_eval = None
    epochs = 4
    batch_size = 16
    model = DigitsModel()
    model = model.cuda() if torch.cuda.is_available() else model

    data = np.load('dataset/wsj0_train.npy', encoding='bytes')
    labels = np.load('dataset/wsj0_train_merged_labels.npy')
    dataset = (data,labels)
    data = np.load('dataset/wsj0_dev.npy', encoding='bytes')
    labels = np.load('dataset/wsj0_dev_merged_labels.npy')
    evalset = (data,labels)
    testset = np.load('dataset/wsj0_test.npy', encoding='bytes') 

    loader = SpeechModelDataLoader(dataset, shuffle=True, batch_size=batch_size)
    ctc = CTCCriterion()

    for e in range(epochs):
        lr = 0.001*(0.1**(e/2))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        epoch_loss = 0
        i = 0
        for data_batch, label_batch, target_lengths in loader:
            optimizer.zero_grad()
            #logits, input_lengths = model(data_batch.unsqueeze(1)) ## Unsqeeze only required for CNN embedding
            logits, input_lengths = model(data_batch)
            loss = ctc.forward((logits, input_lengths, target_lengths), label_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            i = i + 1
            if(i%100==0):
                print("Epoch: ", e, "Iter: ", i,"Loss: {}".format(epoch_loss / (batch_size*(i+1))))
                #break
        if (e+1) % 4 == 0:
            with torch.no_grad():
                avg_ldistance = run_eval(model, evalset)
            if best_eval is None or best_eval > avg_ldistance:
                best_eval = avg_ldistance
                torch.save(model.state_dict(), "models/checkpoint.pt")
            print("Eval: {}".format(avg_ldistance))
        print("Loss: {}".format(epoch_loss /(batch_size*(i+1))))
    run_test(model,testset)





model = DigitsModel(); 
A = torch.FloatTensor(np.ones((20, 100, 40)))
#print (A.size()[-1])

#print (torch.zeros((10,)))
B = [torch.LongTensor([1,2]),torch.LongTensor([3])]
#print (B.view(-1))

#lengths = [len(B[i]) for i in range(len(B))] 
#index = np.arange(len(B))
#C = [[5],[6]]
#
#
#l1, l2 = [list(x) for x in zip(*sorted(zip(lengths, index), key=itemgetter(0), reverse = True))]
#
#print (lengths, l1)
#print (index, l2)

#print (sorted((lengths,index), key=getkey))
#B = rnnUtils.pack_sequence(B)
#print (B)
#B = rnnUtils.pad_packed_sequence(B, batch_first=False, padding_value=0.0)
#print(B[0])
#print(B[1])
#padpack = rnnUtils.PackedSequence(tup)
#print (C)
#print (B.size()[-1])
#print (model(A)[0].shape)
#print (model(A)[1])

#data = np.load('dataset/wsj0_train.npy', encoding='bytes')
labels = np.load('dataset/wsj0_train_merged_labels.npy')
#dataset = (data,labels)
#


## Labels Probability Distribution

labels_flat = []

for i in range(len(labels)):
    for j in range(len(labels[i])):
        labels_flat.append(labels[i][j])

uniq, counts = np.unique(np.array(labels_flat), return_counts=True)
prob = counts/len(labels_flat)
prob = np.append(1.0,prob)
logprob = np.log(prob)
np.save('logprob.npy',logprob)
#print (logprob)
run()
