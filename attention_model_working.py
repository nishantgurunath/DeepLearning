import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.utils.rnn as rnnUtils
from operator import itemgetter
import Levenshtein as L
from vocab import VOCAB
from vocab import VOCAB_MAP
import matplotlib.pyplot as plt


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
                out_data.append(torch.FloatTensor(self.dataset[i]))
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
            total_chars = 0
            s = i*self.batch_size
            e = (i+1)*self.batch_size
            index1 = np.arange(s,e)
            lengths = [len(data[j]) for j in range(s,e)] 
            l1, l2 = [list(x) for x in zip(*sorted(zip(lengths, index1), key=itemgetter(0), reverse = True))]


            if torch.cuda.is_available():
                out_data.append([torch.cuda.FloatTensor(data[l2[j]]) for j in range(0,e-s)])
             
            else:
                out_data.append([torch.FloatTensor(data[l2[j]]) for j in range(0,e-s)])

            for j in l2:
                target_lengths.append(len(labels[j]))
                out_labels.append(labels[j])
            max_length = max(target_lengths)
            mask_labels = np.zeros((len(out_labels),max_length))
            for j in range(len(out_labels)):
                mask_labels[j][0:target_lengths[j]] = np.ones(target_lengths[j])
                out_labels[j] = np.concatenate((out_labels[j],np.full(((max_length - target_lengths[j]),),0)))
            total_chars = sum(target_lengths) - self.batch_size
            out_labels = torch.from_numpy(np.array(out_labels)).long().cuda() if torch.cuda.is_available() else torch.from_numpy(np.array(out_labels)).long()
            mask_labels= torch.from_numpy(mask_labels).float().cuda() if torch.cuda.is_available() else torch.from_numpy(mask_labels).float()
            target_lengths = torch.from_numpy(np.array(target_lengths)).float().cuda() if torch.cuda.is_available() else torch.from_numpy(np.array(target_lengths)).float()
            #total_chars= torch.from_numpy(total_chars).float().cuda() if torch.cuda.is_available() else torch.from_numpy(total_chars).float()
                
            yield (out_data[-1], out_labels, mask_labels, target_lengths)
            i = i + 1
            

        ## Residual Data
        #L = len(data)
        #if torch.cuda.is_available():
        #        yield ([torch.cuda.FloatTensor(data[j]) for j in range(L-num_res,L)], [torch.cuda.LongTensor(labels[j]) for j in range(L-num_res,L)])
        #else:
        #        yield ([torch.FloatTensor(data[j]) for j in range(L-num_res,L)], [torch.LongTensor(labels[j]) for j in range(L-num_res,L)])

class Listener(nn.Module):

    def __init__(self):
        super(Listener, self).__init__()
        self.embed_size = 40
        self.hidden_size_0 = 256
        self.hidden_size_1 = 256
        self.hidden_size_2 = 256
        self.hidden_size_3 = 256
        self.nlayers = 1
        self.directions = 2
        self.rnn_0 = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size_0, num_layers = self.nlayers, bidirectional=True)
        #nn.init.uniform_(self.rnn_0.weight_ih_l0.data,-0.1,0.1)
        #nn.init.uniform_(self.rnn_0.weight_hh_l0.data,-0.1,0.1)
        self.rnn_1 = nn.LSTM(input_size=self.hidden_size_0*self.directions*2, hidden_size=self.hidden_size_1, num_layers = self.nlayers, bidirectional=True)
        #nn.init.uniform_(self.rnn_1.weight_ih_l0.data,-0.1,0.1)
        #nn.init.uniform_(self.rnn_1.weight_hh_l0.data,-0.1,0.1)
        self.rnn_2 = nn.LSTM(input_size=self.hidden_size_1*self.directions*2, hidden_size=self.hidden_size_2, num_layers = self.nlayers, bidirectional=True)
        #nn.init.uniform_(self.rnn_2.weight_ih_l0.data,-0.1,0.1)
        #nn.init.uniform_(self.rnn_2.weight_hh_l0.data,-0.1,0.1)
        self.rnn_3 = nn.LSTM(input_size=self.hidden_size_2*self.directions*2, hidden_size=self.hidden_size_3, num_layers = self.nlayers, bidirectional=True)
        #nn.init.uniform_(self.rnn_3.weight_ih_l0.data,-0.1,0.1)
        #nn.init.uniform_(self.rnn_3.weight_hh_l0.data,-0.1,0.1)
        self.key_layer = nn.Linear(in_features=self.hidden_size_3*self.directions, out_features=256)
        #nn.init.normal_(self.key_layer.weight.data)
        self.val_layer = nn.Linear(in_features=self.hidden_size_3*self.directions, out_features=256)
        #nn.init.normal_(self.val_layer.weight.data)

    def forward(self, features):
        # features: n, t(variable), f
        n = len(features)
        f = len(features[0][0])
        ## PACKING
        #hidden = (torch.zeros(self.nlayers,n,self.hidden_size).cuda(),torch.zeros(self.nlayers,n,self.hidden_size).cuda())
        features = rnnUtils.pack_sequence(features)

        h0, _ = self.rnn_0(features)
        seq_len0 = len(h0[1])
        h0,lengths0 = rnnUtils.pad_packed_sequence(h0, batch_first=False, padding_value=0.0)
        lengths0 = lengths0//2 
        h0 = h0[0:lengths0[0]*2].transpose(0,1)
        h0_size = h0.size()
        h0 = h0.contiguous().view(h0_size[0], int(h0_size[1]/2), h0_size[2]*2)
        h0 = h0.transpose(0,1)
        #print (h0.shape)
        h0 = rnnUtils.pack_padded_sequence(h0, lengths0, batch_first=False)

        h1, _ = self.rnn_1(h0)
        seq_len1 = len(h1[1])
        h1,lengths1 = rnnUtils.pad_packed_sequence(h1, batch_first=False, padding_value=0.0)
        lengths1 = lengths1//2 
        h1 = h1[0:lengths1[0]*2].transpose(0,1)
        h1_size = h1.size()
        h1 = h1.contiguous().view(h1_size[0], int(h1_size[1]/2), h1_size[2]*2)
        h1 = h1.transpose(0,1)
        #print (h1.shape)
        h1 = rnnUtils.pack_padded_sequence(h1, lengths1, batch_first=False)

        h2, _ = self.rnn_2(h1)
        seq_len2 = len(h2[1])
        h2,lengths2 = rnnUtils.pad_packed_sequence(h2, batch_first=False, padding_value=0.0)
        lengths2 = lengths2//2 
        h2 = h2[0:lengths2[0]*2].transpose(0,1)
        h2_size = h2.size()
        h2 = h2.contiguous().view(h2_size[0], int(h2_size[1]/2),h2_size[2]*2)
        h2 = h2.transpose(0,1)
        #print (h2.shape)
        h2 = rnnUtils.pack_padded_sequence(h2, lengths2, batch_first=False)

        h3, _ = self.rnn_3(h2)
        h3,lengths3 = rnnUtils.pad_packed_sequence(h3, batch_first=False, padding_value=0.0)
        if(torch.cuda.is_available()):
            mask = torch.zeros((h3.size()[1], h3.size()[0])).cuda()
        else:
            mask = torch.zeros((h3.size()[1], h3.size()[0]))
        for i in range(h3.size()[1]):
            mask[i][0:lengths3[i]] = torch.ones(lengths3[i])
        key = self.key_layer(h3)
        value = self.val_layer(h3)
        return key, value, mask

class Speller(nn.Module):

    def __init__(self):
        super(Speller, self).__init__()
        self.embed_size = 256
        self.hidden_size = 512
        self.decodeWidth = 160
        self.cell0 = nn.LSTMCell(input_size=self.hidden_size , hidden_size=self.hidden_size)
        #nn.init.uniform_(self.cell0.weight_ih.data,-0.1,0.1)
        #nn.init.uniform_(self.cell0.weight_hh.data,-0.1,0.1)
        self.cell1 = nn.LSTMCell(input_size=self.hidden_size , hidden_size=self.hidden_size)
        #nn.init.uniform_(self.cell1.weight_ih.data,-0.1,0.1)
        #nn.init.uniform_(self.cell1.weight_hh.data,-0.1,0.1)
        #self.cell2 = nn.LSTMCell(input_size=self.hidden_size , hidden_size=self.hidden_size)
        self.embed = nn.Embedding(34, self.embed_size)
        self.softmax = nn.Softmax(dim=1)
        self.query = nn.Linear(in_features=self.hidden_size, out_features=self.embed_size)
        self.scoring = nn.Linear(in_features=(self.embed_size + self.hidden_size), out_features=34)
        #nn.init.uniform_(self.scoring.weight.data,-0.1,0.1)
        #nn.init.normal_(self.scoring.weight.data)
        if(torch.cuda.is_available()): 
            self.s0 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=True)
            self.s1 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=True) # Output LSTM : Query
            self.cs0 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=True) # Output LSTM : Query
            self.cs1 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=True) # Output LSTM : Query
        else:
            self.s0 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.FloatTensor), requires_grad=True)
            self.s1 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.FloatTensor), requires_grad=True) # Output LSTM : Query
            self.cs0 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.FloatTensor), requires_grad=True) # Output LSTM : Query
            self.cs1 = nn.Parameter(torch.zeros(self.hidden_size).type(torch.FloatTensor), requires_grad=True) # Output LSTM : Query

    def forward(self, key, value, mask, labels, training=True): # key : (S,B,E), value : (S,B,E), mask : (B,S), labels : (B,S) 
## ATTENTION
    ## INITIALIZATION
        y = []
        attention = []

        seq_size, batch_size, embed_size = value.size()
        s0 = (self.s0.unsqueeze(0)).expand(batch_size,-1)
        s1 = (self.s1.unsqueeze(0)).expand(batch_size,-1)
        cs0 = (self.cs0.unsqueeze(0)).expand(batch_size,-1)
        cs1 = (self.cs1.unsqueeze(0)).expand(batch_size,-1)

        key = key.transpose(0,1)
        value = value.transpose(0,1)
        labels = labels.transpose(0,1) if (training) else labels

        query = self.query(s1)
        e = torch.bmm(key,query.unsqueeze(2)).view(batch_size,seq_size) # Energy: <Query,Key>
        e = self.softmax(e) # Softmax
        e = e*mask # Mask
        alpha = torch.nn.functional.normalize(e, p = 1, dim = 1)
        c = torch.bmm(alpha.unsqueeze(1),value).view(batch_size,embed_size) # Context: Values

        if (training == False):
            attention.append(alpha[0].cpu().detach().numpy())

        if(torch.cuda.is_available()):
            embeds = self.embed(labels) if (training) else self.embed(torch.ones(batch_size).long().cuda()*33)    # Character Embedding
        else:
            embeds = self.embed(labels) if (training) else self.embed(torch.ones(batch_size).long()*33)    # Character Embedding

        if (training == True):
            inp = torch.cat((embeds[0],c),dim=1)  
        else:
            inp = torch.cat((embeds,c),dim=1)  
        s0,cs0 = self.cell0(inp, (s0, cs0))
        s1,cs1 = self.cell1(s0, (s1,cs1))
        #s2,cs2 = self.cell2(s1, (s2.view(batch_size,self.hidden_size),cs2))
        

    ## TRAINING
        if(training):
            for i in range(1, len(labels)):

                query = self.query(s1)
                e = torch.bmm(key,query.unsqueeze(2)).view(batch_size,seq_size)
                e = self.softmax(e) # Softmax
                e = e*mask
                alpha = torch.nn.functional.normalize(e, p = 1, dim = 1)
                c = torch.bmm(alpha.unsqueeze(1),value).view(batch_size,self.embed_size)
                out = torch.cat((s1,c), dim=1)
                out = self.scoring(out)
                y.append(out)


                rand = np.random.binomial(1,1.0)
                if(rand == 1):
                    #inp = self.embed(labels[i])
                    inp = embeds[i]
                else:
                    arg = torch.max(out,1)[1]
                    #arg = torch.multinomial(self.softmax(out),1,replacement=True).view(-1)
                    inp = self.embed(arg)
                    
                inp = torch.cat((inp,c),dim=1)

                s0,cs0 = self.cell0(inp, (s0,cs0))
                s1,cs1 = self.cell1(s0, (s1,cs1))
                #s2,cs2 = self.cell2(s1, (s2,cs2))

            ## Last Output
            query = self.query(s1)
            e = torch.bmm(key,query.unsqueeze(2)).view(batch_size,seq_size)
            e = self.softmax(e) # Softmax
            e = e*mask
            alpha = torch.nn.functional.normalize(e, p = 1, dim = 1)
            c = torch.bmm(alpha.unsqueeze(1),value).view(batch_size,embed_size)
            out = torch.cat((s1,c), dim=1)
            out = self.scoring(out)
            y.append(out)

        else:
            i = 0
            while(i<self.decodeWidth-1):
                ## Greedy
                query = self.query(s1)
                e = torch.bmm(key,query.unsqueeze(2)).view(batch_size,seq_size)
                e = self.softmax(e) # Softmax
                e = e*mask
                alpha = torch.nn.functional.normalize(e, p = 1, dim = 1)
                attention.append(alpha[0].cpu().detach().numpy())
                c = torch.bmm(alpha.unsqueeze(1),value).view(batch_size,embed_size)
                out = torch.cat((s1,c), dim=1)
                out = self.scoring(out)
                y.append(out)
                arg = torch.max(out,1)[1]
                #arg = torch.multinomial(self.softmax(out),1,replacement=True).view(-1)
                inp = self.embed(arg)
                inp = torch.cat((inp,c),dim=1)
                s0,cs0 = self.cell0(inp, (s0,cs0))
                s1,cs1 = self.cell1(s0, (s1,cs1))
                #s2,cs2 = self.cell2(s1, (s2,cs2))
                i = i + 1

            query = self.query(s1)
            e = torch.bmm(key,query.unsqueeze(2)).view(batch_size,seq_size)
            e = self.softmax(e) # Softmax
            e = e*mask
            alpha = torch.nn.functional.normalize(e, p = 1, dim = 1)
            c = torch.bmm(alpha.unsqueeze(1),value).view(batch_size,embed_size)
            out = torch.cat((s1,c), dim=1)
            out = self.scoring(out)
            y.append(out)

        np.save('./attention.npy',attention)
        y = torch.cat(y).view(len(y),batch_size, 34)
        y = y.transpose(0,1)
        #print (self.scoring.weight.grad)
        return y


def prediction_random_search(listen, spell, data_batch, label_batch): # B x S x 34
    key, value, mask = listen(data_batch)
    logits = spell(key, value, mask, label_batch, training=False)
    batch_size, seq_size, class_size = logits.size()

    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    out = np.zeros((batch_size,seq_size))
    for i in range(len(logits)):
        outcomes = torch.multinomial(softmax(logits[i]),100,replacement=True)
        outcomes = outcomes.transpose(0,1)
        loss = []
        for j in range(100):
            x = spell(key[:,i:i+1,:], value[:,i:i+1,:], mask, outcomes[j][:-1].unsqueeze(0), training=True)
            loss.append(criterion(x.view(-1,34),outcomes[j][1:]))
        out[i] = outcomes[np.argmin(loss)].cpu().detach().numpy()
    return out



def run_eval(listen, spell, test_dataset):
    listen.eval()
    spell.eval()
    loader = SpeechModelDataLoader(test_dataset, shuffle=False, batch_size=1)
    k = 0
    ls = 0
    for data_batch, label_batch, mask_label_batch, total_chars in loader:
        key, value, mask = listen(data_batch)
        x = spell(key, value, mask, label_batch[:,:-1], training=False)
        #print (x)
        out = x.cpu().detach().argmax(dim=2).numpy()
        #print (out)
        #out = prediction_random_search(listen, spell, data_batch, label_batch);
        for i in range(out.shape[0]):
            j = 0
            pred = ''
            true = ''
            while((j < len(out[i])) and (out[i][j] != 32)):
                pred = pred + "".join(VOCAB_MAP[out[i][j]])   ## Prun
                j += 1
            j = 1
            while((j < len(label_batch[i])) and (mask_label_batch[i][j].cpu().detach().numpy() == 1)):
                true = true + "".join(VOCAB_MAP[label_batch[i][j].item()]) ## Mask
                j += 1
            print("Pred: {}, True: {}".format(pred, true[:-1]))
            ls += L.distance(pred, true[:-1])
        k = k + 1
        if(k==1):
            break
    ls = ls/(k);
    return ls


def run_test(listen, spell, test_dataset):
    listen.eval()
    spell.eval()
    loader = SpeechModelDataLoaderTest(test_dataset)
    feature_lengths = []
    File = open("submission.csv", "w")
    File.write('Id,Predicted\n')
    k = 0
    for data_batch in loader:
        key, value, mask = listen(data_batch)
        x = spell(key, value, mask, None, training=False)
        out = x.cpu().detach().argmax(dim=2).numpy()
        #out = prediction_random_search(listen, spell, data_batch, None);
        for i in range(out.shape[0]):
            j = 0
            pred = ''
            true = ''
            while((j < len(out[i])) and (out[i][j] != 32)):
                pred = pred + "".join(VOCAB_MAP[out[i][j]])   ## Prun
                j += 1

            File.write('%d,' % k)
            File.write('%s\n' % pred)
            k += 1
        if(k%10==0):
            print (k)
    File.close()
    return None





def run():
    best_eval = None
    epochs = 10
    batch_size = 16
    listen = Listener()
    spell = Speller()
    listen = listen.cuda() if torch.cuda.is_available() else listen
    spell = spell.cuda() if torch.cuda.is_available() else spell
    #listen_state = torch.load('models/listen3.pt')
    #listen.load_state_dict(listen_state)
    #spell_state = torch.load('models/spell3.pt')
    #spell.load_state_dict(spell_state)

    data = np.load('all/train.npy', encoding='bytes')
    labels = np.load('all/train_int_transcripts.npy')
    dataset = (data,labels)
    data = np.load('all/dev.npy', encoding='bytes')
    labels = np.load('all/dev_int_transcripts.npy')
    evalset = (data,labels)
    testset = np.load('all/test.npy', encoding='bytes') 

    loader = SpeechModelDataLoader(dataset, shuffle=True, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(reduction='none')
    #criterion = nn.CrossEntropyLoss()
    lr = 0.001
    params = list(listen.parameters()) + list(spell.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=0.00001)


    for e in range(epochs):
        epoch_loss = 0
        i = 0
        listen.train()
        spell.train()
        #lr = 0.001*(0.1**(e/4))
        #optimizer = optim.Adam(nn.ModuleList([listen,spell]).parameters(), lr=lr, weight_decay=0.001)
        #optimizer = optim.Adam(listen.parameters(), lr=lr, weight_decay=0.001)

        for data_batch, label_batch, mask, total_chars in loader:
            optimizer.zero_grad()

            key, value, att_mask = listen(data_batch)
            x = spell(key, value, att_mask, label_batch[:,:-1])


            loss = criterion(x.permute(0,2,1),label_batch[:,1:])
            loss = loss*(mask[:,1:])
            #loss = criterion(x.contiguous().view(-1,34),label_batch[:,1:].contiguous().view(-1))
            #loss = loss*(mask[:,1:].contiguous().view(-1))
            loss = loss.sum(dim = 1)/(total_chars)
            loss = loss.mean()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(nn.ModuleList([listen,spell]).parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            i = i + 1

            if(i%100==0):
                print("Epoch: ", e, "Iter: ", i,"Perplexity: {}".format(np.exp(epoch_loss / (i))), "Loss: ", (epoch_loss/(i)))
            #if(i%1==0):
            #    break
        if (e+1) % 1 == 0:
            with torch.no_grad():
                avg_ldistance = run_eval(listen, spell, evalset)
            torch.save(listen.state_dict(), "models/listen" + str(e) + ".pt")
            torch.save(spell.state_dict(), "models/spell" + str(e) + ".pt")
            print("Eval: {}".format(avg_ldistance))
        print("Perplexity: {}".format(np.exp(epoch_loss / (i))))
   # run_test(listen,spell,testset)




## Model

#listen = Listener(); 
#spell = Speller();
#A = torch.FloatTensor(np.ones((20, 100, 40)))
#B = [torch.FloatTensor([np.random.randn(40),np.random.randn(40),np.random.randn(40)]*10),torch.LongTensor([np.random.randn(40),np.random.randn(40)]*10),torch.LongTensor([np.random.randn(40),np.random.randn(40)]*10)]


#key,value,mask = listen(B)
#print (key.shape)
#inputs = torch.from_numpy(np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]]))
#out = spell(key,value,mask,inputs,training=True)
#print (out.shape)
#print(prediction_random_search(listen,spell,B,inputs))

run()




## Transcripts Generation

#data = np.load('./all/train.npy', encoding='bytes')
#transcripts = np.load('./all/train_int_transcripts.npy')
#transcripts = np.load('./all/dev_transcripts.npy')
#print (VOCAB[transcripts[2000][1]])
#print ((np.unique(list(''.join(transcripts)))))
#print ((np.unique(np.concatenate(transcripts))))
#dist =  [len(transcripts[i]) for i in range(len(transcripts))]
#plt.hist(dist,bins=10)
#plt.show()



#
#print (len(data), len(transcripts))
#print (len(data[0]), len(transcripts[0]))
#print (len(data[1]), len(transcripts[1]))
#print (len(data[2000]), len(transcripts[2000]))
#print (transcripts[2000])
#print (transcripts[0])

#transcripts_int = [[] for i in range(len(transcripts))]

#for k in range(len(transcripts)):
#    transcripts[k] = list(transcripts[k])
#    for i in range(len(transcripts[k])):
#        transcripts[k][i] = transcripts[k][i].decode('UTF-8')
#    transcripts[k] = ' '.join(transcripts[k])
#    transcripts[k] = "s" + transcripts[k] + "e"
#    for i in range(len(transcripts[k])):
#        transcripts_int[k].append(VOCAB[transcripts[k][i]])
#np.save('./all/train_char_transcripts.npy', transcripts)
#np.save('./all/train_int_transcripts.npy', transcripts_int)
#np.save('./all/dev_int_transcripts.npy', transcripts_int)
#print (transcripts[0][0])
#print (transcripts_int[0][0])
#print (transcripts[0][1])
#print (transcripts_int[0][1])
#print (len(transcripts_int[2000]))



## Data Loader
#dset = (data,transcripts)
#train_loader = SpeechModelDataLoader(dset,3,shuffle=False)

#i = 0
#for data,labels,mask,char in train_loader:
#    print (labels.shape)
#    print (mask.shape)
#    print (data)
#    print (dset[0][0:3])
#    print (labels)
#    print (dset[1][0:3])
#    i = i + 1
#    if(i==1):
#        break
