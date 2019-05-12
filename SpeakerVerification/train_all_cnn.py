"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import all_cnn as cnn
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.cuda
#import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
from utils import train_load
from utils import dev_load
from utils import test_load
from utils import EER


#def training_plot(metrics):
#    plt.figure(1)
#    plt.plot(metrics[0], 'b')
#    plt.title('Training Loss')
#    plt.savefig('training_loss.png')
#    plt.figure(2)
#    plt.plot(metrics[1], 'b')
#    plt.title('Training Accuracy')
#    plt.savefig('training_accuracy.png')



class DatasetB(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, train_data, train_labels,utt_length,batch_size):
          'Initialization'
          self.train_data = train_data
          self.train_labels = train_labels
          self.l = utt_length
          self.b = batch_size

      def __len__(self):
          'Denotes the total number of samples'
          return len(self.train_data)

      def __getitem__(self, utt_index):
          'Generates one sample of data'
          Num = np.minimum(len(self.train_data) - utt_index, self.b)
          X = [[] for i in range(Num)]
          for i in range(0,Num):
              frame_index = np.random.randint(len(self.train_data[utt_index + i]))
              pad = np.pad(self.train_data[utt_index+i],((0, int(self.l)-1),(0,0)),'wrap')
              X[i] = pad[frame_index:frame_index+self.l]
          Y = self.train_labels[utt_index:utt_index+Num]
          return torch.FloatTensor(np.array(X, dtype = 'float32').reshape(Num,1,self.l,64)), torch.LongTensor(np.array(Y, dtype = 'int64'))



class DevDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dev_data, utt_length):
        'Initialization'
        self.dev_data = dev_data
        self.l = utt_length

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dev_data)

  def __getitem__(self, utt_index):
        'Generates one sample of data'
        frame_index = 0 
        pad = np.pad(self.dev_data[utt_index],((0, int(self.l)-1),(0,0)),'wrap')

        return np.array(pad[frame_index:frame_index+self.l], dtype = 'float32').reshape(1,self.l,64) #################### CHECK



class get_training_stats(): ## NET, DSET, 10000.0, 5, 11

    def __init__(self, mlp, dset,gpu):
        if(gpu == 1):
                self.mlp = mlp.cuda()
        else:
                self.mlp = mlp
        self.dset = dset
        self.gpu = gpu

    def run(self,nepochs,validation):

        criterion = nn.CrossEntropyLoss()  #### Angular Softmax
        self.mlp.train()
        batch_size = 16

        for k in range(0,nepochs):
        ## Training
            total_loss = 0
            total_accuracy = 0
            i = 0
            frame_total = 0.0
            learning_rate = 0.01
            self.optimizer = torch.optim.SGD(self.mlp.parameters(), lr = learning_rate, momentum=0.9, weight_decay = 0.001, nesterov=True) ## Momentum
    
            index = np.arange(0,len(self.dset[0]))
            np.random.shuffle(index)
            train_dataset = DatasetB(self.dset[0][index], self.dset[1][index],14000, batch_size)
    
    
    
            #for train_data_temp, train_labels_temp in train_loader: 
            while (i < (len(self.dset[0])-1)):  # -------------------------------
                train_data_temp = train_dataset[i][0]  # -------------------------------
                train_labels_temp = train_dataset[i][1]
        ## Train Loader
                frame_total = frame_total + len(train_data_temp)
                train_labels_temp = train_labels_temp.reshape(len(train_labels_temp))
        ## Init
                self.optimizer.zero_grad()
    
        ## Forward
                if(self.gpu == 1):
                        x = self.mlp.forward(train_data_temp.float().cuda())
                        y = train_labels_temp.cuda()
                else:
                        x = self.mlp.forward(train_data_temp.float())
                        y = train_labels_temp
    
                cross_entropy = criterion(x, y)
        ## Backward         
                cross_entropy.backward()
        ## Loss       
                loss = cross_entropy.sum()
                total_loss += loss.data[0] 
        ## Step 
                self.optimizer.step()
        ## Classification Error Computation        
                train_prediction = x.cpu().detach().argmax(dim=1)
                train_accuracy = np.sum((train_prediction.cpu().numpy()==y.cpu().numpy()))
                total_accuracy += train_accuracy
                i = i+batch_size
                if (i%1600 == 0):
                    print ("epoch: ",k, "frames: ", frame_total, "training_loss: ", total_loss/frame_total, "training_accuracy: ", total_accuracy/frame_total, "lr: ", learning_rate)
   

##### Validation
        
    def return_cnn(self):
        return self.mlp 

 



def generate_score(x1,x2):  #Embeddings and trials, For test and validation
    cos = nn.CosineSimilarity()
    output = cos(x1, x2)
    return output
    


def main():
    ## Get Data
    print ("Getting Data")
    train_npz = train_load("./", {1,2,3,5})
    train_data = train_npz[0]
    train_labels = train_npz[1]
    #print (len(np.unique(train_labels))) 
    ## 
    dset = (train_npz[0], train_npz[1])

    ## CNN
    #print ("Declaring CNN")
    mlp = cnn.all_cnn_module()

    ## Train
    print ("Training")
    gpu = 1
    trainer = get_training_stats(mlp, dset, gpu=gpu)
    trainer.run(8,0)


    ## Validate
    mlp.eval()
    dev_npz = dev_load('./dev.preprocessed.npz')
    print(len(dev_npz[3]))
    dev_trials = dev_npz[0]
    dev_labels = dev_npz[1]
    dev_data_enroll = dev_npz[2]
    dev_data_test = dev_npz[3]
    dev_dataset_enroll = DevDataset(dev_data_enroll,10000)
    dev_dataset_test = DevDataset(dev_data_test,10000)
    scores = np.zeros(len(dev_trials))
    for i in range(0,len(dev_trials)):
        t1 = dev_trials[i][0]
        t2 = dev_trials[i][1]
        dev_ensemble1 = mlp.forward(torch.cuda.FloatTensor(dev_dataset_enroll[t1][0].reshape(1,1,10000,64)))
        dev_ensemble2 = mlp.forward(torch.cuda.FloatTensor(dev_dataset_test[t2][0].reshape(1,1,10000,64)))
        scores[i] = generate_score(dev_ensemble1, dev_ensemble2)

    print (EER(dev_labels, scores))


    ## Predict
    test_npz = test_load('./test.preprocessed.npz') 
    test_trials = test_npz[0]
    test_dataset_enroll = DevDataset(test_npz[1],14000) ### Check this 5000 for test data
    test_dataset_test = DevDataset(test_npz[2],14000) ### Check this 5000 for test data
    scores = np.zeros(len(test_trials))
    for i in range(0,len(test_trials)):
        t1 = test_trials[i][0]
        t2 = test_trials[i][1]
        scores[i] = generate_score(mlp.forward(torch.cuda.FloatTensor(test_dataset_enroll[t1].reshape(1,1,14000,64)), is_embedding=True), mlp.forward(torch.cuda.FloatTensor(test_dataset_test[t2].reshape(1,1,14000,64)), is_embedding=True))
    np.save('scores.npy',scores)


if __name__ == '__main__':
    main()
