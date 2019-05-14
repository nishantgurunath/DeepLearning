"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import all_cnn as cnn
import all_bayes_cnn as bcnn
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.cuda
#import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import os

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
          return torch.FloatTensor(np.array(X, dtype = 'float32').reshape(Num,1,self.l,20)), torch.LongTensor(np.array(Y, dtype = 'int64'))



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

        return np.array(pad[frame_index:frame_index+self.l], dtype = 'float32').reshape(1,self.l,20) #################### CHECK



class get_training_stats(): ## NET, DSET, 10000.0, 5, 11

    def __init__(self, model, dset):
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.dset = dset

    def run(self,nepochs,validation):

        criterion = nn.CrossEntropyLoss()  #### Angular Softmax
        self.model.train()
        batch_size = 16
        for k in range(0,nepochs):
        ## Training
            total_loss = 0
            total_accuracy = 0
            i = 0
            frame_total = 0.0
            learning_rate = 0.001
            #self.optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate, momentum=0.9, weight_decay = 0.001, nesterov=True) ## Momentum
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate) 
            index = np.arange(0,len(self.dset[0]))
            np.random.shuffle(index)
            train_dataset = DatasetB(self.dset[0][index], self.dset[1][index],2000, batch_size)
    
    
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
                if(torch.cuda.is_available()):
                        x = self.model.forward(train_data_temp.float().cuda())
                        y = train_labels_temp.cuda()
                else:
                        x = self.model.forward(train_data_temp.float())
                        y = train_labels_temp
    
                cross_entropy = criterion(x, y)

        ## Backward         
                cross_entropy.backward()
        ## Loss       
                loss = cross_entropy.item()
                total_loss += loss 
        ## Step 
                self.optimizer.step()
        ## Classification Error Computation        
                train_prediction = x.cpu().detach().argmax(dim=1)
                train_accuracy = (train_prediction.cpu().numpy()==y.cpu().numpy())
                total_accuracy += np.sum(train_accuracy)
                i = i+batch_size
                #if (i%400 == 0):
            print ("epoch: ",k, "frames: ", frame_total, "training_loss: ", total_loss/frame_total, "training_accuracy: ", total_accuracy/frame_total, "lr: ", learning_rate)
                    #break
            if((k+1)%5 ==0):        
                model_path = os.path.join('experiments','model-{}.pkl'.format(k))
                torch.save({'state_dict': self.model.state_dict()}, model_path)




    


def main():
    ## Get Data
#gcloud compute --project "plucky-command-222002" ssh --zone "us-east4-b" "instance-1"
    print ("Getting Data")
#    train_data = np.load('./dataset/train_data_pert.npy')
#    train_labels = np.load('./dataset/train_labels_pert.npy') 
#    print (len(np.unique(train_labels)))
#    print (len(train_data))
#    print (len(train_labels))
#    lengths_dist = np.zeros(len(train_data))
#    for i in range(len(train_data)):
#        lengths_dist[i] = len(train_data[i])
#    np.save('ldistribution.npy',lengths_dist)

   ## 
#    dset = (train_data, train_labels)
   ## CNN
    print ("Declaring CNN")
    model = cnn.all_cnn_module()
    #model = bcnn.all_cnn_module()

   ## Train
    print ("Training")
#    state = torch.load('./experiments/model-14.pkl')
#    model.load_state_dict(state['state_dict'])
#    trainer = get_training_stats(model, dset)
#    trainer.run(5,0)


    ## Validate
    print ("Validating")

    state = torch.load('./experiments/model-4.pkl')
    model.load_state_dict(state['state_dict'])
    model.cuda()
    model.eval()




    total_accuracy = 0
    e_accuracy = np.zeros(11)
    e_elements = np.zeros(11)
    e_map = np.zeros(11)

    test_data = np.load('./dataset/test_data_0.npy')
    test_labels = np.load('./dataset/test_labels_0.npy')
    print (len(test_labels))
    print (len(np.unique(test_labels)))
    test_dataset = DatasetB(test_data, test_labels, 2000, 1)
    i = 0
    while (i<(len(test_data)-1)):
        x = model.forward(test_dataset[i][0].cuda()).cpu().detach()
        test_prediction = x.cpu().detach().argmax(dim=1)
        test_accuracy = (test_prediction.cpu().numpy()==test_dataset[i][1].cpu().numpy())
#        print (np.sum(test_accuracy))
        total_accuracy += np.sum(test_accuracy)
        e_accuracy[int(test_dataset[i][1].cpu().numpy())] += np.sum(test_accuracy)
        e_elements[int(test_dataset[i][1].cpu().numpy())] += 1
        e_map[int(test_dataset[i][1].cpu().numpy())] += ((e_accuracy[int(test_dataset[i][1].cpu().numpy())]*np.sum(test_accuracy))/e_elements[int(test_dataset[i][1].cpu().numpy())])
#        print ((e_accuracy[int(test_dataset[i][1].cpu().numpy())]*np.sum(test_accuracy))/e_elements[int(test_dataset[i][1].cpu().numpy())])
#        print (e_map)
        i = i + 1
        if(i%500 ==0):
            print (i, total_accuracy)
#            break


    test_data = np.load('./dataset/test_data_1.npy')
    test_labels = np.load('./dataset/test_labels_1.npy')
    print (len(np.unique(test_labels)))
    test_dataset = DatasetB(test_data, test_labels, 2000, 1)
    i = 0
    while (i<(len(test_data)-1)):
        x = model.forward(test_dataset[i][0].cuda()).cpu().detach()
        test_prediction = x.cpu().detach().argmax(dim=1)
        test_accuracy = (test_prediction.cpu().numpy()==test_dataset[i][1].cpu().numpy())
        total_accuracy += np.sum(test_accuracy)
        e_accuracy[int(test_dataset[i][1].cpu().numpy())] += np.sum(test_accuracy)
        e_elements[int(test_dataset[i][1].cpu().numpy())] += 1
        e_map[int(test_dataset[i][1].cpu().numpy())] += e_accuracy[int(test_dataset[i][1].cpu().numpy())]*np.sum(test_accuracy)/e_elements[int(test_dataset[i][1].cpu().numpy())]
        i = i + 1
        if(i%500 ==0):
            print (i, total_accuracy)


    test_data = np.load('./dataset/test_data_2.npy')
    test_labels = np.load('./dataset/test_labels_2.npy')
    print (len(np.unique(test_labels)))
    test_dataset = DatasetB(test_data, test_labels, 2000, 1)
    i = 0
    while (i<(len(test_data)-1)):
        x = model.forward(test_dataset[i][0].cuda()).cpu().detach()
        test_prediction = x.cpu().detach().argmax(dim=1)
        test_accuracy = (test_prediction.cpu().numpy()==test_dataset[i][1].cpu().numpy())
        total_accuracy += np.sum(test_accuracy)
        e_accuracy[int(test_dataset[i][1].cpu().numpy())] += np.sum(test_accuracy)
        e_elements[int(test_dataset[i][1].cpu().numpy())] += 1
        e_map[int(test_dataset[i][1].cpu().numpy())] += e_accuracy[int(test_dataset[i][1].cpu().numpy())]*np.sum(test_accuracy)/e_elements[int(test_dataset[i][1].cpu().numpy())]
        i = i + 1
        if(i%500 ==0):
            print (i, total_accuracy)

    test_data = np.load('./dataset/test_data_3.npy')
    test_labels = np.load('./dataset/test_labels_3.npy')
    print (len(np.unique(test_labels)))
    test_dataset = DatasetB(test_data, test_labels, 2000, 1)
    i = 0
    while (i<(len(test_data)-1)):
        x = model.forward(test_dataset[i][0].cuda()).cpu().detach()
        test_prediction = x.cpu().detach().argmax(dim=1)
        test_accuracy = (test_prediction.cpu().numpy()==test_dataset[i][1].cpu().numpy())
        total_accuracy += np.sum(test_accuracy)
        e_accuracy[int(test_dataset[i][1].cpu().numpy())] += np.sum(test_accuracy)
        e_elements[int(test_dataset[i][1].cpu().numpy())] += 1
        e_map[int(test_dataset[i][1].cpu().numpy())] += e_accuracy[int(test_dataset[i][1].cpu().numpy())]*np.sum(test_accuracy)/e_elements[int(test_dataset[i][1].cpu().numpy())]
        i = i + 1
        if(i%500 ==0):
            print (i, total_accuracy)


    test_data = np.load('./dataset/test_data_4.npy')
    test_labels = np.load('./dataset/test_labels_4.npy')
    print (len(np.unique(test_labels)))
    test_dataset = DatasetB(test_data, test_labels, 2000, 1)
    i = 0
    while (i<(len(test_data)-1)):
        x = model.forward(test_dataset[i][0].cuda()).cpu().detach()
        test_prediction = x.cpu().detach().argmax(dim=1)
        test_accuracy = (test_prediction.cpu().numpy()==test_dataset[i][1].cpu().numpy())
        total_accuracy += np.sum(test_accuracy)
        e_accuracy[int(test_dataset[i][1].cpu().numpy())] += np.sum(test_accuracy)
        e_elements[int(test_dataset[i][1].cpu().numpy())] += 1
        e_map[int(test_dataset[i][1].cpu().numpy())] += e_accuracy[int(test_dataset[i][1].cpu().numpy())]*np.sum(test_accuracy)/e_elements[int(test_dataset[i][1].cpu().numpy())]
        i = i + 1
        if(i%100 ==0):
            print (i, total_accuracy)

    print ("No. of Events:", e_elements)
#    print (e_accuracy)
#    print (e_map)
    print ("Event Accuracy :", e_accuracy/e_elements)
    print ("Event Map :", e_map/e_accuracy)
    print ("Total Accuracy: ", total_accuracy/np.sum(e_elements))
    print (e_accuracy[1:].dot(e_elements[1:])/np.sum(e_elements[1:]))

if __name__ == '__main__':
    main()
