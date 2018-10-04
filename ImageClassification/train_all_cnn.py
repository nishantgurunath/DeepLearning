"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import CNN.preprocessing as preprocess
import CNN.all_cnn as cnn
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.cuda
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable



def training_plot(metrics):
    plt.figure(1)
    plt.plot(metrics[0], 'b')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.figure(2)
    plt.plot(metrics[1], 'b')
    plt.title('Training Accuracy')
    plt.savefig('training_accuracy.png')



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, train_data, train_labels):
        'Initialization'
        self.train_data = train_data
        self.train_labels = train_labels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_data)

  def __getitem__(self, sample):
        'Generates one sample of data'

        X = self.train_data[sample]
        Y = self.train_labels[sample]

        return X, np.array(Y, dtype = 'int64')


def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file for submission.
    File should be:
        named 'predictions.txt'
        in the root of your tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))




class get_training_stats(): ## NET, DSET, 10000.0, 5, 11

    def __init__(self, mlp, dset):
        self.mlp = mlp.cuda()
        self.dset = dset

    def run(self,nepochs):

        criterion = nn.CrossEntropyLoss()
        training_loss = np.zeros(nepochs)
        training_accuracy = np.zeros(nepochs)
        self.mlp.train()
    
        for k in range(0,nepochs):
        ## Training
            total_loss = 0
            total_accuracy = 0
            i = 0
            frame_total = 0
            learning_rate = 0.001*(0.1**(k/7))
            self.optimizer = torch.optim.SGD(self.mlp.parameters(), lr = learning_rate, momentum=0.9, weight_decay = 0.001) ## Momentum
    
            train_dataset = Dataset(self.dset[0], self.dset[1])
            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, num_workers = 8)
    
    
    
    
            for train_data_temp, train_labels_temp in train_loader: 
    
        ## Train Loader
                frame_total = frame_total + len(train_data_temp)
                train_labels_temp = train_labels_temp.reshape(len(train_labels_temp))
        ## Init
                self.optimizer.zero_grad()
    
        ## Forward
                x = self.mlp.forward(train_data_temp.float().cuda())
                y = train_labels_temp
    
                cross_entropy = criterion(x, y.cuda())
        ## Backward         
                cross_entropy.backward()
        ## Loss       
                loss = cross_entropy.sum()
                total_loss += loss.data[0] 
        ## Step 
                self.optimizer.step()
        ## Classification Error Computation        
                train_prediction = x.cpu().detach().argmax(dim=1)
                train_accuracy = (train_prediction.cpu().numpy()==y.cpu().numpy()).mean() 
                total_accuracy += train_accuracy
                i = i+1
                if (i%1000 == 0):
                    print ("epoch: ",k, "frames: ", frame_total, "training_loss: ", total_loss/frame_total, "training_accuracy: ", total_accuracy/frame_total, "lr: ", learning_rate)
            training_loss[k] = total_loss/(frame_total)
            training_accuracy[k] = total_accuracy/(frame_total)
            print ("epoch: ",k, "frames: ", frame_total, "training_loss: ", training_loss[k], "training_accuracy: ", training_accuracy[k], "lr: ", learning_rate)
    
        self.metrics = []    
        self.metrics.append(training_loss)
        self.metrics.append(training_accuracy)


def main():
    ## Get Data
    print ("Getting Data")
    train_data = np.load("./dataset/train_feats.npy", encoding='bytes')
    train_labels = np.load("./dataset/train_labels.npy", encoding='bytes')
    test_data = np.load("./dataset/test_feats.npy", encoding='bytes')

    ## Preprocess
    print ("Preprocessing")
    image_size = 32
    train_data, test_data = preprocess.cifar_10_preprocess(train_data,test_data)
    dset = (train_data, train_labels)

    ## CNN
    print ("Declaring CNN")
    mlp = cnn.all_cnn_module()

    ## Train
    print ("Training")
    trainer = get_training_stats(mlp, dset)
    trainer.run(12)
    
    ## Predict
    f = open("predictions.txt", "w")
    for i in range(0,len(test_data)):
        x = mlp.forward(torch.cuda.FloatTensor([test_data[i]]))
        prediction = np.argmax(((x).data).cpu().numpy(), axis = 1)
        f.write("{}\n".format(prediction[0]))
    f.close()
    training_plot(trainer.metrics)

if __name__ == '__main__':
    main()
