import torch as tr
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.model_selection import train_test_split
import numpy as np

class NN:
    def __init__(self, 
                model,
                **kwargs,
                ):

        #  key compoenents
        self.model = model

        #  hyper-parameters
        self.epoch = kwargs["epoch"]
        self.device = kwargs["device"]
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]

        # init opt and loss 
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.BCELoss().to(self.device)
 
    def run_epoch(self,
                X_train,  # X_train: [N_samples,input_dim];  -> Tensor
                y_train,  # y_train: [N_samples,];  -> Tensor
                X_test,   #  X_test: [N_samples,input_dim];  -> Tensor
                y_test,   #  y_test: [N_samples,];  -> Tensor
                verbo=True, 
                ):
        best_valid_loss = float('inf')

        for epoch in range(self.epoch):

            train_loss = self.train(X_train, y_train)
            valid_loss = self.test(X_test, y_test)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #  self.best_model = copy.deepcopy(self.model).cpu()
                if verbo:
                    print(f'Epoch: {epoch+1}:')
                    print(f'Train Loss: {train_loss:.3f}')
                    print(f'Validation Loss: {valid_loss:.3f}')
        return best_valid_loss
        

    def train(self, 
            X_train,  #  X_train: [N_samples,input_dim];  -> Tensor
            y_train,  #  y_train: [N_samples,];  -> Tensor
            ): 
        '''
        Returns:
            local_batch:  [batch_size, input_dim] -> Tensor
            local_labels: [batch_size,];  -> Tensor
        '''
        self.model.train()
        epoch_loss = 0

        for local_batch, local_labels in self.batcher(X_train, y_train, self.batch_size):

            local_batch, local_labels = local_batch.to(self.device), \
                                        local_labels.flatten().to(self.device)

            # print('input are:'); print(local_batch.size()); print(local_labels.size())
            self.opt.zero_grad()
            local_output = self.model(local_batch) #  [batch_size, 1, height, width]
            # print('output are:'); print(local_output.size()); print(local_labels.size())

            local_output = local_output.view(self.batch_size, -1)
            local_labels = local_labels.view(self.batch_size, -1)

            loss = self.loss(local_output, local_labels)
            loss.backward(); self.opt.step()
            epoch_loss += loss.item()
        return epoch_loss

    def test(self, 
            X_test,  #  X_test: [N_samples,input_dim];  -> Tensor
            y_test,  #  y_test: [N_samples,];  -> Tensor
            ): 
        '''
        Returns:
            local_batch:  [batch_size, input_dim] -> Tensor
            local_labels: [batch_size,];  -> Tensor
        '''
        self.model.eval()
        epoch_loss = 0

        for local_batch, local_labels in self.batcher(X_test, y_test, self.batch_size):

            local_batch, local_labels = local_batch.to(self.device), \
                                        local_labels.flatten().to(self.device)

            # print('input are:');  print(local_batch.size());  print(local_labels.size())

            local_output = self.model(local_batch)

            #  print('output are:'); print(local_output.size()); print(local_labels.size())

            loss = self.loss(local_output, local_labels)
            epoch_loss += loss.item()
        return epoch_loss


    def prediction(self, x):
        self.model.eval()
        return self.model(x)

    def batcher(self, x, y, batch_size):
        l = len(y)
        for batch in range(0, l, batch_size):
            yield (x[batch:min(batch + batch_size, l)], y[batch:min(batch + batch_size, l)])



class helper:

    @staticmethod
    def load_np(path):
        return np.load(path)

    @staticmethod
    def format_input(data,  #  [num_sample, 2] -> np.ndarray
                    ):

        n = len(data); num_element = data[0][0][0] * data[0][0][1]

        imgs, labels = tr.zeros((n, num_element)), tr.zeros((num_element)) 

        for i, d in enumerate(data):

            img = tr.from_numpy(d[0].flatten()).float(); label = tr.from_numpy(d[1].flatten()).float()

            imgs[i] = img; labels[i] = label
        

        X_train, X_test, y_train, y_test = train_test_split(imgs, labels, 
                                            test_size=0.33, random_state=None)
        return X_train, y_train, X_test, y_test