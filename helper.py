import torch as tr
import torch.nn as nn


"""
helper class for training
"""

class helper:

    @staticmethod
    def batcher(image):
        pass



class DNN:
    def __init__(self, 
                model,
                optimiser, 
                loss_func, 
                **kwargs,
                ):

        #  key compoenents
        self.model = model
        self.opt = optimiser
        self.loss = loss_func

        #  hyper-parameters
        self.epoch = kwargs["epoch"]
        self.device = kwargs["device"]
        self.batch_size = kwargs["batch_size"]

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

            # print('input are:')
            # print(local_batch.size())
            # print(local_labels.size())

            self.opt.zero_grad()
            local_output = self.model(local_batch)

            # print('output are:')
            # print(local_output.size())
            # print(local_labels.size())

            loss = self.loss(local_output, local_labels)
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        return epoch_loss

    def batcher(self, x, y, batch_size):
        l = len(y)
        for batch in range(0, l, batch_size):
            yield (x[batch:min(batch + batch_size, l)], y[batch:min(batch + batch_size, l)])
    
    def test(self, data):
        return 


        