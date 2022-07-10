import numpy as np
import utils

##########################
### MODEL 2
##########################

class NeuralNetMLP_2:

    def __init__(self, num_features, num_hidden, num_hidden_2, num_classes, random_seed=123, minibatch_size=100):
        super().__init__()
        rng = np.random.RandomState(random_seed)
        
        self.num_classes = num_classes
        self.minibatch_size = minibatch_size
        
        # hidden [50 x 784]
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # hidden 2 [50 x 50]
        self.weight_h_2 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden_2, num_hidden))
        self.bias_h_2 = np.zeros(num_hidden_2)
        
        # output [10 x 50]
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden_2))
        self.bias_out = np.zeros(num_classes)

        
        
    def forward(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h        
        a_h = utils.sigmoid(z_h)
        # Hidden layer 2
        # input dim: [n_examples, n_hidden] dot [n_hidden_2, n_hidden].T
        # output dim: [n_examples, n_hidden_2]
        z_h_2 = np.dot(a_h, self.weight_h_2.T) + self.bias_h_2        
        a_h_2 = utils.sigmoid(z_h_2)

        # Output layer
        # input dim: [n_examples, n_hidden_2] dot [n_classes, n_hidden_2].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h_2, self.weight_out.T) + self.bias_out
        a_out = utils.sigmoid(z_out)
        
        return a_h, a_h_2, a_out

    def backward(self, x, a_h, a_h_2, a_out, y):  
    
        #########################
        ### Output layer weights
        #########################
        
        # onehot encoding
        y_onehot = utils.int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        
        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out # "delta (rule) placeholder"

        # gradient for output weights
        
        # [n_examples, n_hidden_2]
        d_z_out__dw_out = a_h_2
        
        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden_2]
        # output dim: [n_classes, n_hidden_2]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)      
        
        #################################        
        # Part 2: dLoss/dHiddenWeights_2 = DeltaOut * dOutNet/dHiddenAct_2 * dHiddenAct_2/dHiddenNet_2 * dHiddenNet_2/dWeight_2
        # where DeltaHidden2 = dOutNet/dHiddenAct_2 * dHiddenAct_2/dHiddenNet_2
        
        # [n_classes, n_hidden_2]
        d_z_out__a_h_2 = self.weight_out
        
        # output dim: [n_examples, n_hidden_2]
        d_loss__a_h_2 = np.dot(delta_out, d_z_out__a_h_2)
        
        # [n_examples, n_hidden_2]
        d_a_h_2__d_z_h_2 = a_h_2 * (1. - a_h_2) # sigmoid derivative
        
        delta_hidden_2 = d_loss__a_h_2 * d_a_h_2__d_z_h_2
        
        # [n_examples, n_features]
        d_z_h_2__d_w_h_2 = a_h
        # output dim: [n_hidden_2, n_features]
        d_loss__d_w_h_2 = np.dot((delta_hidden_2).T, d_z_h_2__d_w_h_2)
        d_loss__d_b_h_2 = np.sum((delta_hidden_2), axis=0)

        #################################        
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * DeltaHidden2 * dHiddenNet_2/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        
        # [n_classes, n_hidden]
        d_z_h_2__a_h = self.weight_h_2
        
        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_hidden_2, d_z_h_2__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative
        
        # [n_examples, n_features]
        d_z_h__d_w_h = x
        
        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)        

        return (d_loss__dw_out,  d_loss__db_out, 
                d_loss__d_w_h_2, d_loss__d_b_h_2,
                d_loss__d_w_h,   d_loss__d_b_h               
               )
    
    def train(self, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
        early_stopping = utils.EarlyStopping(verbose=False)

        epoch_train_loss = []
        epoch_valid_loss = []
        epoch_train_acc = []
        epoch_valid_acc = []

        for e in range(num_epochs):

            # iterate over minibatches
            minibatch_gen = utils.minibatch_generator(X_train, y_train, self.minibatch_size)

            for X_train_mini, y_train_mini in minibatch_gen:

                #### Compute outputs ####
                a_h, a_h_2, a_out = self.forward(X_train_mini)

                #### Compute gradients ####
                d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h_2, d_loss__d_b_h_2, d_loss__d_w_h, d_loss__d_b_h  = \
                    self.backward(X_train_mini, a_h, a_h_2, a_out, y_train_mini)

                #### Update weights ####            
                self.weight_h -= learning_rate * d_loss__d_w_h
                self.bias_h -= learning_rate * d_loss__d_b_h
                self.weight_h_2 -= learning_rate * d_loss__d_w_h_2
                self.bias_h_2 -= learning_rate * d_loss__d_b_h_2
                self.weight_out -= learning_rate * d_loss__d_w_out
                self.bias_out -= learning_rate * d_loss__d_b_out


            #### Epoch Logging ####        
            train_mse, train_acc, _, _ = utils.compute_mse_and_acc(self, X_train, y_train)
            valid_mse, valid_acc, _, _ = utils.compute_mse_and_acc(self, X_valid, y_valid)
            train_acc, valid_acc = train_acc*100, valid_acc*100
            epoch_train_acc.append(train_acc)
            epoch_valid_acc.append(valid_acc)
            epoch_train_loss.append(train_mse)
            epoch_valid_loss.append(valid_mse)
            print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
                  f'| Train MSE: {train_mse:.2f} '
                  f'| Train Acc: {train_acc:.2f}% '
                  f'| Valid Acc: {valid_acc:.2f}%')

            early_stopping(valid_acc)
            if early_stopping.early_stop:
                break

        return epoch_train_loss, epoch_valid_loss, epoch_train_acc, epoch_valid_acc