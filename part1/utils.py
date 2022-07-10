import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model=None, patience=7, verbose=True, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = -np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.best_model_checkpoint=None
        if model:
            self.model = model
            self.best_model_checkpoint = self.model.state_dict
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            if self.counter == 0:
                if self.best_model_checkpoint:
                    self.best_model_checkpoint = self.model.state_dict
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.best_model_checkpoint:
                self.best_model_checkpoint = self.model.state_dict
            self.val_loss_max = val_loss
            self.counter = 0
        if self.verbose:            
            print(f'self.counter {self.counter} best_score {self.best_score} early_stop {self.early_stop}')

            
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100, return_proba=False):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    preds = []
    targets_list = []
    for i, (features, targets) in enumerate(minibatch_gen):
        
        try:
            _, probas = nnet.forward(features)
        except:
            _, _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        if return_proba:
            preds.extend(list(probas))
        else:
            preds.extend(list(predicted_labels))
        targets_list.extend(list(targets))
        
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        
        num_examples += targets.shape[0]
        mse += loss

    mse = mse/i
    acc = correct_pred/num_examples
    return mse, acc, preds, targets_list


def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets) 


def sigmoid(z):                                        
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary


def minibatch_generator(X, y, minibatch_size=100):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size 
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]