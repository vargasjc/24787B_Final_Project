###########################################################
#
# These functions are the Neural Network
#
###########################################################
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from sklearn.metrics import confusion_matrix


def concatenate_all_EEG_data(data):
    """This takes all the EEG channels from a file and concatenates it togethers"""
    new_data = pd.DataFrame()
    for col in [l for l in data.columns.levels[0] if "EEG" in l]:
        temp_df = data[[col,"Event"]]
        temp_df.columns = temp_df.columns.droplevel(0)
        new_data = pd.concat([new_data,temp_df], axis=0)
    return new_data.drop(labels="gamma", axis=1)

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.features[index]
        y = self.labels[index]

        return X, y

# 
class StratifiedSampler(torch.utils.data.Sampler): #Sampler
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    
    Note: This function was not written by me.
    Courtesy of:
        https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(len(class_vector) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(len(self.class_vector),2).numpy()
        y = np.array(self.class_vector)
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)
    

def get_training_data(data, labels, batch_size=500, num_workers=2):
    # Convert labels vector to numbers and save category encoding
    try:
        labels, categories = pd.factorize(labels["value"], sort=True)
    except:
        labels, categories = pd.factorize(labels, sort=True)
    features, labels = np.array(data), np.array(labels)
    
    # We need to load a baanced amount of labels on each training batch. So we sample evenly.
    #class_sample_count = np.bincount(labels)

    #weights = 1 / torch.Tensor(class_sample_count)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, int(batch_size/6))
    sampler = StratifiedSampler(class_vector=labels, batch_size=batch_size)
    
    return torch.utils.data.DataLoader(Dataset(features.astype(float),labels.astype(int)), 
                                       batch_size=batch_size, 
                                       shuffle=False, 
                                       num_workers=num_workers, sampler=sampler), categories
                                       

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def filter_single_events(predict):
    """This simple function eliminates predictions that only occurred for one second."""
    for i in range(len(predict)):
        if (i >= 2) and (i < len(predict)-1):
            if (predict[i-1] == predict[i-2]) and (predict[i-1] == predict[i+1]):
                #print("Changing label from",str(predict[i]),"to",str(predict[i-1]))
                if predict[i-1] == 0:
                    predict[i] = predict[i-1]
    return predict

def train(data, categories, model, loss_function, optimizer, allow_cuda = True, print_out = True, x_shape = (-1,6)):
    # Gotta tell the model to be in training mode. Test mode disables certain things.
    model.train()
    
    accuracy = 0
    count = 0
    loss_av = 0
    
    classes = {i for i in range(len(categories))}
    class_acc = np.zeros(len(classes)) #+1
    totals_acc = np.zeros(len(classes)) #+1
            
    for idx, (features, labels) in enumerate(data):
        # Gotta make our input variables into tensors of the right size
        x = to_var(features.view(x_shape).float())
        y = to_var(labels.view(-1,1).long())            
        
        # Verifying Data mix
        if print_out:
            print("Data label mix",idx,np.bincount(labels.numpy().squeeze()))
        
        # Run the model. Not necessary to initialize filters because Torch initializes with appropriate algorithm by default
        output = model(x) 
        
        # The loss function. Cross entropy is something like y*log(y_hat)....
        loss = loss_function(output, y.squeeze())
    
        # Cleanup for next iter
        loss_function.zero_grad()
    
        # Back propagation
        loss.backward()
    
        # Calculate the new filters
        optimizer.step()
    
        # We do predictions to calculate the accuracy of this iter
        prediction = torch.max(nn.functional.softmax(output, dim=1), 1)[1]
        prediction = prediction.cpu().data.numpy().squeeze()
        actual = y.cpu().data.numpy().squeeze()
        
        # Implement isolated transition rule (An event is at least 2 seconds)
        #prediction = filter_single_events(prediction)
        
        # Calculate accuracy
        accuracy += np.sum(prediction == actual)
        count += len(y)
        loss_av += loss.cpu().data.numpy()
        
        for i in classes: #set(data.dataset.labels.reshape(-1,1).squeeze()): #range(17):
            filter_i = actual == i
            class_acc[i] += np.sum(prediction[filter_i] == actual[filter_i])
            totals_acc[i] += len(actual[filter_i])
    
    accuracy = accuracy/count
    loss_av = loss_av/idx
    
    if print_out:
        class_acc_final = class_acc/totals_acc
        
        print("\tAccuracy: %.2f%%"%(100*accuracy),"\tLoss: %.2f"%loss_av)
        for v in classes:
            print("\tClass",categories[v],"Accuracy: %.2f%%"%(100*class_acc_final[v]))
    
    return accuracy, loss_av, class_acc, totals_acc


def test(data, categories, model, loss_function, optimizer, allow_cuda = True, x_shape = (-1,6)):
    # Gotta tell the model to be in test mode.
    model.eval()
    
    accuracy = 0
    count = 0
    loss_av = 0
    
    classes = {i for i in range(len(categories))}
    class_acc = np.zeros(len(classes))
    totals_acc = np.zeros(len(classes))
            
    #for idx, (features, labels) in enumerate(data):
    #    # Gotta make our input variables into tensors of the right size
    #    x = to_var(features.view(x_shape).float())
    #    y = to_var(labels.view(-1,1).long())            
    
    # Gotta make our input variables into tensors of the right size
    features = torch.tensor(data.dataset.features)
    labels = torch.tensor(data.dataset.labels)
    
    x = to_var(features.view(x_shape).float())
    y = to_var(labels.view(-1,1).long()) 
    
    # Verifying Data mix
    print("Data label mix",np.bincount(labels.numpy().squeeze()))
    
    # Run the model. Not necessary to initialize filters because Torch initializes with appropriate algorithm by default
    output = model(x) 
    
    # The loss function. Cross entropy is something like y*log(y_hat)....
    loss = loss_function(output, y.squeeze())

    # We do predictions to calculate the accuracy of this iter
    prediction = torch.max(nn.functional.softmax(output, dim=1), 1)[1]
    prediction = prediction.cpu().data.numpy().squeeze()
    actual = y.cpu().data.numpy().squeeze()
    
    # Implement isolated transition rule (An event is at least 2 seconds)
    #prediction = filter_single_events(prediction)
    
    # Calculate accuracy
    accuracy += np.sum(prediction == actual)
    count += len(y)
    loss_av += loss.cpu().data.numpy()
    
    for i in classes: #set(data.dataset.labels.reshape(-1,1).squeeze()): #range(17):
        filter_i = actual == i
        class_acc[i] += np.sum(prediction[filter_i] == actual[filter_i])
        totals_acc[i] += len(actual[filter_i])
    
    accuracy = accuracy/count
    #loss_av = loss_av/idx
    
    class_acc_final = class_acc/totals_acc
    
    print("\tAccuracy: %.2f%%"%(100*accuracy),"\tLoss: %.2f"%loss_av)
    for v in classes:
        print("\tClass",categories[v],"Accuracy: %.2f%%"%(100*class_acc_final[v]))
    
    return accuracy, loss_av, class_acc, totals_acc


def predict(data, model, allow_cuda = True, x_shape = (-1,6)):
    # Gotta tell the model to be in test mode.
    model.eval()
            
    # Gotta make our input variables into tensors of the right size
    features = torch.tensor(data.dataset.features)
    labels = torch.tensor(data.dataset.labels)
    
    x = to_var(features.view(x_shape).float())
    y = to_var(labels.view(-1,1).long())            
    
    # Run the model. Not necessary to initialize filters because Torch initializes with appropriate algorithm by default
    output = model(x) 

    # We do predictions to calculate the accuracy of this iter
    prediction = torch.max(nn.functional.softmax(output, dim=1), 1)[1]
    prediction = prediction.cpu().data.numpy().squeeze()
    actual = y.cpu().data.numpy().squeeze()
    
    # Implement isolated transition rule (An event is at least 2 seconds)
    #prediction = filter_single_events(prediction)
    
    cm = confusion_matrix(actual, prediction)
    row_sum = cm.sum(axis = 1, keepdims=True)
    #plt.title("Confusion Matrix")
    plt.matshow(np.round(100*cm/row_sum,1), cmap=plt.cm.YlGn) #gray
    plt.colorbar()
    plt.show()
    
    print("Confusion Matrix:",cm,sep="\n")
    if len(cm)==2:
        print("Precision: %.2f%%"%(100*cm[1,1]/np.sum(cm[:,1])))
        print("Recall: %.2f%%"%(100*cm[1,1]/np.sum(cm[1,:])))
    
    return prediction, actual


def run_model(data, categories, model, learning_rate = 0.0005, epochs = 10, optimizer = [], allow_cuda = True, x_shape = (-1,1,200), train_mode=True):
    start_time = time.time()
    
    # We have an unbalanced dataset, therefore we apply weights to the loss function to compensate
    bins = np.bincount(data.dataset.labels)
    if torch.cuda.is_available():
        weights = torch.tensor(1/bins).float().cuda() #bins.mean()
    else:
        weights = torch.tensor(1/bins).float()
    loss_function = nn.CrossEntropyLoss(weight=weights)
    
    # Typicial Adam Optimizer
    if optimizer == []:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, eps=1e-08)
    
    accuracy = []
    loss = []
    class_acc = []
    class_acc_tot = []
    
    mode_label = ""
    
    # If Cuda...
    if torch.cuda.is_available() and allow_cuda:
            model.cuda()
            print('Model moved to GPU.')
    
    if train_mode:
        mode_label = "Training"
        for epoch in range(epochs):
            if (epoch % 20 == 0) or (epoch == epochs-1):
                print("Training epoch %3i"%(epoch + 1))
                print_out = True
            else:
                print_out = False
            a, l, ca, ta = train(data, categories, model, loss_function, optimizer, allow_cuda = allow_cuda, print_out = print_out, x_shape = x_shape)
            accuracy.append(a)
            loss.append(l)
            class_acc.append(ca)
            class_acc_tot.append(ta)
            
        class_accuracy = np.sum(class_acc,0)/np.sum(class_acc_tot,0)
        final_accuracy = np.array(class_acc)[-1:,:]/np.array(class_acc_tot)[-1:,:]
        
        try:
            plt.title("Accuracy vs. epoch")
            plt.plot(accuracy,label="Accuracy")
            plt.plot([0,epochs],[np.mean(accuracy),np.mean(accuracy)],"r--")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.grid()
            plt.show()
            plt.title("Loss vs. epoch")
            plt.plot(loss,label="Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.grid()
            plt.show()
            plt.title(mode_label + " - Class Accuracy")
            plt.plot(np.arange(0,len(categories),1),final_accuracy.squeeze(),label="Final accuracy")
            plt.plot(np.arange(0,len(categories),1),class_accuracy,"r--",label="Average accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Class")
            plt.xticks(range(len(categories)), categories, size="small")
            plt.legend()
            plt.grid()
            plt.show()
        except:
            print("Error plotting graphs")
        try:
            prediction, actual = predict(data, model, allow_cuda = True, x_shape = x_shape)
            cm = confusion_matrix(actual, prediction)
        except:
            print("Error using model to predict data!")
            cm = None
    else:
        mode_label = "Test"
        a, l, ca, ta = test(data, categories, model, loss_function, optimizer, allow_cuda = allow_cuda, x_shape = x_shape)
        accuracy.append(a)
        loss.append(l)
        class_acc.append(ca)
        class_acc_tot.append(ta)
        
        class_accuracy = np.sum(class_acc,0)/np.sum(class_acc_tot,0)
        final_accuracy = np.array(class_acc)[-1:,:]/np.array(class_acc_tot)[-1:,:]
        
        try:
            plt.title(mode_label + " - Class Accuracy")
            plt.plot(np.arange(0,len(categories),1),final_accuracy.squeeze(),label="Final accuracy")
            plt.plot(np.arange(0,len(categories),1),class_accuracy,"r--",label="Average accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Class")
            plt.xticks(range(len(categories)), categories, size="small")
            plt.legend()
            plt.grid()
            plt.show()
        except:
            print("Error plotting graphs")
        try:
            prediction, actual = predict(data, model, allow_cuda = True, x_shape = x_shape)
            cm = confusion_matrix(actual, prediction)
        except:
            print("Error using model to predict data!")
            cm = None

    #class_accuracy(data, model, x_shape = x_shape)
    
    print("Running time: %.2fs"%(time.time() - start_time)) 
    
    return accuracy, final_accuracy.squeeze(), class_accuracy, loss, cm
    
###########################################################
#
# These are ancillary functions to save the model parameters
#
###########################################################
    
def save_model(model, optimizer, epochs, categories, train_data, test_data, PATH = ""):
    """Save model to hard drive."""
    torch.save({
            "epochs": epochs,
            "categories": categories,
            "model": model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_data": train_data,
            "test_data": test_data
            }, PATH)

def load_model(filename, PATH = "Models"): #epochs, categories, train_data, test_data,
    file = os.path.join(PATH,filename)
    checkpoint = torch.load(file)
    
    model = checkpoint["model"]
    optimizer = optim.Adam(model.parameters(), lr = 0.001, eps=1e-08)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epochs = checkpoint["epochs"]
    categories = checkpoint["categories"]
    train_data = checkpoint["train_data"]
    test_data = checkpoint["test_data"]
    
    print("State Loaded!")
    try:
        print("\t ->Using",train_data["Train File"],"for Training!") 
        print("\t ->Using",test_data["Test File"],"for Testing!")
    except:
        pass
    
    # Extract Training Data
    try:
        train_accuracy = train_data["Train Accuracy"]
    except:
        train_accuracy = None
    try:
        train_loss = train_data["Train Loss"]
    except:
        train_loss = None
    try:
        train_final_class_accuracy = train_data["Train Final Class Accuracy"]
        train_final_class_accuracy = train_final_class_accuracy[~np.isnan(train_final_class_accuracy)]
    except:
        train_final_class_accuracy = None
    try:
        train_class_accuracy = train_data["Train Av Class Accuracy"]
        train_class_accuracy = train_class_accuracy[~np.isnan(train_class_accuracy)]
    except:
        train_class_accuracy = None
        
    # Extract Testing Data
    try:
        test_accuracy = test_data["Test Accuracy"]
    except:
        test_accuracy = None
    try:
        test_loss = test_data["Test Loss"]
    except:
        test_loss = None
    try:
        test_final_class_accuracy = test_data["Test Final Class Accuracy"]
        test_final_class_accuracy = test_final_class_accuracy[~np.isnan(test_final_class_accuracy)]
    except:
        test_final_class_accuracy = None
    try:
        test_class_accuracy = test_data["Test Av Class Accuracy"]
        test_class_accuracy = test_class_accuracy[~np.isnan(test_class_accuracy)]
    except:
        test_class_accuracy = None
    
    # Printing Results
    print("Final Train Accuracy:",str(train_accuracy[-1:]))
    print("\t\tTrain Final Class Accuracy:",str(train_final_class_accuracy))
    print("\t\tTrain Average Class Accuracy:",str(train_class_accuracy))
    print("\t\tTrain Loss:",str(train_loss[-1:]))
    print("\nFinal Test Accuracy:",str(test_accuracy[-1:]))
    print("\t\tTest Final Class Accuracy:",str(test_final_class_accuracy))
    print("\t\tTest Loss:",str(test_loss[-1:]))
    print("\nModel:",model)
    
    # Training Plots
    try:
        plt.title("Accuracy vs. epoch")
        plt.plot(train_accuracy,label="Accuracy")
        plt.plot([0,epochs],[np.mean(train_accuracy),np.mean(train_accuracy)],"r--")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.grid()
        plt.show()
        plt.title("Loss vs. epoch")
        plt.plot(train_loss,label="Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid()
        plt.show()
        plt.title("Training Class Accuracy")
        plt.plot(np.arange(0,len(categories),1),train_final_class_accuracy,label="Final accuracy")
        plt.plot(np.arange(0,len(categories),1),train_class_accuracy,"r--",label="Average accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Class")
        plt.xticks(range(len(categories)), categories, size="small")
        plt.legend()
        plt.grid()
        plt.show()
    except:
        print("Error plotting training data!")
    # Test Plot
    try:
        plt.title("Test - Class Accuracy")
        try:
            plt.plot(np.arange(0,len(categories),1),test_final_class_accuracy,label="Final accuracy")
            plt.plot(np.arange(0,len(categories),1),test_class_accuracy,"r--",label="Average accuracy")
        except:
            plt.plot(test_final_class_accuracy,label="Final accuracy")
            plt.plot(test_class_accuracy,"r--",label="Average accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Class")
        plt.xticks(range(len(categories)), categories, size="small")
        plt.legend()
        plt.grid()
        plt.show()
    except:
        print("Error plotting test data!")
    
    return model, optimizer, epochs, categories, train_data, test_data
    
