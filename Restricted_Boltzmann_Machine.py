# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:02:55 2020

@author: Nooreldean Koteb
"""

#Boltzmann Machines
#Data preprocessing

#Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing movies dataset
movies = pd.read_csv('ml-1m/movies.dat',   #Dataset
                     sep = '::',           #Char to seperate values by
                     header = None,        #No headers (headers in excel files)
                     engine = 'python',
                     encoding = 'latin-1') #Help encode special charachters in dataset

#Importing users dataset
users = pd.read_csv('ml-1m/users.dat',   #Dataset
                     sep = '::',           #Char to seperate values by
                     header = None,        #No headers (headers in excel files)
                     engine = 'python',
                     encoding = 'latin-1') #Help encode special charachters in dataset

#Importing ratings dataset
ratings = pd.read_csv('ml-1m/ratings.dat',   #Dataset
                     sep = '::',           #Char to seperate values by
                     header = None,        #No headers (headers in excel files)
                     engine = 'python',
                     encoding = 'latin-1') #Help encode special charachters in dataset


#Creating the training and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


#Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

#Converting the data into an array with users in lines and movies in columns & rating as cells
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        #Gets a list of all movies & ratings, if the coresponding user id is the same
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        
        #Making no ratings to 0 for each each movie
        ratings = np.zeros(nb_movies)
        #Links movie indicies to ratings
        ratings[id_movies - 1] = id_ratings
        
        #Append movie ratings for every user
        new_data.append(list(ratings))
    
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Converting the ratings into binary ratings 1(Liked) or 0 (Not Liked)

#Get all values in training set equal to 0 and set them to -1 (Not rated)
training_set[training_set == 0] = -1

#Get all 1-2 ratings and set them to 0 (Not Liked)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0

#Get all 3-5 ratings and set them to 1 (Liked)
training_set[training_set >= 3] = 1

#Get all values in test set equal to 0 and set them to -1 (Not rated)
test_set[test_set == 0] = -1

#Get all 1-2 ratings and set them to 0 (Not Liked)
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0

#Get all 3-5 ratings and set them to 1 (Liked)
test_set[test_set >= 3] = 1


#Creating the architecture of the Neural Network
class RBM():
    
    #Initialization function
    #nv = Number of Visible Nodes
    #nh = Number of Hidden Nodes
    def __init__(self, nv, nh):
        #Weights
        self.w = torch.randn(nh, nv)
        #Bias for hidden nodes
        self.a = torch.randn(1, nh)
        #Bias for visible nodes
        self.b = torch.randn(1, nv)
    
    #Samples of hidden nodes
    #Activates hidden nodes based on a given probability
    #x = visible nurons
    def sample_h(self, x):
        #Product of 2 torch tensors
        #(transpose of W)
        wx = torch.mm(x, self.w.t())
        
        #Activation function
        #expand_as() makes sure bias is applied to every line 
        activation = wx + self.a.expand_as(wx)
        
        #Probability hidden node will be activated given visible node
        p_h_given_v = torch.sigmoid(activation)

        #Will return vector with nodes less than 0.7 activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    #Samples of visible nodes
    #Activates visible nodes based on a given probability
    #x = hidden nurons
    def sample_v(self, y):
        #Product of 2 torch tensors
        wy = torch.mm(y, self.w)
        
        #Activation function
        #expand_as() makes sure bias is applied to every line 
        activation = wy + self.b.expand_as(wy)
        
        #Probability hidden node will be activated given visible node
        p_v_given_h = torch.sigmoid(activation)

        #Will return vector with nodes less than 0.7 activated
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    #Training
    #Contrastive Divergence (Log-likelihood gradient using MCMC)
    #K-step contrastive divergence
    #v0 = input vector
    #vk = visible nodes obtained after k iterations
    #ph0 = vector of probabilities at first iteration of 
    #hidden nodes = 1 given v 0 values 
    #phk = probabilties of the hidden nodes after k samplings
    #given vk
    def train(self, v0, vk, ph0, phk):
        self.w += torch.mm(ph0, v0) - torch.mm(phk, vk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
    #Can add more parameters to this class if I want to, stuff like learning rate

#Visible nodes
nv = len(training_set[0])
#Hidden nodes # of features we are trying to detect, can be tuned
nh = 100
#Batch size, can be tuned
batch_size = 100

#Creating object of RBM class we made
rbm = RBM(nv, nh)


#Training the RBM
#Number of epochs, can be tuned
nb_epochs = 10

for epoch in range(1, nb_epochs + 1):
    #Loss variable
    train_loss = 0
    #Float counter to normalize the train loss
    s = 0.
    
    #Batch of users
    for id_user in range(0, nb_users - batch_size, batch_size):
        #Input batch of observations
        vk = training_set[id_user:id_user+batch_size]
        #Target used to calculate loss
        v0 = training_set[id_user:id_user+batch_size]
        #Initial probability (,_ to get only first return of function)
        ph0,_ = rbm.sample_h(v0)
        
        #K-steps
        for k in range(10):
            #Gip sampeling
            #(_, to get only second return of function)
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            
            #undo updated visible nodes with no rating (-1)
            vk[v0<0] = v0[v0<0]
            
        #calculating phk
        phk,_ = rbm.sample_h(vk)
        
        #Train function
        rbm.train(v0, vk, ph0, phk)
        
        #Update train loss
        #But only for rated movies (>0)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        #Updated counter
        s += 1.
    #Print training epoc and normalized train loss
    print('epoch: '+str(epoch)+'/'+str(nb_epochs)+' train loss: '+str(train_loss/s))
 
           
#Predicting on the test set
#Loss variable
test_loss = 0
#Float counter to normalize the train loss
s = 0.

#Looping through users
for id_user in range(nb_users):
    #Input to activate nurons
    v = training_set[id_user:id_user+1]
    #Target used to calculate loss
    vt = test_set[id_user:id_user+1]
    
    #If atleast 1 movie is rated
    if len(vt[vt>=0]) > 0:
        #Gip sampeling
        #(_, to get only second return of function)
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        
        #Update train loss
        #But only for rated movies (>0)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        #Updated counter
        s += 1.
#Print training epoc and normalized train loss
print('test loss: '+str(test_loss/s))




# #Evaluating
# #Two ways to evaluate an RBM is with RMSE and Average Distance

# #RMSE (Root Mean Squared Error) 
# #Train Phase
# nb_epoch = 10
# for epoch in range(1, nb_epoch + 1):
#     train_loss = 0
#     s = 0.
#     for id_user in range(0, nb_users - batch_size, batch_size):
#         vk = training_set[id_user:id_user+batch_size]
#         v0 = training_set[id_user:id_user+batch_size]
#         ph0,_ = rbm.sample_h(v0)
#         for k in range(10):
#             _,hk = rbm.sample_h(vk)
#             _,vk = rbm.sample_v(hk)
#             vk[v0<0] = v0[v0<0]
#         phk,_ = rbm.sample_h(vk)
#         rbm.train(v0, vk, ph0, phk)
#         train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
#         s += 1.
#     print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
#
# #Test Phase
# test_loss = 0
# s = 0.
# for id_user in range(nb_users):
#     v = training_set[id_user:id_user+1]
#     vt = test_set[id_user:id_user+1]
#     if len(vt[vt>=0]) > 0:
#         _,h = rbm.sample_h(v)
#         _,v = rbm.sample_v(h)
#         test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
#         s += 1.
# print('test loss: '+str(test_loss/s))
#
# #We should get 0.46 which corresponds to 75% successful prediction

# #Average Distance
# #This is what we used in our Model Above
# #Train Phase
# nb_epoch = 10
# for epoch in range(1, nb_epoch + 1):
#     train_loss = 0
#     s = 0.
#     for id_user in range(0, nb_users - batch_size, batch_size):
#         vk = training_set[id_user:id_user+batch_size]
#         v0 = training_set[id_user:id_user+batch_size]
#         ph0,_ = rbm.sample_h(v0)
#         for k in range(10):
#             _,hk = rbm.sample_h(vk)
#             _,vk = rbm.sample_v(hk)
#             vk[v0<0] = v0[v0<0]
#         phk,_ = rbm.sample_h(vk)
#         rbm.train(v0, vk, ph0, phk)
#         train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # Average Distance here
#         s += 1.
#     print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
# #Test Phase
# test_loss = 0
# s = 0.
# for id_user in range(nb_users):
#     v = training_set[id_user:id_user+1]
#     vt = test_set[id_user:id_user+1]
#     if len(vt[vt>=0]) > 0:
#         _,h = rbm.sample_h(v)
#         _,v = rbm.sample_v(h)
#         test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Average Distance here
#         s += 1.
# print('test loss: '+str(test_loss/s))
#
# #We should get a 0.24 which is equivilant to about 75% successful prediction
#
# #Code to check that 0.25 corresponds to 75% success
# import numpy as np
# u = np.random.choice([0,1], 100000)
# v = np.random.choice([0,1], 100000)
# u[:50000] = v[:50000]
# sum(u==v)/float(len(u)) # -> you get 0.75
# np.mean(np.abs(u-v)) # -> you get 0.25

