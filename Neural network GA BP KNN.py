#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as Sc
import numpy as np
from matplotlib import pyplot as plt
import math
import time
from copy import deepcopy
import random
from sklearn.metrics import mean_squared_error 
import copy
import sys
import pprint
from sklearn import metrics
import operator
from collections import defaultdict
from matplotlib import pyplot as plt


# In[2]:


#generating random bias for the network
def generate_bias(NN):
    l=[]
    for i in NN[1:]:
        l.append(np.random.randn(i,1))
    return l

#generating random weights for the network
def generate_weights(NN):
    l=[]
    for i,j in zip(NN[:-1],NN[1:]):
        temp = np.random.uniform(-0.5, 0.5, i*j)
        temp = np.reshape(temp,(j,i))
        l.append(temp)
    return l



#This class represents each network 
class population:
    def __init__(self,NN):
        self.weights = generate_weights(NN)
        self.bias = generate_bias(NN)
        self.no_layers = len(NN)

#backpropogation ...
class backpropogation:
    def __init__(self,layers, x, y):
        self.x = x
        self.y = y
        self.layers = layers
        self.n = len(layers)
        self.learning_rate = 0.05
        self.network = population(layers)
        self.keep={}
        
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))
    
    def sigmoid_derivative(self, X):
        return X*(1-X)
    
    def feed_forward(self, in_1):
        k=1
        for i,j in zip(self.network.bias, self.network.weights):
            net_value = np.dot(j, in_1) + i
            for t in range((self.layers[k])):
                self.keep['net'+str(k+1)+str(t+1)] = net_value[t][0]
                in_1 = self.keep['out'+str(k+1)+str(t+1)] = self.sigmoid(net_value[t][0])
            k+=1
        return in_1
    
    def output_backtrack(self, out_value):
        temp = self.layers[-1]
        for i in range(temp):
            a = self.keep['d_E'+str(self.n)+str(i+1) +'/out'+str(self.n)+str(i+1)] = self.keep['out'+str(self.n)+str(i+1)] - self.y[i]
            tt = self.keep['out'+str(self.n)+str(i+1)]
            b = self.keep['d_out'+str(self.n)+str(i+1)+"/"+'net'+str(self.n)+str(i+1)] = tt*(1-tt)
            for k in range(self.layers[-2]):
                extra = self.network.weights[self.n - 2][i][k]
                extra = extra - self.learning_rate*(a*b)*self.keep['out'+str(self.n-1)+str(k+1)]
                self.network.weights[self.n -2][i][k] = extra
           
                
    def hidden_backtrack(self, out_value, layer_no):
        #pprint.pprint(self.keep)
        for i in range(self.layers[layer_no-1]):
            temp = self.keep['out'+str(layer_no)+str(i+1)]
            a = self.keep['d_out'+str(layer_no)+str(i+1)+'/net'+str(layer_no)+str(i+1)] = self.sigmoid_derivative(temp)
            
            s = 0
            for k in range((self.layers[layer_no])):
                prod_x = self.keep['d_E'+str(layer_no+1)+str(k+1)+'/out'+str(layer_no+1)+str(k+1)]
                prod_y = self.keep['d_out'+str(layer_no+1)+str(k+1)+'/net'+str(layer_no+1)+str(k+1)]
                
                add_t = self.network.weights[layer_no-1][k][i]
                
                prd=(prod_x*prod_y)
                prd_add = prd+add_t
                
                self.keep['d_E'+str(layer_no)+str(k+1)+'/net'+str(layer_no)+str(k+1)] = prd
                self.keep['d_E'+str(layer_no)+str(k+1)+'/out'+str(layer_no)+str(i+1)] = prd_add
                
                s+=prd_add
            self.keep['d_E'+str(layer_no)+str(i+1)+'/out'+str(layer_no)+str(i+1)] = s    
            
            for j in range(self.layers[layer_no-2]):
                lol = s * a * self.network.weights[layer_no - 2][i][j]
                self.keep['d_E'+str(layer_no)+str(i+1)+'/w'+str(layer_no-1)+str(j+1)] = lol
                
                #weight update
                W_init = self.network.weights[layer_no-2][i][j]
#                 print("initial weight =",W_init)
                W_init = W_init - self.learning_rate*(lol)
#                 print("final weight = ",W_init)
                self.network.weights[layer_no-2][i][j] = W_init
               
            
    def backtrack(self):
        for i in range(len(self.y)):
            out_value = self.feed_forward(self.x[i])
            self.output_backtrack(out_value)
            
            for i in range(1,self.n-2):
                self.hidden_backtrack(out_value,self.n - i)
    
    def accuracy(self, test_x,test_y):
        l=[]
        for i in range(len(test_y)):
            predicted_value = self.feed_forward(test_x[i])
            l.append(predicted_value)
        error = mean_squared_error(list(test_y), l)
        return error
    
        
    def display(self):
        pprint.pprint(self.keep)
    
    def return_network(self):
        return self.network
            

 #genetic algorithm class.
class Genetics_Algorithm:
    def __init__(self, n, layers, x, y, network):
        self.num_layers = len(layers)
        self.layers = layers
        self.crossover_rate = 0.4
        self.retain_rate = 0.4
        self.mut_rate = 0.2
        self.max_pop = n
        self.all_pop = list()
        self.x = x
        self.y = y
        self.best_count = int(self.max_pop * self.retain_rate)
        self.bad_count = int((self.max_pop - self.best_count) * self.retain_rate)

        #generate n population
        for i in range(n-1):
            self.all_pop.append(population(layers))
        
        #add Backtrack network..
        self.all_pop.append(network)
        
        self.ct_bias = int(sum(self.layers[1:]))
        self.ct_weights = int(sum([self.all_pop[0].weights[i].size for i in range(self.num_layers-2)]))
        
    #calculate score for entire population
    def calculate_score(self):#doubt
        l=[]
        for network in self.all_pop:
            total = 0
            for i in range(len(self.y)):
                predicted_value = self.get_output(network, self.x[i].reshape(-1,1))
                actual_value = self.y[i]
                #temp = mean_squared_error(actual_value, predicted_value)
                temp = np.power(predicted_value - actual_value, 2)/2
                #sum up the least mean square error..
                total+=np.sum(temp)
            l.append(total)
        return l

          
    #returns the predicted value output layer
    def get_output(self, network,in_1):
        t=1
        for i,j in zip(network.bias, network.weights):
            if t == self.num_layers-1:
                in_1 = self.sigmoid(np.dot(j, in_1) + i)
            else:
                in_1 = self.relu(np.dot(j,in_1) + i)
            t+=1
        return in_1

    #activation sigmoid
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    #activation relu
    def relu(self,x):
        return np.maximum(0,x)
    
    #returns the random bias points
    def get_point_bias(self):
        rand_layer = random.choice(np.arange(1,self.num_layers)) 
        #select random point on layer
        rand_point = random.choice(np.arange(self.layers[rand_layer]))
        return rand_layer-1, rand_point

    
    #returns random weight points
    def get_point_weights(self):
        rand_layer = random.choice(np.arange(1,self.num_layers))
        r_point = random.choice(np.arange(self.layers[rand_layer]))
        c_point = random.choice(np.arange(self.layers[rand_layer-1]))
        return rand_layer -1, r_point, c_point


    #cross breed accross father and mother
    def cross_breed(self, father, mother):
        child = copy.deepcopy(father)
        #cross the bias
        for i in range(self.ct_bias):
            lay_n, p_n = self.get_point_bias()
            if random.uniform(0,1) < self.crossover_rate:
                #print("bias",lay_n, p_n)
                child.bias[lay_n][p_n][0] = mother.bias[lay_n][p_n][0]
        
        #cross the weights
        for i in range(self.ct_weights):
            lay_n, row, col = self.get_point_weights()
            if random.uniform(0,1) < self.crossover_rate:
                #print("weight",lay_n,row,col)
                child.weights[lay_n][row][col] = mother.weights[lay_n][row][col]
        return child
        

        
    #returns the mutated child
    def mutate(self, child):
        c = copy.deepcopy(child)
        for i in range(self.ct_bias):
            lay_n, p_n = self.get_point_bias()
            temp = random.uniform(-0.5, 0.5)
            if random.uniform(0,1) < self.mut_rate:
                c.bias[lay_n][p_n] += temp

        for i in range(self.ct_weights):
            lay_n, row, col = self.get_point_weights()
            temp = random.uniform(-0.5,0.5)
            if random.uniform(0,1) < self.mut_rate:
                child.weights[lay_n][row][col] += temp
        return c

    
    #returns the list of predicted value wrt test value
    def accuracy(self, test_x,test_y):
        l=[]
        for i in range(len(test_y)):
            predicted_value = self.get_output(self.all_pop[0], test_x[i].reshape(-1,1))
            l.append(predicted_value[0])
        return l

    #for updating weight to get greater accuracy..
    def update_weights(self):
            
        #calculate score for each population
        error_rate_score = list(zip(self.calculate_score(),self.all_pop))
        #sort based on error
        error_rate_score.sort(key = lambda x:x[0])

        error_list=[]
        for kk in error_rate_score:
            error_list.append(kk[1])
        #take some good ones..
        best_list = error_list[:self.best_count ]
        
        #take some bad ones
        for i in range(random.randint(0,self.bad_count)):
            best_list.append(random.choice(error_list[self.best_count:]))
        
        #breed new child if current population is less than original
        diff = self.max_pop - len(best_list)

        for i in range(diff):
            F = random.choice(best_list)
            M = random.choice(best_list)

            if F!=M:
                C = self.cross_breed(F,M)
                C = self.mutate(C)
                best_list.append(C)
        
            self.all_pop = best_list       


# In[3]:



file_path = 'Andhra_dataset2.csv'
df = pd.read_csv(file_path)
col = df.columns

#cleaning..
plus = []
neg = []
k=0
for i in df['reslt']:
    if i == 1:
        plus.append(df.iloc[k])
    else:
        neg.append(df.iloc[k])
    k+=1

d={1:0,0:0}
for i in range((len(plus[0]))-1):
    cp=0
    cn =0
    d2={}
    d1={}
    for k in plus:
        if not np.isnan(k[i]):
            cp+=1
            d[1]+=1
            if k[i] in d1:
                d1[k[i]]+=1
            else:
                d1[k[i]] = 1

    for k in neg:
        if not np.isnan(k[i]):
            cn+=1
            d[0]+=1
            if k[i] in d2:
                d2[k[i]] +=1
            else:
                d2[k[i]] = 1
    
    valp = 0
    valn = 0
    for k in d1.keys():
        valp +=k*d1[k]
    valp/=cp
    
    for k in d2.keys():
        valn+=k*d2[k]
    valn/=cn
    
    for k in range(len(plus)):
        if np.isnan(plus[k][i]):
            plus[k][i] = valp
    
    for k in range(len(neg)):
        if np.isnan(neg[k][i]):
            neg[k][i] = valn
df = pd.DataFrame(plus+neg,columns=col)           


# In[4]:


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)


# function for normalizing columns in dataframe
def normalize(df):
    result = df.copy() # returns the copy of the dataframe
    max_vals = {} # dictionary for storing the maximum values ( its needed for normalizing the test data)
    min_vals = {} # dictionary for storing the minimum values ( ... )
    for feature in df.columns[:-1]:          # iterating through every feature
        max_value = df[feature].max()        # returning the max value of that feature
        min_value = df[feature].min()        # ............. min .....................
        max_vals[feature] = max_value        # adding the max value of a particular feature to a dictionary
        min_vals[feature] = min_value        # .......... min .............................................
        result[feature] = (df[feature] - min_value) / (max_value - min_value)     # normalizing :)
    return result, max_vals, min_vals        # return the modified dataframe and the other two dictionaries

# function to remove the un-needed columns
def remove_useless(df): 
    df.drop(['education', ], axis = 1, inplace = True)
    
# function to preprocess (normalize, remove useless columns) the test dataframe
def preprocess_test(df, max_vals, min_vals):
    remove_useless(df)
    df.fillna(df.mean(), inplace=True)  # replacing NaN cells with column average
    for feature in df.columns[:-1]:
        df[feature] = ( df[feature] - min_vals[feature] ) / (max_vals[feature] - min_vals[feature])
        
# function for plotting graph
def plot_graph(a):
    x=list(a.keys())
    y=list(a.values())
    plt.xlim(1,15)
    plt.ylim(-0.05,0.3)
    plt.xlabel("K Value")
    plt.ylabel("Error")
    plt.xticks(x)
    plt.plot(x,y)
    plt.show()


# In[5]:


#start..



col = np.array(col[:-1])
X = df[col]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 43)

sc = Sc()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# layering up the cnn

layers = [len(col), 4, 2]
fig = plt.figure(figsize=(12, 12))

draw_neural_net(fig.gca(), .1, .9, .1, .9, layers)

max_pop = 20

obj_back = backpropogation(layers, X_train, y_train)
        


# In[6]:


st = time.time()
mn_error = 1
#iterate for backpropogation..

for i in range(2000):
    err = obj_back.accuracy(X_test, y_test)
    if mn_error > err:
        network = obj_back.return_network()
        mn_error = err
        
    obj_back.backtrack()

print("max accuracy obtained in BackProp=>",1-mn_error)
print()
print()

#pass the result network to genetic algorithm...

# obj_sa = Simulated_Annealing(network, x_train, y_train, layers)
obj_ga = Genetics_Algorithm(max_pop, layers, X_train, y_train, network)

#iterate for Genetic algorithm..

for i in range(750):
    prd = obj_ga.accuracy(X_test,y_test)
    err = mean_squared_error(list(y_test), prd)
    
    if mn_error > err:
        mn_error = err
        best_prediction = prd
        
    obj_ga.update_weights()
    

print("max accuracy obtained in GA =>", 1-mn_error)




#KNN..
# function for normalizing columns in dataframe
def normalize(df):
    result = df.copy() # returns the copy of the dataframe
    max_vals = {} # dictionary for storing the maximum values ( its needed for normalizing the test data)
    min_vals = {} # dictionary for storing the minimum values ( ... )
    for feature in df.columns[:-1]:          # iterating through every feature
        max_value = df[feature].max()        # returning the max value of that feature
        min_value = df[feature].min()        # ............. min .....................
        max_vals[feature] = max_value        # adding the max value of a particular feature to a dictionary
        min_vals[feature] = min_value        # .......... min .............................................
        result[feature] = (df[feature] - min_value) / (max_value - min_value)     # normalizing :)
    return result, max_vals, min_vals        # return the modified dataframe and the other two dictionaries

# function to remove the un-needed columns
def remove_useless(df): 
    df.drop(['education', ], axis = 1, inplace = True)
    
# function to preprocess (normalize, remove useless columns) the test dataframe
def preprocess_test(df, max_vals, min_vals):
    remove_useless(df)
    df.fillna(df.mean(), inplace=True)  # replacing NaN cells with column average
    for feature in df.columns[:-1]:
        df[feature] = ( df[feature] - min_vals[feature] ) / (max_vals[feature] - min_vals[feature])
        
# function for plotting graph
def plot_graph(a):
    x=list(a.keys())
    y=list(a.values())
    plt.xlim(1,15)
    plt.ylim(-0.05,0.3)
    plt.xlabel("K Value")
    plt.ylabel("Error")
    plt.xticks(x)
    plt.plot(x,y)
    plt.show()
    
# loading data file into the program. give the location of your csv file
dataset = pd.read_csv("Andhra_dataset2.csv")
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state = 41)
remove_useless(train)
train.fillna(train.mean(), inplace=True)  # replacing NaN cells with column average
train, max_vals, min_vals = normalize(train)
preprocess_test(test, max_vals, min_vals)



# Euclidean Distance function
def euclid_dist(a, b, length):
    return np.sqrt(sum([np.square(a[_] - b[_]) for _ in range(length)])) # returns a pandas series with one value


def KNN(trainSet, testRow, K):
    distance_dict = dict()
    num_columns = testRow.shape[1]
    for _ in range(len(trainSet)):
        distance_dict[_] = euclid_dist(testRow, trainSet.iloc[_], num_columns)[0]
    t = sorted(distance_dict.items(), key=lambda x:x[1])
    neighbors = list(map(operator.itemgetter(0), t[0:K]))
    most_frequent = defaultdict(int)
    for i in neighbors:
        result = trainSet.iloc[i][-1]
        most_frequent[result] += 1
    return int(max(most_frequent.items(), key=operator.itemgetter(1))[0]), neighbors



from sklearn.metrics import confusion_matrix
plot_dict={}
for k in range(1, 7):
    print("For K =", k)
    y_true = []
    y_pred = []
    for i in range(test.shape[0]):
        lis_test = []
        lis_test.append(list(test.iloc[i].to_numpy()))
        lis_test = pd.DataFrame(lis_test)
        result, neigh = KNN(train, lis_test, k)
        y_true.append(test.iloc[i][-1])
        y_pred.append(result)
    cm = confusion_matrix(y_true, y_pred)
    acc = (cm[1][1]+cm[0,0])/(cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1])
    plot_dict[k] = 1 - acc
    print("Accuracy:", acc)
plot_graph(plot_dict)


# In[ ]:





# In[ ]:




