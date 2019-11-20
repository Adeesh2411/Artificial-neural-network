
#ANN using Simulated Annealing and Genetic Algorithm(eagle strategy)..

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

# start ...
def generateColumns(start, end):
    l=[]
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return np.array(l)

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
        l.append(np.random.randn(j,i))
    return l


#This class represents each network 
class population:
    def __init__(self,NN):
        self.weights = generate_weights(NN)
        self.bias = generate_bias(NN)
        self.no_layers = len(NN)

        
#simulated annealing..
class Simulated_Annealing:
    def __init__(self, child, x, y, layers):
        self.network = child
        self.alpha = 0.9
        self.x = x
        self.y = y
        self.layers = layers
        self.num_layers = len(layers)
    
    #annealing procedure..
    def annealing(self):
        T = 1
        prd = self.accuracy(self.network, self.x, self.y)
        initial_error = mean_squared_error(list(self.y), prd)
        init_network = copy.deepcopy(self.network)
        
        for i in range(25):
            cur_cost =  mean_squared_error(list(self.y), self.accuracy(self.network, self.x, self.y))
            neigh_network = self.random_neighbour()
            neigh_cost = mean_squared_error(list(self.y), self.accuracy(neigh_network, self.x, self.y))
            diff = cur_cost - neigh_cost

            if diff <=0:
                temp = np.exp(-(diff/T))
                if np.random.uniform(0,1) <= temp:
                    self.network = neigh_network
                    cur_cost = neigh_cost
            else:
                self.network = neigh_network
                cur_cost = neigh_cost
                
            T = T * self.alpha #update temperature 
            
        #final error for network trained by SA
        final_error = mean_squared_error(list(self.y), self.accuracy(self.network, self.x, self.y))
        
        if initial_error > final_error:
            print("weight updated in Simulated annealing")
            print("previous error = >",initial_error)
            print("current error = >",final_error)
            print()
            return self.network
        
        #else return initial network
        return init_network          
        
    #generate random population
    def random_neighbour(self):
        return population(self.layers)

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
    
    #returns the list of predicted value wrt test value
    def accuracy(self,network, test_x,test_y):
        l=[]
        for i in range(len(test_y)):
            predicted_value = self.get_output(network, test_x[i].reshape(-1,1))
            l.append(predicted_value[0])
        return l
        
#genetic algorithm class.
class Genetics_Algorithm:
    def __init__(self, n, layers, x, y):
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
        for i in range(n):
            self.all_pop.append(population(layers))
        
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
    def update_weights(self,nn):
        if nn == 0:
            self.all_pop[0] = Simulated_Annealing(self.all_pop[0], self.x, self.y, self.layers).annealing()
            return
            
        #calculate score for each population
        error_rate_score = list(zip(self.calculate_score(),self.all_pop))
        #sort based on error
        error_rate_score.sort(key = lambda x:x[0])

        error_list=[]
        for kk in error_rate_score:
            error_list.append(kk[1])
        #take some good ones..
        best_list = error_list[:self.best_count]
        
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
                #C = Simulated_Annealing(C, self.x, self.y, self.layers).annealing()
                best_list.append(C)
        
            self.all_pop = best_list

#get the column name from data set..
eyes = generateColumns(1,12)
#np.random.seed(1)
#specify file path here
file_path = 'D:\learning\SEM 5\AI\project\Project_Related\EYE\Project\ANN-using-Eagle-stratergy\Eyes.csv'
df = pd.read_csv(file_path)
X = df[eyes]
y = df['truth_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

sc = Sc()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# layering up the cnn
layers = [24,12,12,6]

#maximum population in a given network..
max_pop = 15
obj1 = Genetics_Algorithm(max_pop, layers, X_train, y_train)

st = time.time()
mn_error = 1

print("Training Started ... ")
print()

for i in range(50):
    prd = obj1.accuracy(X_test,y_test)
    err = mean_squared_error(list(y_test), prd)
    
    if mn_error > err:
        mn_error = err
        #best_weight = obj1.all_pop[0].weights
        #best_bias = obj1.all_pop[0].bias
        best_prediction = prd

    print("iteration => : ",(i+1))
    print("time taken => ",time.time()-st)
    print("error => : ",err)
    print("minmum error so far => ",mn_error)
    print(".........................................................................")
    if i<=10:
        obj1.update_weights(0)# 0 for simulated annealing
    else:
        obj1.update_weights(1)# 1 for GA
    
print("max accuracy obtained = > ",(1-mn_error)*100)