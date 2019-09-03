# -*- coding: utf-8 -*-
"""
Created in August 2018

@author: Zhengui
"""
# codes for the data set of mechanical turk dots

import numpy as np
import matplotlib.pyplot as plt

## rankings mechanical turk dots
DATAset1 = [(74, [1,2,3,4]), 
         (66, [1,3,4,2]), 
         (56, [1,3,2,4]), 
         (46, [1,2,4,3]),
         (42, [3,1,2,4]), 
         (42, [1,4,2,3]),
         (41, [2,1,3,4]),
         (38, [2,1,4,3]),
         (36, [2,3,1,4]),
         (35, [1,4,3,2]),
         (33, [2,4,1,3]),
         (31, [2,4,3,1]),
         (30, [3,4,2,1]),
         (27, [3,2,1,4]),
         (26, [3,1,4,2]),
         (24, [2,3,4,1]),
         (22, [4,1,3,2]),
         (20, [4,2,1,3]),
         (20, [3,2,4,1]),
         (19, [3,4,1,2]),
         (19, [4,3,2,1]),
         (19, [4,2,3,1]),
         (17, [4,1,2,3]),
         (12, [4,3,1,2])]

DATAset2 = [(91, [1,2,3,4]),
            (64, [2,1,3,4]), 
            (59, [1,3,2,4]),
            (58, [1,2,4,3]),
            (50, [2,1,4,3]),
            (49, [1,3,4,2]),
            (46, [1,4,2,3]),
            (35, [1,4,3,2]),
            (33, [2,3,1,4]),
            (30, [3,1,2,4]),
            (29, [3,2,1,4]),
            (27, [2,3,4,1]),
            (26, [4,1,2,3]),
            (22, [3,1,4,2]),
            (22, [3,2,4,1]),
            (22, [2,4,3,1]),
            (19, [3,4,2,1]),
            (19, [2,4,1,3]),
            (19, [4,2,3,1]),
            (19, [4,3,2,1]),
            (18, [4,1,3,2]),
            (17, [4,3,1,2]),
            (11, [4,2,1,3]),
            (9, [3,4,1,2])]

DATAset3 = [(129, [1,2,3,4]),
            (86, [1,2,4,3]),
            (75, [2,1,3,4]),
            (71, [1,3,2,4]),
            (52, [2,1,4,3]),
            (44, [1,3,4,2]),
            (42, [1,4,2,3]),
            (34, [3,1,2,4]),
            (31, [1,4,3,2]),
            (30, [3,2,1,4]),
            (27, [2,3,1,4]),
            (24, [2,4,1,3]),
            (17, [2,3,4,1]),
            (16, [2,4,3,1]),
            (15, [3,4,2,1]),
            (14, [4,1,2,3]),
            (13, [3,1,4,2]),
            (13, [4,2,1,3]),
            (13, [4,3,2,1]),
            (12, [4,3,1,2]),
            (11, [3,4,1,2]),
            (11, [4,2,3,1]),
            (10, [3,2,4,1]),
            (10, [4,1,3,2])]


DATAset4 = [(169, [1,2,3,4]),
            (88, [2,1,3,4]),
            (78, [1,2,4,3]),
            (76, [1,3,2,4]),
            (43, [2,1,4,3]),
            (32, [1,4,2,3]),
            (27, [1,3,4,2]),
            (26, [3,1,2,4]),
            (26, [3,2,1,4]),
            (26, [2,3,1,4]),
            (25, [2,3,4,1]),
            (25, [1,4,3,2]),
            (19, [3,2,4,1]),
            (18, [3,1,4,2]),
            (16, [4,1,2,3]),
            (15, [4,2,1,3]),
            (15, [3,4,1,2]),
            (13, [2,4,1,3]),
            (12, [4,1,3,2]),
            (12, [3,4,2,1]),
            (11, [4,3,2,1]),
            (8, [4,3,1,2]),
            (8, [2,4,3,1]),
            (6, [4,2,3,1])]

DATAset = (DATAset1, DATAset2, DATAset3, DATAset4)

## ----------------------- function to calculate k_1 -------------------------
def CalK1(rankings, num, q_ratio, gamma, N1):
    num_experts = sum(num)  # total number of rankings in R 
    num_UniqueRankings = rankings.shape[0]  # number of unique rankings
    q = num_experts / q_ratio    # q-support   
    K1 = np.zeros(num_UniqueRankings)   # to record K1 for each unique ranking
    
    A_R = []  # to record all the A matrices
    pos_mean_R = [] # to record the mean position of the items in all rankings
    
    # to find the index of the first N-q+1 rankings, will be used later
    temp_sum = 0 
    for i in range(num_UniqueRankings):
        temp_sum = temp_sum + num[i]
        if temp_sum >= num_experts-q+1:
            i_Nq1 = i
            break
        
    # Caculate k1^{i}(q)  for all rankings
    for i in range(num_UniqueRankings):    
        ranking_i = rankings[i]  # get the ith unique ranking        
        A = np.zeros((len(ranking_i), len(ranking_i))) # inintialize matrix A
        pos_mean = np.zeros((len(ranking_i), len(ranking_i)))
        
        index_j = 0 
        for j in ranking_i:       # for the item in ranking_i                         
            f_j = 0  # initial value of the f function for item j of raning_i
            pos = np.zeros(num_UniqueRankings) # to record the position of j in all unique rankings
            exist_i2 = 0   # to denote if item j already considered in other previously considered rankings
            
            # -- if no. of unconsidered rankings greater than q,  
            # -- check all considered rankings --
            if sum(num[i:])>= q:                 
 
                for i2 in range(num_UniqueRankings):
                    # Get the potision of the selected item j in expters' rankings
                    rank_j = np.where(rankings[i2] == j)[0] 
                    # H function
                    H_j = 0
                    if rank_j.size > 0:      # if the item is in the ranking
                        if i2 < i:  # if the item j existing in one already considred ranking
                            if A_R[i2][rank_j[0]][rank_j[0]] != 0:  # if q-support pattern
                                pos_mean[index_j][index_j] = pos_mean_R[i2][rank_j[0]][rank_j[0]]
                                pos[i] = index_j + 1                                                    
                                pos_var = np.abs(pos[i]- pos_mean[index_j][index_j])   # deviation of current postion from mean
                                
                                A[index_j][index_j] = 1 * np.power(gamma, pos_var)
                                
                                exist_i2 = 1
                                break
                        
                            else:
                                exist_i2 = 1
                                break
                                                    
                        else: # if current pattern not considered in constructed matrices
                            H_j = num[i2]         # count the number   
                            pos[i2] = rank_j + 1 # index starts from 0 in phython, so position = rank_j+1               
                            f_j = f_j + H_j       # f function  
                            
                            # check if there is a possibility to be q-support
                            if f_j + sum(num[i2+1:]) < q: # if no possibility
                                break
           
                # set the value of A
                if f_j < q and exist_i2 == 0: 
                    A[index_j][index_j] = 0
                elif f_j >= q and exist_i2 == 0:
                    pos_mean[index_j][index_j] = np.dot(pos, num) / f_j   # mean of positions 
                    pos_var = np.abs(pos[i]- pos_mean[index_j][index_j])   # deviation of current postion from mean
                    A[index_j][index_j] = 1 * np.power(gamma, pos_var)
             
            # -- if no. of unconsidered rankings less than q,  
            # -- no need to check all considered rankings --
            else: # when sum(num[i:]) < q:  only check the first N-q+1 rankings
                if i_Nq1 >= i: # this is possible for this dataset as 1 ranking may happen several times
                    i_Nq1 = i - 1
                    
                for i2 in range(i_Nq1+1):
                    # Get the potision of the selected item j in expters' rankings
                    rank_j = np.where(rankings[i2] == j)[0] 
                    
                    if rank_j.size > 0:      # if the item is in the ranking
                        if A_R[i2][rank_j[0]][rank_j[0]] != 0:  # if q-support pattern
                            pos_mean[index_j][index_j] = pos_mean_R[i2][rank_j[0]][rank_j[0]]
                            pos[i] = index_j + 1                                                    
                            pos_var = np.abs(pos[i]- pos_mean[index_j][index_j])   # deviation of current postion from mean
                                
                            A[index_j][index_j] = 1 * np.power(gamma, pos_var)
                            break                         
                
            index_j = index_j + 1
               
        K1[i] = np.sum(A) / N1
        A_R.append(A)
        pos_mean_R.append(pos_mean)

    return K1

## ---------------------------------------------------------------------------
K1_mean_DATAset = list() # to record the K2_mean of all the datasets
K1_varUp_DATAset = list() # to record the max upper variance of individal K2 to K2_mean
K1_varDown_DATAset = list() # to record the max lower variance of individal K2 to K2_mean

N1 = 4. 

for DATA in DATAset:   # consider each dataset R
    num = np.zeros(len(DATA))  # to record the number of each ranking in R
    rankings = np.zeros((len(DATA), len(DATA[0][1])))
    for i in range(len(DATA)):
        num[i] = DATA[i][0] 
        rankings[i] = DATA[i][1] 
            
    num_experts = sum(num)  # total number of experts 
       
    # ----------------- fixed gamma, diffrent q ------------------
    gamma = 1  # =1 means no weight
    #q_ratio = np.array([3, 2, 1.5, 1.2, 1])  # no. of rankings/q_ratio = q-support
    q_ratio = np.linspace(1, 3, num = 100)
    
    K1_mean = np.zeros(q_ratio.size) # to record the K1_mean of dataset R
    K1_varUp  = np.zeros(q_ratio.size) # to record the max upper variance of individal K1 to K1_mean
    K1_varDown = np.zeros(q_ratio.size) # to record the max lower variance of individal K1 to K1_mean
            
    index_i = 0
    for i in q_ratio:
        K1 = CalK1(rankings, num, i, gamma, N1)
    
        K1_mean[index_i] = np.dot(K1, num) / num_experts
        K1_varUp[index_i] = np.max(K1) - K1_mean[index_i]
        K1_varDown[index_i] = K1_mean[index_i] - np.min(K1)

        index_i = index_i + 1
               
    K1_mean_DATAset.append(K1_mean)
    K1_varUp_DATAset.append(K1_varUp)
    K1_varDown_DATAset.append(K1_varDown)


#print('Gamma = 1, no weight, different value of q:')    
#plt.figure()
#plt.errorbar(num_experts/q_ratio, K1_mean_DATAset[0], [K1_varDown_DATAset[0], K1_varUp_DATAset[0]], fmt='bo',capsize=4)
#plt.errorbar(num_experts/q_ratio, K1_mean_DATAset[3], [K1_varDown_DATAset[3], K1_varUp_DATAset[3]], fmt='go',capsize=4)
##plt.grid(True)  
#plt.xlabel('q')
#plt.ylabel(r'$\bar{\kappa}_1$')
#plt.legend(('Dataset 1', 'Dataset 4'))   
#plt.savefig("K1qSupport_distance.png")
#plt.show()

# for plot average k1 without variance
plt.figure()
plt.plot(num_experts/q_ratio, K1_mean_DATAset[0], 'b', 
         num_experts/q_ratio, K1_mean_DATAset[1], 'r', 
         num_experts/q_ratio, K1_mean_DATAset[2], 'k',
         num_experts/q_ratio, K1_mean_DATAset[3], 'g',
         )
#plt.grid(True)
plt.legend(('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'))        
plt.xlabel('q')
plt.ylabel(r'$\bar{\kappa}_1$')
plt.savefig("K1meanNoWeight.png")
plt.show()
 

## ------------------------ q fixed, different \gamma ------------------------
q_ratio = 2
gamma = np.array([1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2])

N1= 4

K1_mean_DATAset = list()
K1_varUp_DATAset = list()
K1_varDown_DATAset = list()

for DATA in DATAset:
    num = np.zeros(len(DATA))  # to record the number of each ranking in R
    rankings = np.zeros((len(DATA), len(DATA[0][1])))
    for i in range(len(DATA)):
        num[i] = DATA[i][0] 
        rankings[i] = DATA[i][1] 
        
    num_experts = sum(num)  # total number of experts 

    K1_mean = np.zeros(gamma.size)
    K1_varUp  = np.zeros(gamma.size)
    K1_varDown = np.zeros(gamma.size)
    
    index_i = 0    
    for i in gamma:
        K1 = CalK1(rankings, num, q_ratio, i, N1) 
         
        K1_mean[index_i] = np.dot(K1, num) / num_experts       
        K1_varUp[index_i] = np.max(K1)
        K1_varDown[index_i] = np.min(K1)
            
        index_i = index_i + 1  
       
    K1_mean_DATAset.append(K1_mean)
    K1_varUp_DATAset.append(K1_varUp)
    K1_varDown_DATAset.append(K1_varDown)


#print('q fixed, different Gamma, i.e., different weights:')
#plt.figure()
#plt.plot(gamma, K1_mean_DATAset[0], 'b')
#plt.fill_between(gamma, K1_varDown_DATAset[0], K1_varUp_DATAset[0], color  = (230. / 255., 230. / 255., 230. / 255.))
#plt.xlabel('$\lambda$')
#plt.title('Dataset 1')
#plt.ylabel(r'$\bar{\kappa}_1$')
#plt.savefig("K1Generaliz.png")


plt.figure()
plt.plot(gamma, K1_mean_DATAset[0], 'g-.', 
         gamma, K1_mean_DATAset[1], 'r--', 
         gamma, K1_mean_DATAset[2], 'k:',
         gamma, K1_mean_DATAset[3], 'b-',
         )
#plt.grid(True)
plt.legend(('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'))        
plt.xlabel('$\gamma$')
plt.ylabel(r'$\bar{\kappa}_1(\lceil\frac{N}{2}\rceil)$')
plt.savefig("K1mean.png")
plt.show()

## -------------------- q fixed, \gamma fixed to detect outliers -------------
K1_mean_DATAset = list()
K1_DATAset = list()

q_ratio = 2  # for a specific q
gamma = 0.5

N1 = 4

for DATA in DATAset:
    num = np.zeros(len(DATA))  # to record the number of each ranking in R
    rankings = np.zeros((len(DATA), len(DATA[0][1])))
    for i in range(len(DATA)):
        num[i] = DATA[i][0] 
        rankings[i] = DATA[i][1] 
        
    num_experts = sum(num)  # total number of experts 
          
    K1 = CalK1(rankings, num, q_ratio, gamma, N1)       
    K1_mean = np.dot(K1, num) / num_experts       
       
    K1_mean_DATAset.append(K1_mean)
    K1_DATAset.append(K1)

print('Dataset 1:')    
print((K1_DATAset[0] - K1_mean_DATAset[0])/K1_mean_DATAset[0])
print('\n') 

print('Dataset 2:')    
print((K1_DATAset[1] - K1_mean_DATAset[1])/K1_mean_DATAset[1])
print('\n')

print('Dataset 3:')    
print((K1_DATAset[2] - K1_mean_DATAset[2])/K1_mean_DATAset[2])
print('\n')

print('Dataset 4:')    
print((K1_DATAset[3] - K1_mean_DATAset[3])/K1_mean_DATAset[3])
print('\n')   


print(K1_mean_DATAset)