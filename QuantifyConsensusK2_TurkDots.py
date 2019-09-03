# -*- coding: utf-8 -*-
"""
Created in August 2019

@author: Zhengui
"""

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


## ---------------------- function to calculate k_2 -------------------------
def CalK2(rankings, num, q_ratio, Lambda, N2):
    num_experts = sum(num)  # total number of rankings in R 
    num_UniqueRankings = rankings.shape[0]  # number of unique rankings
    q = num_experts / q_ratio    # q-support   
    K2 = np.zeros(num_UniqueRankings)   # to record K2 for each unique ranking
    
    A_R = []  # to record all the A matrices
    dis_mean_R = [] # to record the mean distance of the items in all rankings
    
    # to find the index of the first N-q+1 rankings, will be used later
    temp_sum = 0 
    for i in range(num_UniqueRankings):
        temp_sum = temp_sum + num[i]
        if temp_sum >= num_experts-q+1:
            i_Nq1 = i
            break

    # Caculate k2^{i}(q)  for all rankings
    for i in range(num_UniqueRankings):    
        ranking_i = rankings[i]  # get the ith ranking
        A = np.zeros((len(ranking_i), len(ranking_i))) # inintialize matrix A
        dis_mean = np.zeros((len(ranking_i), len(ranking_i)))
                
        index_j = 0   # will be used to define the range k of inner loop
        for j in ranking_i[0:-1]:   # will compute the index_j th colum of matrix A
            
            index_k = index_j + 1            
            for k in ranking_i[index_j+1:]:   # will compute the index_k th row of matrix A
                f_j_k = 0  # initial value of the f function for r_{lj}r_{lk}
                dis = np.zeros(num_UniqueRankings)   # to record position gap between j and k
                
                exist_j_k = 0;   # to denote if item j already considered in other rankings
    
                # -- if no. of unconsidered rankings greater than q,  
                # -- check all considered rankings --
                if sum(num[i:])>= q:                 
                   
                    for i2 in range(num_UniqueRankings):
                        # Get the potisions of the selected two items (selected ranked items from the ranking of expert i) in expters' rankings
                        rank_j = np.where(rankings[i2] == j)[0] # the lower the more preferable
                        rank_k = np.where(rankings[i2] == k)[0]
                    
                        # H function
                        H_j_k = 0
                        # if both are in the ranking & position of k larger than j
                        if rank_j.size > 0  and rank_k.size > 0 and rank_k - rank_j > 0:
                            if i2 < i: # the current element of A for ranking i already considered before
                                if A_R[i2][rank_k[0]][rank_j[0]] != 0:  # if q-support pattern
                                    dis_mean[index_k][index_j] = dis_mean_R[i2][rank_k[0]][rank_j[0]]
                                    dis[i] = index_k - index_j
                                    dis_var = np.abs(dis[i]- dis_mean[index_k][index_j])   # deviation of current dis from mean
                                
                                    A[index_k][index_j] =  1* np.power(Lambda, dis_var)
                                    exist_j_k = 1;
                                    break
                                else:
                                    exist_j_k = 1;
                                    break
                            else:  # if current pattern not considered in constructed matrices                  
                                H_j_k = num[i2]
                                dis[i2] = rank_k - rank_j
                                f_j_k = f_j_k + H_j_k   # f function 
                                
                                # check if there is a possibility to be q-support
                                if f_j_k + sum(num[i2+1:]) < q: # if no possibility
                                    break
                                
                    
                    if f_j_k < q  and exist_j_k == 0: 
                        A[index_k][index_j] = 0
                    elif f_j_k >= q and exist_j_k == 0:
                        dis_mean[index_k][index_j] = np.dot(dis, num) / f_j_k                   
                        dis_var = np.abs(dis[i]- dis_mean[index_k][index_j])                   
                        A[index_k][index_j] = 1 * np.power(Lambda, dis_var)
    
                # -- if no. of unconsidered rankings less than q,  
                # -- no need to check all considered rankings --
                else: # when sum(num[i:]) < q:  only check the first N-q+1 rankings
                    if i_Nq1 >= i: # this is possible for this dataset as 1 ranking may happen several times
                        i_Nq1 = i - 1
                        
                    for i2 in range(i_Nq1+1):
                        # Get the potisions of the selected two items (selected ranked items from the ranking of expert i) in expters' rankings
                        rank_j = np.where(rankings[i2] == j)[0] # the lower the more preferable
                        rank_k = np.where(rankings[i2] == k)[0]

                        if rank_j.size > 0  and rank_k.size > 0 and rank_k - rank_j > 0:
                            if A_R[i2][rank_k[0]][rank_j[0]] != 0:  # if q-support pattern
                                dis_mean[index_k][index_j] = dis_mean_R[i2][rank_k[0]][rank_j[0]]
                                dis[i] = index_k - index_j
                                dis_var = np.abs(dis[i]- dis_mean[index_k][index_j])   # deviation of current dis from mean
                                
                                A[index_k][index_j] =  1* np.power(Lambda, dis_var)
                                break              
    
                index_k = index_k + 1 
                                    
            index_j = index_j + 1
            
            
        K2[i] = (np.sum(A) - np.trace(A))/N2
        A_R.append(A)
        dis_mean_R.append(dis_mean)
        
        
    return K2, A_R     

            
##---------------------------------------------------------------------------

K2_mean_DATAset = list() # to record the K2_mean of all the datasets
K2_varUp_DATAset = list() # to record the max upper variance of individal K2 to K2_mean
K2_varDown_DATAset = list() # to record the max lower variance of individal K2 to K2_mean

for DATA in DATAset:   # consider each dataset R
    num = np.zeros(len(DATA))  # to record the number of each ranking in R
    rankings = np.zeros((len(DATA), len(DATA[0][1])))
    for i in range(len(DATA)):
        num[i] = DATA[i][0] 
        rankings[i] = DATA[i][1] 
        
    num_experts = sum(num)  # total number of experts
        
    # -------- fixed \lambda, different values of q ----------  
    Lambda = 1      #  = 1 means no weight    
    #q_ratio = np.array([3, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2, 1.8, 1.7, 1.6, 1.5, 1.2, 1])    # no. of rankings/q_ratio = q-support 
    q_ratio = np.linspace(1, 2, num = 400)
    
    N2 = 4 * 3 / 2.
    
    K2_mean = np.zeros(q_ratio.size) # to record the K2_mean of dataset R
    K2_varUp  = np.zeros(q_ratio.size) # to record the max upper variance of individal K2 to K2_mean
    K2_varDown = np.zeros(q_ratio.size) # to record the max lower variance of individal K2 to K2_mean
            
    index_i = 0    
    for i in q_ratio:
        K2, A_R = CalK2(rankings, num, i, Lambda, N2)
         
        K2_mean[index_i] = np.dot(K2, num) / num_experts
        K2_varUp[index_i] = np.max(K2) - K2_mean[index_i]
        K2_varDown[index_i] = K2_mean[index_i] - np.min(K2)
            
        index_i = index_i + 1
       
    K2_mean_DATAset.append(K2_mean)
    K2_varUp_DATAset.append(K2_varUp)
    K2_varDown_DATAset.append(K2_varDown)
    
print('lambda = 1, no weight, different value of q:')    
plt.figure()
plt.errorbar(1/q_ratio, K2_mean_DATAset[0], [K2_varDown_DATAset[0], K2_varUp_DATAset[0]], fmt='bo',capsize=4)
plt.errorbar(1/q_ratio, K2_mean_DATAset[3], [K2_varDown_DATAset[3], K2_varUp_DATAset[3]], fmt='go',capsize=4)
#plt.grid(True)  
plt.xlabel('q / N')
plt.ylabel(r'$\bar{\kappa}_2(q)$')
plt.legend(('Dataset 1', 'Dataset 4'))   
plt.savefig("K2qSupport_distance.png")
plt.show()

# for plot average k1 without variance
plt.figure()
plt.plot(1/q_ratio, K2_mean_DATAset[0], 'g-.',  
         1/q_ratio, K2_mean_DATAset[1], 'r--',
         1/q_ratio, K2_mean_DATAset[2], 'k:',
         1/q_ratio, K2_mean_DATAset[3], 'b-',
         )
#plt.grid(True)
plt.legend(('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'))        
plt.xlabel('q / N')
plt.ylabel(r'$\bar{\kappa}_2(q)$')
plt.savefig("K2meanNoWeight.png")
plt.show()
## --------------------------------------------------------------------------


## ------------------------  q fixed, different \lembda  --------------------
K2_mean_DATAset = list()
K2_varUp_DATAset = list()
K2_varDown_DATAset = list()

q_ratio = 2  # for a specific q
Lambda = np.array([1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2])

N2 = 4 * 3 / 2.

for DATA in DATAset:
    num = np.zeros(len(DATA))  # to record the number of each ranking in R
    rankings = np.zeros((len(DATA), len(DATA[0][1])))
    for i in range(len(DATA)):
        num[i] = DATA[i][0] 
        rankings[i] = DATA[i][1] 
        
    num_experts = sum(num)  # total number of experts 

    K2_mean = np.zeros(Lambda.size)
    K2_varUp  = np.zeros(Lambda.size)
    K2_varDown = np.zeros(Lambda.size)
            
    index_i = 0    
    for i in Lambda:
        K2, A_R = CalK2(rankings, num, q_ratio, i, N2) 
         
        K2_mean[index_i] = np.dot(K2, num) / num_experts       
        K2_varUp[index_i] = np.max(K2)
        K2_varDown[index_i] = np.min(K2)
            
        index_i = index_i + 1
       
    K2_mean_DATAset.append(K2_mean)
    K2_varUp_DATAset.append(K2_varUp)
    K2_varDown_DATAset.append(K2_varDown)
    

plt.figure()
plt.plot(Lambda, K2_mean_DATAset[0], 'g-.', 
         Lambda, K2_mean_DATAset[1], 'r--', 
         Lambda, K2_mean_DATAset[2], 'k:',
         Lambda, K2_mean_DATAset[3], 'b-',
         )
#plt.grid(True)
plt.legend(('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'))        
plt.xlabel('$\lambda$')
plt.ylabel(r'$\bar{\kappa}_2(\lceil\frac{N}{2}\rceil)$')
plt.savefig("K2mean.png")
plt.show()
## --------------------------------------------------------------------------


## -----------------   q fixed, \lembda fixed to detect outliers  -----------
K2_mean_DATAset = list()
K2_DATAset = list()

q_ratio = 2  # for a specific q
Lambda = 0.5

N2 = 4 * 3 / 2.

for DATA in DATAset:
    num = np.zeros(len(DATA))  # to record the number of each ranking in R
    rankings = np.zeros((len(DATA), len(DATA[0][1])))
    for i in range(len(DATA)):
        num[i] = DATA[i][0] 
        rankings[i] = DATA[i][1] 
        
    num_experts = sum(num)  # total number of experts 
            
    K2, A_R = CalK2(rankings, num, q_ratio, Lambda, N2)       
    K2_mean = np.dot(K2, num) / num_experts       
       
    K2_mean_DATAset.append(K2_mean)
    K2_DATAset.append(K2)

print('Dataset 1:')    
print((K2_DATAset[0] - K2_mean_DATAset[0])/K2_mean_DATAset[0])
print('\n') 

print('Dataset 2:')    
print((K2_DATAset[1] - K2_mean_DATAset[1])/K2_mean_DATAset[1])
print('\n')

print('Dataset 3:')    
print((K2_DATAset[2] - K2_mean_DATAset[2])/K2_mean_DATAset[2])
print('\n')

print('Dataset 4:')    
print((K2_DATAset[3] - K2_mean_DATAset[3])/K2_mean_DATAset[3])
print('\n')   
