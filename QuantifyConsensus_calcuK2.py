# -*- coding: utf-8 -*-
"""
Created on March 12, 2018

@author: Zhengui
"""

import numpy as np
import matplotlib.pyplot as plt

rankings_google = np.array([[0,68,9,59,11,5,3,69,79,70,21,4,36,32,76,40,60,51,80,81,42,29,82,83,73],
[5,0,9,11,59,76,3,36,21,79,90,70,4,60,93,35,50,40,42,92,86,87,73,98,94],
[0,9,11,5,3,59,70,84,85,21,76,4,32,42,51,62,12,80,67,60,55,86,87,73,29],
[0,9,3,11,5,2,59,4,32,21,60,35,42,61,62,12,51,17,63,64,55,65,66,40,67],
[5,0,68,9,3,11,59,69,70,71,4,21,60,32,36,61,65,72,73,58,74,75,76,77,78],
[0,9,59,5,11,3,88,60,89,69,90,70,32,91,75,92,93,94,61,40,86,87,95,96,97]]) 

rankings_bing = np.array([[0,9,1,11,36,8,4,2,22,5,37,15,38,6,28,34,29,19,39,40,35,24,41,32,42],
[0,1,9,2,11,5,50,8,4,56,57,22,3,40,51,7,41,15,19,37,36,55,42,58,38],
[0,7,9,2,10,8,4,6,5,1,11,14,3,40,13,43,19,44,45,46,47,48,49,17,28],
[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
[9,1,25,4,7,2,0,26,5,11,15,8,27,10,28,29,30,31,6,19,32,22,33,34,35],
[0,9,3,1,2,11,50,4,8,7,6,38,40,25,10,51,15,19,52,36,53,14,54,55,28]])

################ function to calculate k_2 ################
def CalK2(rankings, q_ratio, Lambda):
    num_experts = rankings.shape[0] # no. of rankings in R 
    q = num_experts / q_ratio   # q-support   
    K2 = np.zeros(num_experts)   # to record the K2 for each ranking
    
    A_R = []  # to record all the A matrices
        
    # Caculate k2^{i}(q)  for all rankings
    for i in range(num_experts):    
        
        ranking_i = rankings[i]  # get the ith ranking

        A = np.zeros((len(ranking_i), len(ranking_i))) # inintialize matrix A
        
        index_j = 0   # will be used to define the range k of inner loop

        for j in ranking_i[0:-1]:   # will compute the index_j th colum of matrix A
            
            index_k = index_j + 1
            
            for k in ranking_i[index_j + 1:]:   # will compute the index_k th row of matrix A
                          
                f_j_k = 0  # initial value of the f function for r_{lj}r_{lk}
                dis = np.zeros(num_experts)   # to record position gap between j and k
                
                exist_j_k = 0;   # to denote if item j already considered in other rankings
           
                
                for i2 in range(num_experts):
                    # Get the potisions of the selected two items (selected ranked items from the ranking of expert i) in expters' rankings
                    rank_j = np.where(rankings[i2] == j)[0] # the lower the more preferable
                    rank_k = np.where(rankings[i2] == k)[0]
                    
                    # H function
                    H_j_k = 0
                    # if both are in the ranking & position of k larger than j
                    if rank_j.size > 0  and rank_k.size > 0 and rank_k - rank_j > 0:
                        if i2 < i:
                            A[index_k][index_j] =  A_R[i2][rank_k[0]][rank_j[0]]
                            exist_j_k = 1;
                            break
                        else:                    
                            H_j_k = 1
                            dis[f_j_k] = rank_k - rank_j
                            f_j_k = f_j_k + H_j_k   # f function 
                                    
                if f_j_k < q  and exist_j_k == 0: 
                    A[index_k][index_j] = 0
                elif f_j_k >= q and exist_j_k == 0:
                    dis_mean = np.sum(dis) / f_j_k
                    
                    dis_var = np.abs(dis[0:f_j_k] - dis_mean)
                    
                    dis_var_mean = np.mean(dis_var)
                    
                    A[index_k][index_j] = (f_j_k / num_experts) * np.power(Lambda, dis_var_mean)
                    
                index_k = index_k + 1 
                                    
            index_j = index_j + 1
            
            
        K2[i] = np.sum(A) - np.trace(A)
        A_R.append(A)
        
    return K2         
#######################################################


########### Evaluation of rankings with fixed lambda, diffrent q ##########
def evaluateQ(Lambda, q_ratio):       
 
    K2_google_mean = np.zeros(q_ratio.size)
    K2_google_varUp  = np.zeros(q_ratio.size)
    K2_google_varDown = np.zeros(q_ratio.size)
    
    K2_bing_mean = np.zeros(q_ratio.size)
    K2_bing_varUp  = np.zeros(q_ratio.size)
    K2_bing_varDown = np.zeros(q_ratio.size)
    
    index_i = 0
    
    for i in q_ratio:
        K2_google = CalK2(rankings_google, i, Lambda)
        K2_bing = CalK2(rankings_bing, i, Lambda)
        
        K2_google_mean[index_i] = np.mean(K2_google)
        K2_google_varUp[index_i] = np.max(K2_google) - K2_google_mean[index_i]
        K2_google_varDown[index_i] = K2_google_mean[index_i] - np.min(K2_google)
            
        K2_bing_mean[index_i] = np.mean(K2_bing)
        K2_bing_varUp[index_i] = np.max(K2_bing) - K2_bing_mean[index_i]
        K2_bing_varDown[index_i] = K2_bing_mean[index_i] - np.min(K2_bing)
            
        
        index_i = index_i + 1
    
    num_experts = K2_google.shape[0]
    plt.figure()
    plt.errorbar(num_experts/q_ratio, K2_google_mean, [K2_google_varDown,K2_google_varUp], fmt='bo',capsize=4)
    plt.errorbar(num_experts/q_ratio, K2_bing_mean, [K2_bing_varDown,K2_bing_varUp], fmt='r*',capsize=4)
    
    plt.grid('True')
    plt.legend(['Google', 'Bing'])       
    plt.xlabel('q')
    plt.ylabel(r'$\bar{\kappa}_2$')
    plt.savefig("K2qSupport_distance.png")
    plt.show()
    
    # for plot average k1 without variance
    plt.figure()
    plt.plot(num_experts/q_ratio, K2_google_mean, 'bo', num_experts/q_ratio, K2_bing_mean, 'r*')
    plt.grid('True')
    plt.legend(['Google', 'Bing'])
             
    plt.xlabel('q')
    plt.ylabel(r'$\bar{\kappa}_2$')
    plt.savefig("K2meanNoWeight.png")
    plt.show()
##########################################################################
    
    
########### Evaluation of rankings with fixed q, diffrent \lembda #########
def evaluateLembda(q_ratio, Lambda): 

    K2_google_mean = np.zeros(Lambda.size)
    K2_google_varUp  = np.zeros(Lambda.size)
    K2_google_varDown = np.zeros(Lambda.size)
    #
    K2_bing_mean = np.zeros(Lambda.size)
    K2_bing_varUp  = np.zeros(Lambda.size)
    K2_bing_varDown = np.zeros(Lambda.size)
    
    index_i = 0
    
    for i in Lambda:
        K2_google = CalK2(rankings_google, q_ratio, i) 
        K2_bing = CalK2(rankings_bing, q_ratio, i)
        
        K2_google_mean[index_i] = np.mean(K2_google)
        K2_google_varUp[index_i] = np.max(K2_google)
        K2_google_varDown[index_i] = np.min(K2_google)
            
        K2_bing_mean[index_i] = np.mean(K2_bing)
        K2_bing_varUp[index_i] = np.max(K2_bing)
        K2_bing_varDown[index_i] = np.min(K2_bing)
    
        index_i = index_i + 1
    
    
    plt.figure()
    plt.plot(Lambda, K2_google_mean, 'b', Lambda, K2_bing_mean, '-.r')
    plt.legend(['Google', 'Bing'])
    plt.fill_between(Lambda, K2_google_varDown, K2_google_varUp, color  = (230. / 255., 230. / 255., 230. / 255.))
    plt.fill_between(Lambda, K2_bing_varDown, K2_bing_varUp, color  = (190. / 255., 190. / 255., 190. / 255.))
    plt.xlabel('$\lambda$')
    plt.ylabel(r'$\bar{\kappa}_2$')
    plt.savefig("K2Generaliz.png")
##########################################################################

if __name__ == '__main__':
    
    Lambda = 1      #  = 1 means no weight   
    q_ratio = np.array([3, 2, 1.5, 1.2, 1])   # no. of rankings/q_ratio = q-support
    evaluateQ(Lambda, q_ratio)
       
    q_ratio = 1.5
    Lambda = np.array([1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2])
    evaluateLembda(q_ratio, Lambda) 