# -*- coding: utf-8 -*-
"""
Created in August 2019

@author: Zhengui
"""

import numpy as np
import matplotlib.pyplot as plt

## -- Function to read all the rankings of the 12 teams for the 66 queries -- 
def extractRankings(input_file, topN, output_file):

    fr = open(input_file, 'r')
    
    # Read file (line by line)   
    i = 0
    previous_inquire_name = ''
    
    rank_list = []
    for line in fr:
        #print(line)
        
        line_list = line.split()
        #print(line_list[2])    # print the ranked item
        
        inquire = line_list[0].split('.') # split e.g., 'qtest.1' to 'qtest' and '1'
        #print(inquire)
        
        if inquire[1] != previous_inquire_name and inquire[1] != '62':            
            previous_inquire_name = inquire[1]
            #print(previous_inquire_name)
            
            curr_rank = []
            curr_rank.append(line_list[2])
            j = 1
            
        elif  inquire[1] != 62:
            j = j + 1
            if j <= topN:
                curr_rank.append(line_list[2])
                
                #print(curr_rank)
                if j == topN:
                    rank_list.append(curr_rank)
            else:                 
                continue
       
    fr.close()
#    print(rank_list)
      
    # Save results
    fw = open(output_file, 'w')
    
    for rank in rank_list:
        str_line = ' '.join(rank)
        fw.write(str_line)
        fw.write('\n')
        
    fw.close()
    
    
    return rank_list

## --------------------------------------------------------------------------   


## ------------------- function to calculate k_2 ------------------------
def CalK2(Lambda, q_ratio, rankings, N2):  
    
    num_experts = len(rankings) # no. of rankings in R 
    q = num_experts / q_ratio   # q-support   
    K2 = np.zeros(num_experts)   # to record the K2 for each ranking
    
    A_R = []  # to record all the A matrices
    dis_mean_R = [] # to record the mean distance of the items in all rankings

    # Caculate k2^{i}(q)  for all rankings
    for i in range(num_experts):    
        
        ranking_i = rankings[i]  # get the ith ranking

        A = np.zeros((len(ranking_i), len(ranking_i))) # inintialize matrix A
        dis_mean = np.zeros((len(ranking_i), len(ranking_i)))
        
        index_j = 0   # will be used to define the range k of inner loop
        for j in ranking_i[0:-1]:   # will compute the index_j th colum of matrix A
            
            index_k = index_j + 1
            for k in ranking_i[index_j + 1:]:   # will compute the index_k th row of matrix A
                          
                f_j_k = 0  # initial value of the f function for r_{lj}r_{lk}
                dis = np.zeros(num_experts)   # to record position gap between j and k
                
                exist_j_k = 0;   # to denote if item j already considered in other rankings
                
                # -- if no. of unconsidered rankings greater than q,  
                # -- check all considered rankings --
                if num_experts-i >= q:                 
 
                    for i2 in range(num_experts):
                        # Get the potisions of the selected two items (selected ranked items from the ranking of expert i) in expters' rankings
                        if j in rankings[i2]:
                            rank_j = np.array([rankings[i2].index(j)])
                        else:
                            rank_j = np.array([])
                        
                        if k in rankings[i2]:
                            rank_k = np.array([rankings[i2].index(k)])
                        else:
                            rank_k = np.array([])
                   
                        # H function
                        H_j_k = 0
                        # if both are in the ranking & position of k larger than j
                        if rank_j.size > 0  and rank_k.size > 0 and rank_k[0] - rank_j[0] > 0:
                            if i2 < i:
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
   
                            else:                    
                                H_j_k = 1
                                dis[i2] = rank_k[0] - rank_j[0]
                                f_j_k = f_j_k + H_j_k   # f function
                                
                                # check if there is a possibility to be q-support
                                if f_j_k+num_experts-(i2+1)  < q: # if no possibility
                                    break   
                                    
                    if f_j_k < q  and exist_j_k == 0: 
                        A[index_k][index_j] = 0
                    elif f_j_k >= q and exist_j_k == 0:
                        dis_mean[index_k][index_j] = np.sum(dis) / f_j_k
                        dis_var = np.abs(dis[i]- dis_mean[index_k][index_j])
                    
                        A[index_k][index_j] = 1 * np.power(Lambda, dis_var)
                
                # -- if no. of unconsidered rankings less than q,  
                # -- no need to check all considered rankings --
                else: 
                    for i2 in range(int(num_experts-q+1)):  # only check the first N-q+1 rankings
                        # Get the potisions of the selected two items (selected ranked items from the ranking of expert i) in expters' rankings
                        if j in rankings[i2]:
                            rank_j = np.array([rankings[i2].index(j)])
                        else:
                            rank_j = np.array([])
                        
                        if k in rankings[i2]:
                            rank_k = np.array([rankings[i2].index(k)])
                        else:
                            rank_k = np.array([])
                   
                        # if both are in the ranking & position of k larger than j
                        if rank_j.size > 0  and rank_k.size > 0 and rank_k[0] - rank_j[0] > 0:
                            if A_R[i2][rank_k[0]][rank_j[0]] != 0:  # if q-support pattern
                                dis_mean[index_k][index_j] = dis_mean_R[i2][rank_k[0]][rank_j[0]]
                                dis[i] = index_k - index_j
                                dis_var = np.abs(dis[i]- dis_mean[index_k][index_j])   # deviation of current dis from mean
                                
                                A[index_k][index_j] =  1* np.power(Lambda, dis_var)
                                break
                            
                
                index_k = index_k + 1 
                                    
            index_j = index_j + 1
            
            
        K2[i] = (np.sum(A) - np.trace(A)) / N2
        A_R.append(A)
        dis_mean_R.append(dis_mean)
        
    return K2
## --------------------------------------------------------------------------   

  
        
## ----------------------------------    MAIN   ----------------------------
if __name__ == '__main__':
    print('Starting...')
    
    ##---Read all the rankings of the 12 teams for the 66 queries----  
    INPUT_FILES = ['CUNI_EN_Run.1.dat', 'ECNU_EN_Run.1.dat', 'FDUSGInfo_EN_Run.1.dat', 'GRIUM_EN_Run.1.dat', 'KISTI_EN_RUN.1.dat', 'KUCS_EN_Run.1.dat', 'LIMSI_EN_run.1.dat', 'Miracl_EN_Run.1.dat', 'TeamHCMUS_EN_Run.1.dat', 'UBML_EN_Run.1.dat', 'USST_EN_Run.1.dat', 'YorkU_EN_Run.1.dat']
    OUTPUT_FILES = ['TOP20_CUNI_EN.dat', 'TOP20_ECNU_EN.dat', 'TOP20_FDUSGInfo_EN.dat', 'TOP20_GRIUM_EN.dat', 'TOP20_KISTI_EN.dat', 'TOP20_KUCS_EN.dat', 'TOP20_LIMSI_EN.dat', 'TOP20_Miracl_EN.dat', 'TOP20_TeamHCMUS_EN.dat', 'TOP20_UBML_EN.dat', 'TOP20_USST_EN.dat', 'TOP20_YorkU_EN.dat']
   
    rankings_all = []
    
    for i in range(len(INPUT_FILES)):
        input_file = 'Run1\\' + INPUT_FILES[i]
        output_file = OUTPUT_FILES[i]
    
        topN = 20        # Care about topN items
 
        rank_list = extractRankings(input_file, topN, output_file)  # rankings of a team for all queries
        
        rankings_all.append(rank_list)  # record all teams' rankings

        
    ##-------------- Fixed Lambda, q_ratio -------------
    Lambda = 0.9  
    q_ratio = 2  # no. of rankings/q_ratio = q-support
    
    N2 = topN * (topN - 1) / 2.
    
    num_experts = len(rankings_all)  # 12 teams
      
    # extract the ranking set for query j  (j = 0, ..., 65)
    j = 0
    
    K2 = []
    K2_mean = []
    K2_varUp = []
    K2_varDown = [] 
    
    for j in range(len(rankings_all[0])):
        
        R_j = []  # to record the rankings of all the teams for query j
        for i in range(num_experts):  #  expert i
            R_j.append(rankings_all[i][j])
              
        K2_j = CalK2(Lambda, q_ratio, R_j, N2)
        K2_j_mean = np.mean(K2_j)
        K2_j_varUp = np.max(K2_j) - K2_j_mean
        K2_j_varDown = K2_j_mean - np.min(K2_j)    
         
        j = j + 1
        
        K2.append(K2_j)
        K2_mean.append(K2_j_mean)
        K2_varUp.append(K2_j_varUp)
        K2_varDown.append(K2_j_varDown)
    
    query_list = [x for x in range(1,68) if x !=62]  
    
    plt.figure()
    plt.bar(query_list, K2_mean)
    plt.grid('False')
    plt.xlim([0,68])          
    plt.xlabel('Query')
    #plt.ylabel(r'$\bar{\kappa}_2(\lceil\frac{N}{2}\rceil)$')
    plt.ylabel(r'$\bar{\kappa}_2(6)$')
    plt.savefig("K2meanCLEF.png")
    plt.show()
    
    temp3 = np.sort(K2_mean)
    temp4 = np.argsort(K2_mean)
    

    # --------------- for  Lambda = 0.9 --------------- 
    K2_mean[57]  # for query 58
    # 0.26908766461186845
    K2_mean[24] # for query 25
    # 0.17456994507271986
    K2_mean[23] # for query 24
    # 0.14852879455572565
    K2_mean[54] # for query 55
    # 0.15013113787975732
    
    K2_mean[62] # for query 64
    #   0.0
    K2_mean[47] # for query 48
    #  0.0
    K2_mean[10] # for query 11
    #  0.0034343543189698533
    K2_mean[32] # for query 33
    # 0.006562211167002729



    
    