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
            
        elif  inquire[1] != 62:  # some teams gave the ranking for query 62 
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
## ---------------------------------------------------------------------------   

  
## -------------------------- function to calculate k_1 --------------------------
def CalK1(gamma, q_ratio, rankings, N1):
    num_experts = len(rankings)  # number of rankings in R 
    q = num_experts / q_ratio    # q-support   
    K1 = np.zeros(num_experts)       # to record K1 for each ranking
    
    A_R = []  # to record all the A matrices
    pos_mean_R = [] # to record the mean position of the items in all rankings
   
    
    # Caculate k1^{i}(q)  for all rankings
    for i in range(num_experts):
    
        ranking_i = rankings[i]  # get the ith ranking
        
        A = np.zeros((len(ranking_i), len(ranking_i))) # inintialize matrix A
        pos_mean = np.zeros((len(ranking_i), len(ranking_i)))
        
        index_j = 0   

        for j in ranking_i:       # for the item in ranking_i
                          
            f_j = 0  # initial value of the f function for item j of raning_i
            pos = np.zeros(num_experts) # to record the position of j in all rankings
            exist_j = 0   # to denote if item j already considered in other rankings
            
            # -- if no. of unconsidered rankings greater than q,  
            # -- check all considered rankings --
            if num_experts-i >= q:                 
 
                for i2 in range(num_experts):
                    # Get the potision of the selected item j in expters' rankings
                    if j in rankings[i2]:
                        rank_j = np.array([rankings[i2].index(j)])
                    else:
                        rank_j = np.array([])
                
                    # H function
                    H_j = 0
                
                    if rank_j.size > 0:      # if the item is in the ranking
                        if i2 < i:  # if the item j existing in one already considred ranking
                            if A_R[i2][rank_j[0]][rank_j[0]] != 0:  # if q-support pattern
                                pos_mean[index_j][index_j] = pos_mean_R[i2][rank_j[0]][rank_j[0]]
                                pos[i] = index_j + 1
                                pos_var = np.abs(pos[i]- pos_mean[index_j][index_j])   # deviation of current postion from mean
                                A[index_j][index_j] = 1 * np.power(gamma, pos_var)
                        
                                exist_j = 1
                                break
                            else:
                                exist_j = 1
                                break
                    
                        else:  # if current pattern not considered in constructed matrices              
                            H_j = 1         # count 1
                            pos[i2] = rank_j + 1 # index starts from 0 in phython, so position = rank_j+1               

                            f_j = f_j + H_j       # f function 
                            
                            # check if there is a possibility to be q-support
                            if f_j+num_experts-(i2+1)  < q: # if no possibility
                                break
   
                
                # set the value of A
                if f_j < q and exist_j == 0: 
                    A[index_j][index_j] = 0
                elif f_j >= q and exist_j == 0:            
                    pos_mean[index_j][index_j] = np.sum(pos) / f_j
                    pos_var = np.abs(pos[i]- pos_mean[index_j][index_j])   # deviation of current postion from mean
    
                    A[index_j][index_j] = 1 * np.power(gamma, pos_var)
                
                
            # -- if no. of unconsidered rankings less than q,  
            # -- no need to check all considered rankings --
            else: 
                for i2 in range(int(num_experts-q+1)):  # only check the first N-q+1 rankings
                    # Get the potision of the selected item j in expters' rankings
                    if j in rankings[i2]:
                        rank_j = np.array([rankings[i2].index(j)])
                    else:
                        rank_j = np.array([])
                
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
## --------------------------------------------------------------------------
        
    
## ----------------------------------    MAIN   -----------------------------
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
        
    ##------------- Fixed gamma, q_ratio -------------
    gamma = 0.9   
    q_ratio = 2  # no. of rankings/q_ratio = q-support
    
    N1 = topN
    
    num_experts = len(rankings_all)  # 12 teams
      
    # extract the ranking set for query j  (j = 0, ..., 65)
    j = 0
    
    K1 = []
    K1_mean = []
    K1_varUp = []
    K1_varDown = [] 
    
    for j in range(len(rankings_all[0])):
        
        R_j = []  # to record the rankings of all the teams for query j
        for i in range(num_experts):  #  expert i
            R_j.append(rankings_all[i][j])
              
        K1_j = CalK1(gamma, q_ratio, R_j, N1)
        K1_j_mean = np.mean(K1_j)
        K1_j_varUp = np.max(K1_j) - K1_j_mean
        K1_j_varDown = K1_j_mean - np.min(K1_j)    
         
        j = j + 1
        
        K1.append(K1_j)
        K1_mean.append(K1_j_mean)
        K1_varUp.append(K1_j_varUp)
        K1_varDown.append(K1_j_varDown)
    
    query_list = [x for x in range(1,68) if x !=62]  
    
    plt.figure()
    plt.bar(query_list, K1_mean)
    plt.grid('False')
    plt.xlim([0,68])         
    plt.xlabel('Query')
    plt.ylabel(r'$\bar{\kappa}_1(6)$')
    #plt.ylabel(r'$\bar{\kappa}_1(\lceil\frac{N}{2}\rceil)$')
    plt.savefig("K1meanCLEF.png")
    plt.show()
    
    temp1 = np.sort(K1_mean)
    temp2 = np.argsort(K1_mean)

    
    # ------------------ for  gamma = 0.9 ---------------    
    K1_mean[57]  # for query 58
    # 0.472145105609275
    K1_mean[24]
    # 0.44710891712542516
    K1_mean[23]
    # 0.41779620456906547
    K1_mean[54]
    # 0.4042048125135694
    
    K1_mean[62] # for query 64
    # 0.026077969900242054   
    K1_mean[47]
    # 0.03583320112287755
    K1_mean[10]
    # 0.08654263660739923
    K1_mean[32]
    # 0.10816580872128433