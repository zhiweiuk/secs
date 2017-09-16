import numpy as np
import math

import Symbol

class Concordance:
    def __init__(self):
        pass

    def calculate_concordance(self,X):

        return 0

class WeightedConsensusMeasure(Concordance):
    '''
    
     If you have any questions or want to report bugs,
     please contact Dr. Zhiwei Lin at http://scm.ulster.ac.uk/zhiwei.lin/
    '''

    def calculate_concordance(self, R,lmbd=1,gamma=1):
        '''
        Given a set R of rankings, this will calculate   $\kappa(R)$ of R
        shown in the paper.

        :param R:
        :type R:
        :return $\kappa(X)$, $\ell(X)$, the set of the longest common subsequences and the smallest covering set :
        :rtype:
        '''
        x=R[0]
        IDs=[]
        max_length=0
        for r in R:
            if len(r)>max_length:
                max_length=len(r)

        #Initialization
        for y in R:
            IDs_k=[]
            for x_j in x:

                found=False
                for i in range(len(y)):
                    if y[i]==x_j:
                        IDs_k.append(i)
                        found=True
                        break

                if not found:
                    IDs_k.append(-max_length*2)
            IDs.append(IDs_k)

        M=np.array(IDs).transpose()
        d=np.where(np.all(M>-1,axis=1),np.std(M, axis=1),-1)
        d_gamma=np.power(gamma,d)

        A_= np.diag(np.where(np.all(M>-1,axis=1),d_gamma,0))

        gaps=np.repeat(M,len(M),axis=0).transpose()-np.tile(M,(len((M)),1)).transpose()
        W=np.where(np.all(np.logical_and(np.abs(gaps)<len(M), gaps>0),axis=0), np.sum(gaps,axis=0)*(1.0/len(R)),-1).reshape(len(M),len(M))

        L=np.where(W>0, np.power(lmbd,W),0)

        sum=np.sum(A_)
        kappa= sum

        kappa_=[sum]
        L_t=L
        while True:
            sum=np.sum(L_t)
            if sum>0:
                kappa_.append(sum)
                kappa=kappa+sum
                L_t=np.dot(L_t,L)
            else:
                break;

        return kappa,kappa_



