import numpy as np

import Symbol

class Concordance:
    def __init__(self):
        pass

    def calculate_concordance(self,X):

        return 0

class ConcordanceAndSmallestCoveringSet(Concordance):
    '''
    This class is related to the paper of "Concordance and the Smallest Covering Set of Preference Orderings"
    by Dr. Zhiwe Lin  at Ulster University
       Prof. Hui Wang at Ulster University
       Prof. Cees H. Elzinga at VU University Amsterdam and NIDI
       
    The contribution of the work includes:
        1) to measure concordance of preference orderings
        2) to extract meaningful descriptive patterns from preference orderings
	 Paper available at http://arxiv.org/pdf/1609.04722v1.pdf

     If you have any questions or want to report bugs,
     please contact Dr. Zhiwei Lin at http://scm.ulster.ac.uk/zhiwei.lin/
    '''

    def Upsilon(self, x, T_matrix,varPhi,u, i,omega):
        '''
        Equation (38)

        :param x:
        :type x:
        :param T_matrix:
        :type T_matrix:
        :param varPhi:
        :type varPhi:
        :param u:
        :type u:
        :param i:
        :type i:
        :param omega:
        :type omega:
        :return:
        :rtype:
        '''
        if omega[i]==0:
            return []
        Q=[j for j in range(i+1,len(x)) if  T_matrix[j][i]==1]

        if len(Q)==0:
            return [u]
        S=[]
        for j in Q:
            v=list(u)
            v.append(x[j])
            S.extend(self.Upsilon(x,T_matrix,varPhi,v,j, omega))

        return S

    def check_occurrence(self,x,T_matrix,LCSs):
            no_occurrences=[]
            for (i,sym) in enumerate(x):
                if T_matrix[i][i]==0:
                    continue
                occurrence=False
                for lcs in LCSs:
                    if sym in lcs:
                        occurrence=True
                        break
                if not occurrence:
                    no_occurrences.append(i)
            return no_occurrences


    def extractSmallestCoveringSet(self,x, T_matrix,varPhi, LCSs,omega):
        '''
        Extract the smallest covering set from X
        :param x:
        :type x:
        :param T_matrix:
        :type T_matrix:
        :param varPhi:
        :type varPhi:
        :param LCSs:
        :type LCSs:
        :param omega:
        :type omega:
        :return:
        :rtype:
        '''
        A=list(LCSs)

        no_occurrences=self.check_occurrence(x,T_matrix,A)

        while len(no_occurrences)!=0:
            i=no_occurrences[0]

            if omega[i]==0:
                continue

            C=self.Theta_function(x,T_matrix,varPhi,[x[i]],i,omega)
            for c in C:
                B=self.Upsilon(x,T_matrix,varPhi, c, i, omega)

            omega[i]=0

            if len(B)>0:
                A.extend(B)
            no_occurrences=self.check_occurrence(x,T_matrix,A)

        return A

    def Theta_function(self, x,T_matrix, varPhi, u, i,omega):
        '''
        Equation (34)
        :param x:
        :type x:
        :param T_matrix:
        :type T_matrix:
        :param varPhi:
        :type varPhi:
        :param u:
        :type u:
        :param i:
        :type i:
        :param omega:
        :type omega:
        :return:
        :rtype:
        '''

        if omega[i]==0:
            return []

        P=[j for j in range(i) if varPhi[j]+1==varPhi[i] and T_matrix[i][j]==1 ]

        if len(P)==0:
            return [u]
        S=[]
        for j in P:
            v=[x[j]]
            v.extend(u)
            S.extend(self.Theta_function(x,T_matrix,varPhi, v,j,omega))

        return S

    def extract_all_lcs(self, x, T_matrix, varPhi, omega, llcs):
        '''
        To return a set of  all longest common subsequences when
        calculate_concordance is called.
        '''

        R=[i for i in range(len(varPhi)) if varPhi[i]==llcs]
        LCSs=[]

        for i in R:
            LCSs.extend(self.Theta_function(x,T_matrix,varPhi,[x[i]],i,omega))

        return LCSs


    def calculate_concordance(self,X):
        '''
        Given a set X of ordering sequences, this will calculate the concordance  $\kappa(X)$ of X
        shown in the paper.

        :param X:
        :type X:
        :return $\kappa(X)$, $\ell(X)$, the set of the longest common subsequences and the smallest covering set :
        :rtype:
        '''
        x=X[0]
        I=[]

        #Initialization
        for y in X:
            I_k=[]
            for x_j in x:

                found=False
                for i in range(len(y)):
                    if y[i]==x_j:
                        I_k.append(i)
                        found=True
                        break

                if not found:
                    I_k.append(float('inf'))
            I.append(I_k)

        T_matrix=np.zeros((len(x),len(x)))

        for j in range(len(x)):
            for i in range(j+1):
                T_matrix[j][i]=1
                for k in range(len(X)):
                    if I[k][i]> I[k][j] or I[k][j]==float('inf'):
                        T_matrix[j][i]=0
                        break;
        psi=np.zeros(len(x)+1)
        varPhi=np.zeros(len(x))

        omega=[1 for i in x]
        psi[0]=1
        for j in range(len(x)):
            length=0
            psi[j+1]+=T_matrix[j][j]

            for i in range(0,j):
                psi[j+1]+=psi[(i+1)%(j+1)]*T_matrix[j][i]
                length=np.maximum(length,varPhi[i]*T_matrix[j][i])
            varPhi[j]=(length+1)*T_matrix[j][j]

        concordance=np.sum(psi)
        llcs=np.max(varPhi)
        LCSs=self.extract_all_lcs(x, T_matrix, varPhi, omega,llcs)
        SCS=self.extractSmallestCoveringSet(x, T_matrix,varPhi, LCSs,omega)

        return concordance,llcs,LCSs,SCS


if __name__=="__main__":
    def print_results(X,con_results):
        print("================================================")
        print("********************************")
        print("The Set X of the Ordering Sequences:")
        for x in X:
            print (x)
        print("********************************")
        print("k(X)= {}   --The concordance of X".format(con_results[0]))
        print("l(x)= {}   --The length of the longest common subsequences".format(con_results[1]))
        print("All LCSs: {} ".format(con_results[2]))
        print("The  Smallest Covering Set C(X): {} ".format(con_results[3]))
        print("================================================\n\n")

    X=[]
    X.append([Symbol.SymbolItem(i) for i in ['a', 'b', 'c', 'd','e', 'f']])
    X.append([Symbol.SymbolItem(i) for i in [ 'a', 'c', 'f', 'b','d','e']])
    X.append([Symbol.SymbolItem(i) for i in ['a',  'b', 'd','c', 'f', 'e']])
    con=ConcordanceAndSmallestCoveringSet()
    print_results(X,con.calculate_concordance(X))

    X=[]
    X.append([Symbol.SymbolItem(i) for i in ['a', 'b', 'c', 'd','f', 'e']])
    X.append([Symbol.SymbolItem(i) for i in [ 'a', 'd', 'b', 'c','f','e']])
    X.append([Symbol.SymbolItem(i) for i in ['a',  'd', 'f','b', 'e', 'c']])
    con=ConcordanceAndSmallestCoveringSet()
    print_results(X,con.calculate_concordance(X))


    X=[]
    X.append([Symbol.SymbolItem(i) for i in ['f', 'a', 'b', 'c', 'd', 'e']])
    X.append([Symbol.SymbolItem(i) for i in ['e', 'a', 'f', 'd', 'b', 'c']])
    X.append([Symbol.SymbolItem(i) for i in ['a', 'f', 'e', 'd', 'b', 'c']])
    con=ConcordanceAndSmallestCoveringSet()
    print_results(X,con.calculate_concordance(X))



    X=[]
    X.append([Symbol.SymbolItem(i) for i in ['a', 'b', 'c', 'd', 'e']])
    X.append([Symbol.SymbolItem(i) for i in ['a', 'b', 'd', 'c', 'e']])
    X.append([Symbol.SymbolItem(i) for i in ['b', 'd', 'c', 'e']])
    con=ConcordanceAndSmallestCoveringSet()
    print_results(X, con.calculate_concordance(X))


    X=[]
    X.append([Symbol.SymbolItem(i) for i in ['f', 'a', 'b', 'c', 'd', 'e']])
    X.append([Symbol.SymbolItem(i) for i in ['f', 'a', 'b', 'c', 'd', 'e']])
    X.append([Symbol.SymbolItem(i) for i in ['f', 'a', 'b', 'c', 'd', 'e']])
    con=ConcordanceAndSmallestCoveringSet()
    print_results(X, con.calculate_concordance(X))

    X=[]
    X.append([Symbol.SymbolItem(i) for i in ['a', 'b', 'c', 'd', 'e']])
    X.append([Symbol.SymbolItem(i) for i in ['e', 'a', 'd', 'b', 'c']])
    X.append([Symbol.SymbolItem(i) for i in ['a', 'e', 'd', 'b', 'c']])
    con=ConcordanceAndSmallestCoveringSet()
    print_results(X,con.calculate_concordance(X))


