import numpy as np
from scipy import sparse
import Symbol

class Kernel:
    def kernel(self,x,y):
        pass

class SequenceKernel(Kernel):

    def preceeding_index(self,x):
            L_x=[-1 for i in range(len(x)) ]

            for ind,val in enumerate(x):
                for i in range(ind+1,len(x)):
                    if val==x[i]:
                        L_x[i]=int(ind)
                        break
            return L_x

    def kernel(self,x,y):

        return 0

class VersatileKernel(SequenceKernel):
    '''
      This class implements the algorithms published in
        Cees H. Elzinga, Hui Wang, Versatile string kernels, Theoretical Computer Science, Volume 495, 2013,
        Pages 50-65, ISSN 0304-3975, http://dx.doi.org/10.1016/j.tcs.2013.06.006.
    '''

    def __init__(self,len_range=[],char_weights={},lmb=1,ita=1):

        self.lmb=lmb
        self.ita=ita
        self.M=[]
        self.len_range=len_range
        self.char_weights=char_weights

    def kernel(self,x,y):
        m=len(x)
        n=len(y)

        row=[]
        column=[]
        data=[]
        for i in range(m):
            for j in range(n):
                if x[i].similarTo(y[j]):
                    row.append(i)
                    column.append(j)
                    w=self.ita
                    if self.char_weights.has_key(x[i]):
                        w*=np.power(self.char_weights[x[i]],2)
                    data.append(w)

        self.M.append(sparse.coo_matrix((data,(row,column)),shape=(m,n)))
        T=np.row_stack((row,column))


        longest=1
        T1=T
        while(True):
            longest_temp=longest
            row=[]
            column=[]
            data=[]
            T_T1=[]
            for i in range(len(T[0])):
                for j in range(len(T1[0])):
                    if T[0][i]==T1[0][j] and T[1][i]==T1[1][j]:
                        T_T1.append(j)
                        break

            for i in range(len(T[0])):
                d=0
                for j in range(i+1, len(T[0])):
                    if T[0][i]<T[0][j] and T[1][i]<T[1][j]:
                        if longest_temp==longest:
                            longest_temp+=1
                        gap=(T[0][j]-T[0][i])+(T[1][j]-T[1][i])
                        R=self.M[0].getrow(T[0][i])

                        d+=(self.M[0].data[T_T1[i]]* self.M[longest-1].data[j]* np.power(self.lmb,gap))
                if d!=0:
                    row.append(T[0][i])
                    column.append(T[1][i])
                    data.append(d)

            self.M.append(sparse.coo_matrix((data,(row,column)),shape=(m,n)))

            if longest_temp==longest or len(column)==0:
                break
            longest=longest_temp

            T=np.row_stack((row,column))

        kern=0
        if len(self.len_range)==0:
            for m in self.M:
                kern+=m.sum()
        else:
            for l in self.len_range:
                if l>len(self.M):
                    break
                kern+=m.sum()

        return kern






class EmbeddingsKernel(SequenceKernel):
    '''
    This is an implementation of Theorem 2 in
     Cees Elzinga, Sven Rahmann, Hui Wang. Algorithms for subsequence combinatorics.
     Theoretical Computer Science, 2008, pp 394-404.
    '''

    def kernel(self,x,y):
        D=np.zeros((len(x)+1,len(y)+1))

        for i,xi in enumerate(x):
            for j,yj in enumerate(y):
                if xi==yj:
                    D[i+1][j+1]=D[i][j+1]+D[i+1][j]+1
                else:
                    D[i+1][j+1]=D[i][j+1]+D[i+1][j]-D[i][j]

        return D[len(x),len(y)]



class AllCommonSubsequencesKernel(SequenceKernel):
    '''
    This is an implementation of Lemma 6 in
     Cees Elzinga, Sven Rahmann, Hui Wang. Algorithms for subsequence combinatorics.
     Theoretical Computer Science, 2008, pp 394-404.
    '''


    def kernel(self,x,y):
        D=np.ones((len(x)+1,len(y)+1))
        L_x=self.preceeding_index(x)

        for i,xi in enumerate(x):
            ly=-1
            for j,yj in enumerate(y):
                D[i+1,j+1]=D[i,j+1]

                if xi==yj:
                    ly=j
                if ly!=-1:
                    D[i+1,j+1]=D[i,j+1]+D[i,ly]
                    if L_x[i]!=-1:
                        D[i+1,j+1]-=D[L_x[i],ly]

        return D[len(x),len(y)]

if __name__=="__main__":

    x=[Symbol.SymbolItem(i) for i in ['a', 'b', 'a', 'c']]
    y=[Symbol.SymbolItem(i) for i in ['b', 'a', 'c', 'b']]

    char_weights={Symbol.SymbolItem('a'):np.sqrt(2),Symbol.SymbolItem('c'):np.sqrt(3)}
    k=VersatileKernel(lmb=1, char_weights=char_weights)
    print(k.kernel(x,y))


    char_weights={Symbol.SymbolItem('a'):np.sqrt(1),Symbol.SymbolItem('c'):np.sqrt(1)}

    x=[Symbol.SymbolItem(i) for i in "abacbaca"]
    y=[Symbol.SymbolItem(i) for i in "bacbaacbc"]
    ker=AllCommonSubsequencesKernel()
    print ker.kernel(x, y)
    k=VersatileKernel(lmb=1, char_weights=char_weights)
    print(k.kernel(x,y))
    k=EmbeddingsKernel()
    print(k.kernel(x,y))



    x=[Symbol.SymbolItem(i) for i in "abcbaa"]
    y=[Symbol.SymbolItem(i) for i in "bbaca"]
    ker=AllCommonSubsequencesKernel()
    print ker.kernel(x, y)
    k=VersatileKernel(lmb=1, char_weights=char_weights)
    print(k.kernel(x,y))
    k=EmbeddingsKernel()
    print(k.kernel(x,y))



    x=[Symbol.SymbolItem(i) for i in "dabcdfafcdbbbdaa"]
    y=[Symbol.SymbolItem(i) for i in "dabcdfafcdbbbdaa"]
    ker=AllCommonSubsequencesKernel()
    print ker.kernel(x, y)
    k=VersatileKernel(lmb=1, char_weights=char_weights)
    print(k.kernel(x,y))
    k=EmbeddingsKernel()
    print(k.kernel(x,y))


