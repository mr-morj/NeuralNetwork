import numpy as np
import random as rd
import seaborn as sb
import matplotlib.pyplot as plt
import math
import copy
I = [[1, -1, -1, -1, 1,
     1, 1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
     
     [1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E
     
     [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     1, 1, 1, -1, 1],  #U
     
     [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, -1, -1, 
     -1, 1, -1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1], #A
     
     [1, -1, -1, -1, -1, 
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1], #L
    
    [-1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, -1, 1,
     1, 1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
    
    [1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, 1, 1, 1, 1,
     -1, -1, 1, -1, -1,
     -1,- 1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, 1, -1, 1,
     1, -1, 1, -1, 1,
     -1, 1, -1, 1, -1],  #W
     
     [-1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1],  #O
     
     [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1],  #R
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, 1]  #K

     ]


I2 = [[1, -1, -1, -1, 1,
     1, 1, 1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
     
     [1, 1, -1, 1, -1,
     1, 1, -1, -1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E
     
     [1, -1, -1, -1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, 1, 1, -1, 1],  #U
     
     [1, 1, 1, -1, -1,
     1, -1, -1, 1, -1,
     1, -1, 1, 1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, -1, -1, 
     -1, 1, -1, 1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, -1,
     1, -1, -1, 1, 1], #A
     
     [1, -1, -1, -1, -1, 
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, 1, -1, -1,
     1, 1, -1, 1, -1], #L
    
    [-1, -1, -1, -1, -1,
     -1, -1, -1, 1, -1,
     -1, -1, -1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, -1, 1,
     1, 1, -1, 1, 1,
     1, -1, 1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1], #N
    
    [1, 1, 1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, -1, 1, -1,
     1, 1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, 1, 1, 1, -1,
     -1, -1, 1, -1, -1,
     -1,- 1, 1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     1, -1, 1, -1, -1,
     1, -1, 1, -1, 1,
     -1, 1, -1, 1, -1],  #W
     
     [-1, 1, 1, 1, -1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1],  #O
     
     [1, 1, 1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, -1, -1, 1,
     1, -1, 1, 1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, 1]  #K

     ]

I4 = [[1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, 1, -1, -1,
     -1, -1, -1, 1, 1,
     1, -1, -1, -1, 1], #N
     
     [1, 1, 1, 1, -1,
     -1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, -1,
     -1, 1, -1, 1, -1],  #E
     
     [-1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, -1, 1],  #U
     
     [1, 1, 1, 1, -1,
     -1, -1, -1, 1, -1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, -1, -1, 
     -1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1], #A
     
     [1, -1, -1, -1, -1, 
     -1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1,
     -1, 1, 1, -1, -1], #L
    
    [-1, -1, -1, -1, -1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, -1, 1,
     -1, 1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, -1, -1, 1, 1,
     -1, -1, -1, -1, 1], #N
    
    [-1, 1, 1, -1, -1,
     -1, -1, -1, -1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, -1, 1, 1, 1,
     -1, -1, -1, -1, -1,
     -1,- 1, -1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, -1, 1, -1, -1,
     -1, -1, 1, -1, 1,
     -1, 1, -1, -1, -1],  #W
     
     [-1, 1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1,-1,
     1, -1, -1, -1, 1,
     -1, 1, 1, -1, -1],  #O
     
     [-1, 1, -1, 1, -1,
     1, -1, -1, 1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1],  #R
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, -1, -1,
     -1, -1, -1, -1, 1]  #K

     ]

I6 = [[1, -1, -1, -1, -1,
     1, -1, -1, 1, 1,
     1, -1, 1, -1, -1,
     -1, -1, -1, 1, 1,
     1, -1, -1, 1, 1], #N
     
     [1, 1, 1, 1, -1,
     -1, -1, -1, 1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, 1, -1,
     -1, 1, -1, 1, -1],  #E
     
     [-1, -1, -1, -1, -1,
     1, -1, -1, 1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     1, 1, 1, 1, 1],  #U
     
     [1, 1, 1, 1, 1,
     -1, -1, -1, 1, 1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, -1, 1],  #R
     
     [-1, -1, 1, 1, -1, 
     -1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1], #A
     
     [1, -1, -1, -1, -1, 
     -1, -1, -1, -1, -1,
     1, -1, -1, -1, 1,
     -1, -1, -1, -1, -1,
     -1, 1, 1, 1, -1], #L
    
    [1, -1, -1, -1, -1,
     -1, 1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, 1, -1,
     -1, -1, -1, -1, -1],  #space
    
    [1, -1, -1, 1, 1,
     -1, 1, -1, -1, 1,
     -1, -1, 1, -1, 1,
     1, -1, -1, 1, 1,
     -1, -1, -1, -1, 1], #N
    
    [-1, 1, 1, -1, -1,
     -1, 1, -1, -1, -1,
     1, 1, 1, -1, -1,
     1, 1, -1, -1, -1,
     1, 1, 1, 1, -1],  #E

     [1, 1, 1, 1, 1,
     -1, -1, -1, -1, -1,
     -1,- 1, -1, -1, -1,
     -1, 1, 1, -1, -1,
     -1, -1, -1, -1, -1],  #T
     
     [1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, 1, 1, -1, 1,
     -1, -1, 1, -1, 1,
     -1, 1, -1, -1, -1],  #W
     
     [-1, 1, -1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1,-1,
     1, -1, -1, -1, 1,
     -1, 1, 1, -1, 1],  #O
     
     [-1, 1, -1, 1, 1,
     1, -1, -1, 1, -1,
     1, -1, 1, -1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1],  #R
     
     [1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, 1, -1,
     -1, -1, 1, -1, 1]  #K

     ]

Coord = [[0, 0], [0,1], [0,2], [0,3],
         [1, 0], [1,1], [1,2], [1,3],
         [2, 0], [2,1], [2,2], [2,3],
         [3, 0], [3,1], [3,2], [3,3]]
W = np.random.uniform(0,1, (16, 25))
print(W[0])
nu0 = 0.1
t2 = 1000
sg = 2

def noise(I = []):
    print(I[0])
    I2=copy.deepcopy(I)
    for i in range(len(I)):
        for j in range(3):
            r = rd.randint(0,len(I[0])-1)
            I2[i][r]*=-1
    #print(I[0])
    return I2

def t1(): 
    return 1000/(math.log(sg))
def nu(n):
    return nu0*math.exp(-n/t2)

def sgn(n):
    return sg*math.exp(-n/t1())

def h(n,i,j) :
   
    return math.exp(-math.pow(-np.linalg.norm(np.array(Coord[i])-np.array(Coord[j])),2)/(2*math.pow(sgn(n),2)))

def test(I=[],W=[]):
        T = []
        for q in range(len(I)):
            x = I[q]
            n=15
            min = np.linalg.norm(I[q]-W[15])
            for i in range(15):
                j = np.linalg.norm(I[q]-W[i])
                if j < min:
                    min = j
                    n = i
            T.append(n)
        return T



for k in range(1000):
    x = rd.randint(0, 11)
    min = np.linalg.norm(I[x]-W[15])
    n=15
    for i in range(15):
        j = np.linalg.norm(I[x]-W[i])
        if j < min:
            min = j;
            n = i
    for z in range(len(W)):
            W[z] = W[z] + nu(k)*h(k,n,z)*(I[x]-W[z])



for k in range(8000):
    x = rd.randint(0, 11)
    min = np.linalg.norm(I[x]-W[11])
    n=15
    for i in range(15):
        j = np.linalg.norm(I[x]-W[i])
        if j < min:
            min = j;
            n = i
    for z in range(len(W)):
            W[z] = W[z] + 0.1*h(k,n,z)*(I[x]-W[z])

for i in range(16):
    print(W[i])



Test=test(I,W)
Test2=test(I2,W)
Test4=test(I4,W)
Test6=test(I6,W)

er2 = []
er4 = []
er6 = []

    
for ii in range(len(I)):
    ht = []
    ht2 =[]
    ht4 =[]
    ht6 =[]
    x=I[ii]
    y=I2[ii]
    z=I4[ii]
    v=I6[ii]
    
    num = 0
    n = Test[ii]
    i=j=0
    for i in range(4):
        ht.append([])
        for j in range(4):
            ht[i].append(h(k,n,num))
            num+=1
    num = 0
    n = Test2[ii]
    n1 = Test4[ii]
    n2 = Test6[ii]
    i=j=0
    for i in range(4):
        ht2.append([])
        ht4.append([])
        ht6.append([])
        for j in range(4):
            ht2[i].append(h(k,n,num))
            ht4[i].append(h(k,n1,num))
            ht6[i].append(h(k,n2,num))
            num+=1

    f, ax=plt.subplots(4, 2, figsize = (16,16))
    sb.heatmap(np.reshape(I[ii],(5,5)), cmap = sb.cm.rocket_r, ax=ax[0][0], cbar=False, yticklabels=False, xticklabels=False )
    sb.heatmap(ht, ax = ax[0][1], cmap="YlGnBu", linewidths=0.1, linecolor = 'black', cbar=False, xticklabels=False, yticklabels=False )
    sb.heatmap(np.reshape(I2[ii],(5,5)), cmap = sb.cm.rocket_r, ax=ax[1][0], cbar=False, yticklabels=False, xticklabels=False  )
    sb.heatmap(ht2, ax = ax[1][1], cmap="YlGnBu", linewidths=0.1, linecolor = 'black', cbar=False, xticklabels=False, yticklabels=False )
    sb.heatmap(np.reshape(I4[ii],(5,5)), cmap = sb.cm.rocket_r, ax=ax[2][0], cbar=False, yticklabels=False, xticklabels=False  )
    sb.heatmap(ht4, ax = ax[2][1], cmap="YlGnBu", linewidths=0.1, linecolor = 'black', cbar=False, xticklabels=False, yticklabels=False )
    sb.heatmap(np.reshape(I6[ii],(5,5)), cmap = sb.cm.rocket_r, ax=ax[3][0], cbar=False, yticklabels=False, xticklabels=False  )
    sb.heatmap(ht6, ax = ax[3][1], cmap="YlGnBu", linewidths=0.1, linecolor = 'black', cbar=False, yticklabels=False )
    
    plt.show();
    
    if (ht==ht2): 
        er2.append(1) 
    else: er2.append(0)
    if (ht==ht4): 
        er4.append(1) 
    else: er4.append(0)
    if (ht==ht6): 
        er6.append(1) 
    else: er6.append(0)
words = ['N','E', 'U', 'R', 'A', 'L', '_', 'n', 'e', 't', 'w', 'o', 'r', 'k']    
print(f'2 Changes | {er2} | {er2.count(1)} symbols is correct | {er2.count(0)} symbols is false | Accuracy is {er2.count(1)/14}') 
print(f'4 Changes | {er4} | {er4.count(1)} symbols is correct | {er4.count(0)} symbols is false | Accuracy is {er4.count(1)/14}')
print(f'6 Changes | {er6} | {er6.count(1)} symbols is correct | {er6.count(0)} symbols is false | Accuracy is {er6.count(1)/14}')
plt.plot(words, er2, label='2 changes', linestyle=':', marker='1')
plt.plot(words, er4, label='4 changes', linestyle=':', marker='2')
plt.plot(words, er6, label='6 changes', linestyle=':', marker='3')
plt.title('Result of changes pixels (1 if correct, 0 if false)')
plt.legend()
plt.show()
        
        
    
