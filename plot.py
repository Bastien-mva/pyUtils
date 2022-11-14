import matplotlib.pyplot as plt 
import numpy as np 


def plotList(myList,label, ax = None):
    if ax == None:
        ax = plt.gca()
    ax.plot(np.arange(len(myList)), myList, label = label)

def lissage(Lx,Ly,p):
    '''Fonction qui débruite une courbe par une moyenne glissante
    sur 2P+1 points'''
    Lxout=[]
    Lyout=[]
    for i in range(p,len(Lx)-p):   
        Lxout.append(Lx[i])
    for i in range(p,len(Ly)-p):
        val=0
        for k in range(2*p):
            val+=Ly[i-p+k]
        Lyout.append(val/2/p)
            
    return Lxout,Lyout
