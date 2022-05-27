# # -*- coding: utf-8 -*-
# """
# Created on Wed Jun 10 18:21:30 2020

# @author: СолеваяШкура
# """
import numpy as np
import math
# from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def gen_image(arr):
     two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
     plt.imshow(two_d, interpolation='nearest')
     return plt

def category(arr,inp):
    result=len(arr)*[0]
    for i in range(len(arr)):
        result[i-1]=28*[0]
    #res=[28]
    for num1 in range(len(arr)):
        for num in range(16):
            if num!=(arr[num1-1]-1):
                result[num1-1][num-1]=0
            else:
                result[num1-1][num-1]=1
       #result[num1]=res
       # res.clear()
    return result


from mlxtend.data import loadlocal_mnist
x_train, y_train = loadlocal_mnist(images_path='C:/mnist/fashion/train-images.idx3-ubyte', labels_path='C:/mnist/fashion/train-labels-idx1-ubyte')
x_val, y_val = loadlocal_mnist(images_path='C:/mnist/fashion/t10k-images-idx3-ubyte', labels_path='C:/mnist/fashion/t10k-labels-idx1-ubyte')
x_train=x_train = x_train.astype(np.float) / 255 - 0.5
x_val=x_val = x_val.astype(np.float) / 255 - 0.5

x_train=x_train.reshape(-1,28*28)
x_val=x_val.reshape(-1,28*28)



y_train=category(y_train,x_train)
y_val=category(y_val,x_val)


x_train=x_train[0:10]
x_val=x_val[0]
y_train=y_train[0:10]
y_val=y_val[0]

learningspeed=0.3
countep=5
startweight=[1,1]
#startweight=[1,2,5...]
par1=0.6
        
class  Net: 
    amountep=0
    speed=0
    weight0=0
    weight1=0
    def __init__(self,n,s):
         self.weight0=0
         self.weight1=0
         self.amountep=n
         self.speed=s
         return
     
    def squesy(self, income,weigh):
        resul=np.array(math.trunc(len(income)/7))
        for i in range(math.trunc(len(income)/7)):
            arr=np.array([income[7*i-7],income[7*i+6]])
            resul[i-1]=((np.sum(arr))*weigh[i-1])/7
        return resul
        

       #### доделать squery  и run 
    def compute(self,income,par):
         #par1=income.reshape(len(income),-1)
         return 1/(1+np.exp(-(par*income)))
 
    def train(self,income,output,startw):
        if (len(startw)<len(output[0])):
            w0=np.array([2*np.random.random()-1]*112)                
        else:
            w0=startw
        w1=np.array([2*np.random.random()-1]*16 )  
        restrain=np.array(self.amountep)
        l = 10 #len(output[0])
        for e in range(self.amountep):                  
            for num in range(l):
                l0=income[num-1]
                l1=self.compute(l0,par1)
                l1=self.squesy(l1,w0)
                l2=self.compute(l1,par1-0.1)
                l2=self.squesy(l2,w1)               
                l2_del= (output[num-1]-l2)*(l2*(1-l2))                            
                w1 += l2_del*self.speed
                for i in range(len(w0))-1:
                    w0[i-1] += (l2_del[i]*self.speed)/2
                    w0[i]+=(l2_del[i]*self.speed)/2
            restrain[e-1]=(np.max(l2_del))
        plt.figure()    
        plt.plot(restrain)
        plt.xlabel('Number try')
        plt.ylabel('trainig err')
        plt.grid()
        plt.show()
        self.weight0=w0
        self.weight1=w1
        return
    
    def run(self,income,output):
        l = 1
        resrun=[0]*self.amountep
        reserr=[0]*self.amountep
        for i in range(l):
            resrun[i-1]=[0]*i
        for e in range(self.amountep):
            for num in range(l):
                if l==1:
                    l0=income
                else:
                    l0=income[num-1]
                l1=self.compute(l0,self.weight0)
                l2=self.compute(l1,self.weight1)
                if l==1:
                    resrun[e-1]=output-l2
                else:
                    resrun[e-1][num-1]=output[num-1]-l2  
            reserr.apend(np.max(resrun*l2*(1-l2)))
        # plt.figure()
        # plt.plot(resrun)
        # plt.xlabel('Number try')
        # plt.ylabel('trainig res')
        # plt.grid()
        # plt.show()
        # plt.figure(3)
        # plt.plot(reserr)
        # plt.xlabel('Number try')
        # plt.ylabel('trainig err')
        # plt.grid()
        # plt.show()
        return  
        
             
OurNetwork=Net(countep,learningspeed)
OurNetwork.train(x_train,y_train,startweight)  
OurNetwork.run(x_val,y_val)
    
      
        
 

  
        
