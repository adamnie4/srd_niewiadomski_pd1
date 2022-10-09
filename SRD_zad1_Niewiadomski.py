#!/usr/bin/env python
# coding: utf-8

# In[176]:


import numpy as np
from sklearn.linear_model import LinearRegression
import statistics
import matplotlib.pyplot as plt


# # Na poczatku robie wykres dla 1 zmiennej w modelu 
# 

# In[177]:


sizes = list(range(10, 210, 10))
#Dla tysiąca powtórzeń bo jak daje 10000 to komputer się strasznie wiesza 
reps = 1000

def sim_r_squared(n):
    x = np.random.normal(size=n)
    y = 1 + x + np.random.normal(size=n)
    
    x = x.reshape(-1,1)
    #Regresja liniowa:
    model = LinearRegression().fit(x,y)
    #Wspl R_sq:
    r_sq = model.score(x,y)
    return r_sq
    


# In[146]:





# In[178]:


#lista = []
#for i in range(len(sizes)):
#    lista.append(0)
    
squared_q95 = []
squared_q5 = []
squared_mean = []


for i in range(0,len(sizes)):
    print(sizes[i])
    result = []
    dodanie =[]
    for j in range(reps):
        dodanie = sim_r_squared(sizes[i])
        result.append(dodanie)
    #print(f"result {result}")
    x=statistics.mean(result)
    y=np.quantile(result,0.95)
    z=np.quantile(result,0.05)

    squared_mean.append(x)
    #print(f"squared mean{squared_mean}")
    
    squared_q95.append(y)
    #print(f"squared_q95 {squared_q95}")
    
    squared_q5.append(z)
    #print(f"quared_q5 {squared_q5}")
    

print(f"squared mean: {squared_mean}")
print(f"squared_q95:  {squared_q95}")
print(f"quared_q5: {squared_q5}")

    


# # PIERWSZY WYKRES

# In[179]:


plt.plot(sizes,squared_mean,'o')
plt.plot(sizes,squared_q5)
plt.plot(sizes,squared_q95)
plt.xlabel("Sample size")
plt.ylabel("R_square")
plt.ylim(min(squared_q5),max(squared_q95))


# # Teraz bede pracowal dla więcej niz 1 zmiennej

# In[274]:


sizes_k = list(range(10, 210, 10))
reps = 1000
k = 9 

def sim_sq_k(n,k):
    x = np.random.normal(0, 1, (n, k))
    z = np.full((n,1),1.0)
    b = np.append(x,z,axis=1)
    y = np.random.normal(size=n)
    #print(b)
    y = y.reshape(-1,1)
    #print(y)
    
    model_k = LinearRegression().fit(x,y)
    r_sq_k = model_k.score(x,y)

    return r_sq_k
    #for i in range(reps):
        


# In[277]:


squared_q95_k = []
squared_q5_k = []
squared_mean_k = []

for i in range(0,len(sizes_k)):
    print(sizes_k[i])
    result_k = []
    dodanie_k =[]
    for j in range(reps):
        dodanie_k = sim_sq_k(sizes_k[i],k)
        result_k.append(dodanie_k)
    #print(f"result {result}")
    x=statistics.mean(result_k)
    y=np.quantile(result_k,0.95)
    z=np.quantile(result_k,0.05)

    squared_mean_k.append(x)
    #print(f"squared mean{squared_mean}")
    
    squared_q95_k.append(y)
    #print(f"squared_q95 {squared_q95}")
    
    squared_q5_k.append(z)
    #print(f"quared_q5 {squared_q5}")
    

print(f"squared mean: {squared_mean_k}")
print(f"squared_q95:  {squared_q95_k}")
print(f"quared_q5: {squared_q5_k}")



# # Wykres drugi

# In[278]:


plt.plot(sizes_k,squared_mean_k,'o')
plt.plot(sizes_k,squared_q5_k)
plt.plot(sizes_k,squared_q95_k)
plt.xlabel("Sample size")
plt.ylabel("R_square")
#plt.ylim(min(squared_q5_k),max(squared_q95_k))

