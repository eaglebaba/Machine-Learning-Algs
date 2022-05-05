#!/usr/bin/env python
# coding: utf-8

# In[18]:


#Question 1
import matplotlib.pyplot as plt
import numpy as np


# In[79]:


X = np.matrix([[1, 0], 
               [1, 3],
              [1, 6]
              ])

XT = np.matrix.transpose(X)


y = np.matrix([[1], 
               [4],
              [5]]
              )

XT_X = np.matmul(XT, X)


XT_y = np.matmul(XT, y)


betas = np.matmul(np.linalg.inv(XT_X), XT_y)
plt.plot(np.matmul(X, betas))


# In[80]:


x = np.array([0,3,6])
y = np.array([1,4,5])
plt.scatter(x,y)
plt.plot(x,y,'r-')


# In[38]:


x = np.array([0,3,6])
y = np.array([1,4,5])
k, d = np.polyfit(x, y, 1)
y_pred = k*x + d

plt.plot(x, y, '.')
plt.plot(x, y_pred)


# In[48]:


corr_matrix= np.corrcoef(x,y)
corr = corr_matrix[0,1]
R_sq = corr**2
 
print(R_sq)


# In[49]:


residuals = y-y_pred


# In[50]:


residuals


# In[54]:


plt.scatter(residuals,y_pred)
plt.title("Residual vs Fitted Graph")
plt.show()


# In[83]:


X = np.matrix([[1, 0, 0], 
               [1, 3, 9],
              [1, 6, 36]
              ])
XT = np.matrix.transpose(X)
y = np.matrix([[1], 
               [4],
              [5]]
              )
XT_X = np.matmul(XT, X)
XT_y = np.matmul(XT, y)
betas = np.matmul(np.linalg.inv(XT_X), XT_y)
plt.plot(X*betas)


# In[62]:


#Question 2
x = np.array([0,3,6])
y = np.array([1,4,5])
model = np.poly1d(np.polyfit(x, y, 2))
polyline = np.linspace(1, 10, 50)
plt.scatter(x,y)
plt.plot(polyline, model(polyline))
plt.show()


# In[64]:


print(model)


# In[ ]:


data = pd.read_excel()


# In[84]:


cd ..


# In[88]:


cd Michael Adeyeye


# In[89]:


cd Desktop


# In[90]:


ls


# In[93]:


#Question 3
pip install openpyxl


# In[136]:


data = pd.read_excel("Conventry_data.xlsx")


# In[137]:


data.head()


# In[139]:


dates = data['Date_in text'].to_numpy()


# In[140]:


day_len = data['Day Lenght in Mins'].to_numpy()


# In[150]:


plt.scatter(dates, day_len)
plt.plot(dates, day_len,'b--')


# In[161]:


polyline = np.linspace(1, 50000, 1000)
model = np.poly1d(np.polyfit(dates, day_len, 2))
plt.plot(polyline, model(polyline))
plt.show()


# In[162]:


print(model)


# In[163]:


polyline = np.linspace(1, 50000, 1000)
model2 = np.poly1d(np.polyfit(dates, day_len, 3))
plt.plot(polyline, model2(polyline))
plt.show()


# In[164]:


print(model2)


# In[165]:


polyline = np.linspace(1, 50000, 1000)
model3 = np.poly1d(np.polyfit(dates, day_len, 4))
plt.plot(polyline, model3(polyline))
plt.show()


# In[166]:


print(model3)


# In[169]:


#define function to calculate r-squared
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = ssreg / sstot

    return results

#find r-squared of polynomial model with degree = 2
polyfit(dates, day_len, 2)


# In[170]:


#find r-squared of polynomial model with degree = 3
polyfit(dates, day_len, 3)


# In[171]:


#find r-squared of polynomial model with degree = 4
polyfit(dates, day_len, 4)


# In[173]:


#Plotting the models on the same axis
plt.plot(polyline, model(polyline))
plt.plot(polyline, model2(polyline))
plt.plot(polyline, model3(polyline))


# In[ ]:


#The model can be improved if the sklearn library in python is used to build the model instead of the polyfit method used.

