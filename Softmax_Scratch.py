#!/usr/bin/env python
# coding: utf-8

# In[36]:


from mlxtend.plotting import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import pandas as pd
import scipy
from sklearn import datasets


# # SoftMax Regression

# In instances that we require more than two classes we can use multinomial logistic regression aka SoftMax regression.
# - Unlike logistic regression where y is binary: $y_i\in \{0,1\}$
# - In multinomial logistic regression our target variable y ranges over a number of classes: $y_i\in \{1,2,..,k\}$
#   
# Our goal is to estimate the probability of y being in each potential class (i.e taking on each of the K different values): 
#   
# $$P(y=k|x)$$
# 
# Therefore, given our indepeendent variables x, we want our hypothesis to estimate the probability that $y^{(i)}$ is a member of the class given $x_i$ namely, $P(y=k|x)$ for each value of $k=1,…,K$.  

# ## Generalization of Logistic Regression (Sigmoid Function)
# The multinomial logistic classifier uses a generalization of the sigmoid, called
# the softmax function, to compute $p(y = k|x)$.
# - Sigmoid function: $$\frac{1}{1+e^{-z}}$$
#   
# - Softmax function: $$\frac{e^{z_i}}{\sum_{i=1}^{m}e^{z_j}}$$
#   
# - Both functions take z as an input: $$z = \sum_{i=1}^{m} w_ix_i = {\theta^Tx}$$
#   
#   
# - **SoftMax regression** estimates the probability of y being in **each potential K**, contrastly, Logistic regession uses a threshold function to convert the predicted probability into a binary outcome:
# 
# $$ \hat{y} =
#   \begin{cases}
#     1       & \quad \text{if } \hat{y} \geq 0.5\\
#     0  & \quad \text{if } \hat{y} < 0.5
#   \end{cases}
# $$
# 

# ## Hypothesis
# $$h_\theta (x) = 
#  \begin{pmatrix}
#   P(y = 1|x;\theta) \\
#   P(y = 2|x;\theta) \\
#   \vdots\\
#   P(y = k|x;\theta)
#  \end{pmatrix} = \frac{1}{\sum_{j=1}^{K}exp^{\theta_j^Tx}} \begin{pmatrix}
#   exp^{\theta_1^Tx} \\
#   exp^{\theta_2^Tx} \\
#   \vdots\\
#   exp^{\theta_k^Tx}
#  \end{pmatrix} = \frac{exp^{(\theta_k^Tx)}}{\sum_{j=1}^{K}exp^{(\theta_j^Tx)}}$$
# 
# Notably, the term $\frac{1}{\sum_{i=1}^{m}e^{z_j}}$ normalizes the distribution so that it sums to one. This allows the 
# softmax function to take a vector $z = [z1,z2,...,zk]$ of k arbitrary values and map them into a probability 
# distribution $[0,1]$. 
# 
# We can express the probability that $y^{(i)}$ is equal to each class given $x^{(i)}$ parameterised by $\theta$ as follows:
# 
# $$P(y^{(i)}=k|x^{(i)};\theta) =  \frac{exp^{\theta_k^Tx^{(i)}}}{\sum_{j=1}^{K}exp^{\theta_j^Tx^{(i)}}}$$

# # Cost Function
# 
# The loss function for multinomial logistic regression is a generalization of the loss function for logistic regression from 2 to K classes
# 
# $$LogRegression = J(w)= \sum_{i=1}^{m}-[y^{(i)}log(\hat{y}^{(i)})+(1-y)log(1-\hat{y}^{(i)})]$$
# 
# Cross-Entropy loss: 
# $$L_{CE}(\hat{y},y) = -\sum_{k=1}^{K}y_klog\hat{y}_k = -\sum_{k=1}^{K}y_klog(y=k|x;\theta)$$
# 
# 
# 
# Softmax Regression generalizes the two terms in the above equation. The loss function for a single example x is thus the sum of the logs of the K
# output classes, each weighted by $y_k$
# , the probability of the true class.
# 
# ###### One Hot Encoded Vector
# 
# Notably, only one class is the correct one so the vector y takes the value
# 1 only for this value of k and the rest take the value 0:
# 
# $$OHE = 
#  \begin{pmatrix}
#   0 \\
#   1\\
#   \vdots\\
#   0
#  \end{pmatrix}$$
#  
# For example, if we are predicting cat then only $y_2$ is the correctly labeled instance. This means the terms in the  loss function above will be 0 except for the term corresponding to the true class $y_2$.
# 
# $$L_{CE}(\hat{y},y) = -\sum_{k=1}^{K}y_2log\hat{y}_2 = -\sum_{k=1}^{K}log\hat{y}_2$$
# 
# Where $y_2$ becomes 1 so we can just drop it and $\hat{y} = P(y=k|x;\theta) =  \frac{exp^{\theta_k^Tx}}{\sum_{j=1}^{K}exp^{\theta_j^Tx}}$
# 
# We can therefore write our cost function as the negative log likelihood loss:
# 
# $$L_{CE}(\hat{y},y) =-\sum_{k=1}^{K}\mathbb{1}\{y=k\}log\hat{p}(y=k|x;\theta)$$
# 
# $$J(\theta) =-\sum_{k=1}^{K}\mathbb{1}\{y=k\}log\frac{exp^{\theta_k^Tx}}{\sum_{j=1}^{K}exp^{\theta_j^Tx}}$$
# 
# 
# So when we are dealing with the correctly labeled instance $\mathbb{1}\{y^{(i)}=k\} = 1$.
# 
# 

# In[10]:


# One Hot Encoding
def One_Hot(y):
    Unique,inverse = np.unique(y, return_inverse=True)
    return np.eye(Unique.shape[0])[inverse]
#One_Hot(y)


# # Gradient Descent
# There is no closed form solution to find the minimum of J(θ). Therefore, we will use an iterative optimization algorithm:
# 
# $$\nabla_{\theta^{k}}J(\theta) = - \sum_{i=1}^{m}[x^{(i)}(\mathbb{1}\{y^{(i)}=k\} - P(y^{(i)}=k|x^{(i)};\theta))]$$
# 
# $$\nabla_{\theta^{k}}J(\theta) = - \sum_{i=1}^{m}[x^{(i)}(\mathbb{1}\{y^{(i)}=k\} - \frac{exp^{\theta_k^Tx}}{\sum_{j=1}^{K}exp^{\theta_j^Tx}})]$$
# 

# In[15]:


iris = sns.load_dataset('iris')
df = pd.DataFrame(iris)
X = df.drop('species',axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[55]:


class SoftMaxRegression():
    def __init__ (self,n_epochs=50,learning_rate=0.1,tol= 1e-4,penalty =0):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.penalty = penalty

    
    def fit(self,X,y):
        X = np.c_[np.ones((len(X))),X]
        m = X.shape[0]
        
        #convergence check
        previous_loss = -float('inf')
        self.converged = False

        y_mult = self.One_Hot(y)
        weight = np.zeros([X.shape[1],len(np.unique(y))])
        momentum = weight * 0
        self.loss_history = []
        self.y_mult = y_mult
        
        for epoch in range(self.n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y_mult[random_index:random_index+1]
                scores = np.dot(xi,weight)
                softmax_prob = self.softmax(scores)
                loss = (-1 / m) * np.sum(yi * np.log(softmax_prob)) + (self.penalty/2)*np.sum(np.square(weight))
                self.loss_history.append(loss)

                gradient = (-1/m) * np.dot(xi.T,(yi-softmax_prob))
                gradient[1:] += (self.penalty/m) * weight[1:]
                self.learning_rate = self.learning_schedule(epoch * m + i)
                momentum = 0.8 * momentum + self.learning_rate * gradient
                weight -= momentum

        self.weight = weight
        return self
    
    def predict(self,someX):
        someX = np.c_[np.ones((len(someX))),someX]
        probs = self.softmax(np.dot(someX,self.weight))
        preds = np.argmax(probs,axis=1)
        self.proability = probs
        self.prediction = preds
        return probs,preds
    
    def accuracy(self,Y_test,X_test):
        Y_test =  np.unique(Y_test, return_inverse=True)[1]
        accuracy = np.sum(X_test[1] == Y_test)/(float(len(Y_test)))
        return accuracy
    
    def softmax(self, z):
        z -= np.max(z)
        return (np.exp(z).T/np.sum(np.exp(z),axis=1)).T
    
    def One_Hot(self, y):
        Unique,inverse = np.unique(y, return_inverse=True)
        self.ohe = np.eye(Unique.shape[0])[inverse]
        return self.ohe
    
    def sigmoid(sef,z):
        return 1/(1+np.exp(-z))
    
    def learning_schedule(self,t,t0 =5, t1 = 50):
        return t0 / (t + t1)
        
        
            
            


# In[62]:


softreg = SoftMaxRegression().fit(X_train,y_train)
y_pr = softreg.predict(X_test)
softreg.accuracy(y_test,y_pr)


# In[ ]:




