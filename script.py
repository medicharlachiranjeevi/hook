import numpy as np
import pickle
import sys
import pandas as pd

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
cl = np.unique(y)
n= len(cl)
X_list=[]
for i in range(n):
    X_list.append(X[i].tolist()+y[i].tolist())
df = pd.DataFrame(X_list)

y=df.groupby(2).mean()
