
# coding: utf-8

# In[2]:

from sklearn.linear_model import LogisticRegression
import util
import numpy as np
import time

# change to your data file path
train_data_file_path = 'data/train.csv'
#test_data_file_path = 'data/test-simple.csv'

train_label, train_data = util.load_weekday_dept_without_grouping(train_data_file_path)
#test_label, test_data = util.load_weekday_dept_without_grouping(test_data_file_path, False)


# In[3]:

depts={}
depts_num = 0
processed_train_data = []
for i in range(0, len(train_data)):
    dept = str(train_data[i][1])
    if dept not in depts:
        depts[dept] = depts_num
        depts_num = depts_num + 1
    processed_train_data.append((train_data[i][0], depts[dept]))
print processed_train_data


# In[57]:

print len(train_label)
print len(processed_train_data)
np.savetxt('temp/train_data.csv', processed_train_data, fmt='%s', delimiter='|')
np.savetxt('temp/train_label.csv', train_label, fmt='%s', delimiter='|')


# In[10]:

# unable to handle categorical data
# do NOT run this block
'''
from sklearn.ensemble import RandomForestClassifier
t1 = time.time()
clf = (n_estimators=10)
clf = clf.fit(train_data, train_label)
t2 = time.time()
print (t2-t1)
'''


# In[5]:

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
get_ipython().magic(u'load_ext rpy2.ipython')
rpy2.robjects.numpy2ri.activate()


# In[ ]:




# In[49]:

'''
train_data_vector = robjects.StrVector(train_data)
train_label_vector = robjects.IntVector(train_label)
robjects.r.assign('r_train_label', train_label_vector)
robjects.r.assign('r_train_data', train_data_vector)
#print robjects.r('r_train_data')
print robjects.r('r_train_label')
'''


# In[6]:

get_ipython().run_cell_magic(u'R', u'', u'r_train_data <- read.csv(file="temp/train_data.csv",head=FALSE,sep="|")\nr_train_label <- read.csv(file="temp/train_label.csv",head=FALSE,sep="|")\nr_train_label <- as.factor(r_train_label[,1])\nr_train_label')


# In[59]:

get_ipython().run_cell_magic(u'R', u'', u'print (dim(r_train_data))\nprint (length(r_train_label))\n#r_train_label')


# In[7]:

get_ipython().run_cell_magic(u'R', u'', u'#http://stackoverflow.com/questions/25715502/using-randomforest-package-in-r-how-to-get-probabilities-from-classification-mo\nlibrary(randomForest)\n#bestmtry <- tuneRF(r_train_data, r_train_label[,1], ntreeTry=100)\n#x <- cbind(r_train_label, r_train_data)\n# Fitting model\nfit <- randomForest(r_train_data, r_train_label, r_train_data, r_train_label, ntree=100)\nprint (fit)\n#Predict Output \n#predicted = predict(fit, type="response", r_train_data)\n#print(predicted)\n#predicted = predict(fit, type="prob", r_train_data)\n#print(predicted)')


# In[8]:

get_ipython().run_cell_magic(u'R', u'', u'#fit$predicted\nlength(fit$predicted)')


# In[9]:

get_ipython().run_cell_magic(u'R', u'', u'write.csv(fit$predicted, file = "temp/fit_predicted_by_R.csv")')


# In[ ]:



