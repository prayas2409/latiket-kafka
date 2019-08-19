#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns; sns.set(style='white')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.metrics import precision_score, accuracy_score


# In[4]:


# data = pd.read_csv('/home/administrator123/Akanksha/Company_Work/datasetcpulogs/Data/CpuLogData.csv')
data = pd.read_csv('https://raw.githubusercontent.com/prayas2409/logs/master/CpuLogData.csv')
data['Dates'] = pd.to_datetime(data['DateTime']).dt.date
data['Time'] = pd.to_datetime(data['DateTime']).dt.time
data.shape


# In[5]:


u_name = ['mohitkr1301@gmail.com', 'sapnapatil344@gmail.com', 'akankshakaple@gmail.com'
         ,'you@example.com']
for i in u_name:
    data.drop(data[data['user_name'] == i].index, inplace=True)
data.shape


# In[9]:


data['user_name'].unique()
# data.columns


# In[10]:


feature_eng_col = ['Cpu Working Time', 'Cpu idle Time', 'number of software interrupts since boot'
                   ,'number of interrupts since boot', 'disk_read_count', 'disk_write_count'
                   ,'disk_read_bytes', 'disk_write_bytes', 'time spent reading from disk'
                   ,'time spent writing to disk', 'time spent doing actual I/Os'
                   ,'number of bytes sent', 'number of bytes received'
                   ,'number of packets sent', 'number of packets recived']

# To be checked again

# Remove constant data
constant_col = ['Cpu Count', 'Usage Cpu Count ', 'number of system calls since boot'
                , 'system_total_memory', 'total number of errors while receiving'
                ,'total number of errors while sending','total number of incoming packets which were dropped'
                ,'total number of outgoing packets which were dropped', 'disk_total_memory']

# Remove Inter dependent data
inter_dependet_cols = ['system_free_memory', 'disk_free_memory', 'system_avalible_memory', 'system_used_memory']

# Remove object data 
object_col = ['DateTime', 'Dates', 'Time', 'boot_time']


# In[11]:


def remove_col(data,col_list):
    data.drop(columns=col_list, inplace=True)
    return data
data = remove_col(data, inter_dependet_cols)
# data.columns
data = remove_col(data, constant_col)
data.shape


# In[12]:


data.columns


# In[13]:


# data['user_name'].unique()
data[data['user_name'] == 'honeykrsingh16@gmail.com']['Cpu idle Time'].iloc[270:]


# In[14]:


def visualization(data):

    dt = data[data['user_name']=='5152ibrahim@gmail.com']
    dt2 = dt[dt['Dates']==dt['Dates'].unique()[2]]
    
    for i in range(1,len(dt2.columns)):
        if data[dt2.columns[i]].dtype != 'object':
            print(dt2.columns[i])
#             plt.figure(figsize=(15,7))
            sb.lineplot(x='Time', y=dt2.columns[i], data=dt2)
            plt.xticks(rotation=90)
            plt.show()
# visualization(data)


# In[16]:


# # def feature_engineering(features):
# user_name = data['user_name'].unique()
# final_df = pd.DataFrame()
# # print(user_name)
# for u_name in user_name:
#     df = data[data['user_name'] == u_name]
#     df.sort_values('DateTime', inplace=True)
#     df.reset_index(drop=True, inplace=True)
# #     print(df.shape)
#     print(u_name)
#     for col in feature_eng_col:
#         l1 =[]
#         l2 = []
#         for index in range(1,len(df)):
            
#             if (df["Dates"].iloc[index] == df["Dates"].iloc[index-1]):
#                 if (df[col].iloc[index]-df[col].iloc[index-1]) < 0:
#                     l2.append(np.average(l1))
#                     l2.extend(l1)
#                     l1=[]
#                 elif index == len(df)-1:
#                     if (df[col].iloc[index]-df[col].iloc[index-1]) < 0:
#                         l2.append(np.average(l1))
#                         l2.extend(l1)
#                         l1=[]
#                     else:
# #                         l1.append(df[col].iloc[index]-df[col].iloc[index-1])
#                         l2.append(np.average(l1))
#                         l2.extend(l1)
#                         l2.append(df[col].iloc[index])
#                 else :
#                     l1.append(df[col].iloc[index]-df[col].iloc[index-1])
                
#             else:
#                 l2.append(np.average(l1))
#                 l2.extend(l1)
#                 l1=[]
# #             print(index)
# #         print(col)
#         df[col+"_fe"]=l2
#     print(u_name, len(l2), df.shape)
#     final_df = final_df.append(df)
# print(final_df.shape)


# In[17]:


# def feature_engineering(features): by Prayas
user_name = data['user_name'].unique()
final_df = pd.DataFrame()
# print(user_name)
for u_name in user_name:
    df = data[data['user_name'] == u_name]
    df.sort_values('DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
#     print(df.shape)
    print(u_name)
    for col in feature_eng_col:
        l1 =[]
        l1.append(df[col].iloc[0])
        for index in range(1,len(df)):
            
            if (df["Dates"].iloc[index] == df["Dates"].iloc[index-1]):
                # As when curr smaller than prev
                if (df[col].iloc[index]-df[col].iloc[index-1]) <= 0:
                    l1.append(df[col].iloc[index])
                    # curr > prev
                elif (df[col].iloc[index]-df[col].iloc[index-1]) > 0:
                    l1.append(df[col].iloc[index]-df[col].iloc[index-1])
#                 # commenting below as above will take it competely    
#                 elif index == len(df)-1:
#                     if (df[col].iloc[index]-df[col].iloc[index-1]) < 0:
#                         l2.append(np.average(l1))
#                         l2.extend(l1)
#                         l1=[]
#                     else:
# #                         l1.append(df[col].iloc[index]-df[col].iloc[index-1])
#                         l2.append(np.average(l1))
#                         l2.extend(l1)
#                         l2.append(df[col].iloc[index])
#                 else :
#                     l1.append(df[col].iloc[index]-df[col].iloc[index-1])
                
            else:
                l1.append(df[col].iloc[index])
#             print(index)
#         print(col)
        df[col+"_fe"]=l1
    print(u_name, len(l2), df.shape)
    final_df = final_df.append(df)
print(final_df.shape)


# In[18]:


final_df.drop(columns=feature_eng_col, inplace=True)
final_df.shape


# In[ ]:


# final_df.columns


# In[ ]:


# for col in final_df.columns:
#     if final_df[col].dtypes != 'object':
#         print(col, min(final_df[col]))


# In[ ]:


# take = ['Cpu Working Time_fe', 'cpu_percent', 'cpu avg load over 1 min',
#        'cpu avg load over 5 min', 'cpu avg load over 15 min',  'disk_read_count_fe',
#        'disk_write_count_fe','time spent reading from disk_fe', 'time spent writing to disk_fe',
#        'time spent doing actual I/Os_fe', 'Dates', 'user_name', 'Time']


# for col in final_df.columns:
#     if col not in take:
#         final_df.drop(columns=col, inplace=True)


# In[ ]:


final_data = final_df
final_data.shape


# In[ ]:


# print(final_data['DateTime'].dtype)
# final_df.dtypes
final_df.shape


# In[19]:


from sklearn.preprocessing import StandardScaler 
def std(data):
    # standardize the data attributes
    for col in data.columns:
        if data[col].dtype == 'object':
            data = data.drop(columns=col)
    sc = StandardScaler()
    standardized_X = sc.fit_transform(data)
    new_data = pd.DataFrame(standardized_X, columns=data.columns)
    
    return new_data
new_data = std(final_df)


# In[20]:


final_df.shape, new_data.shape


# In[21]:


new_data['Dates'] = list(final_df['Dates'])
new_data['user_name'] = list(final_df['user_name'])
new_data['Time'] = list(final_df['Time'])


# In[27]:


def visualization(data):
    dt = data[data['user_name']=='kiranraikar777@gmail.com']
    dt2 = dt[dt['Dates']==dt['Dates'].unique()[2]]
    
    for i in range(1,len(dt2.columns)):
        if final_df[dt2.columns[i]].dtype != 'object':
            print(dt2.columns[i])
#             plt.figure(figsize=(15,7))
            sb.lineplot(x='Time', y=dt2.columns[i], data=dt2)
            plt.xticks(rotation=90)
            plt.show()
        
visualization(new_data)


# In[25]:


for col in final_df.columns:
    if final_df[col].dtypes == 'object':
        final_df.drop(columns=col, inplace=True)


# In[28]:


final_data.shape


# In[29]:


final_df.duplicated().sum()


# In[30]:


# Remove the outlier from the whole dataset
def remove_outlier(df):
    low = .20
    high = .80
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

df= remove_outlier(final_df)
# sb.boxplot(df)
plt.figure(figsize=(15,7))
# plt.xticks(rotation='vertical')
s = df.boxplot()
s.set_xticklabels(df.columns,rotation=90)


# In[ ]:


df.isna().sum()


# In[31]:


from sklearn.preprocessing import StandardScaler 
def std(data):
    # standardize the data attributes
    for col in data.columns:
        if data[col].dtype == 'object':
            data = data.drop(columns=col)
    sc = StandardScaler()
    standardized_X = sc.fit_transform(data)
    new_data = pd.DataFrame(standardized_X, columns=data.columns)
    
    return new_data
new_data = std(df)


# In[32]:


from sklearn.cluster import KMeans
wcss = []

X = np.array(new_data)

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X.reshape(-1,1))
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(new_data)
new_data['y'] = y_kmeans

sb.countplot(x=y_kmeans, data=new_data)


# In[ ]:


# # sb.boxplot(final_df)
# from pandas.api.types import is_numeric_dtype
# def remove_outlier(df, name):
#     low = .25
#     high = .75
#     quant_df = df.quantile([low, high])
# #     for name in list(df.columns):
# #         if is_numeric_dtype(df[name]):
#     for i in range(len(df)):
#         if df[name].iloc[i]  > quant_df.loc[low, name] or df[name].iloc[i] < quant_df.loc[high, name]:
# #             print(df[name].iloc[i])
#             df[name].iloc[i] = np.average(df[name])
# #             print(df[name].iloc[i])
#     return df
# l = ['system_active_memory','disk_used_memory','system_inactive_memory','system_buffers_memory',
#        'system_cached_memory','system_shared_memory']
# # for i in l:
# #     print(i)
# df= remove_outlier(new_data,'system_active_memory')
# #     sb.boxplot(df)
# plt.figure(figsize=(15,7))
# plt.xticks(rotation='vertical')
# s = df.boxplot()
# s.set_xticklabels(df.columns,rotation=90)


# In[ ]:


# low = .20
# high = .80
# name = 'system_active_memory'
# quant_df = df.quantile([low, high])
# print(quant_df.loc[low, name], quant_df.loc[high, name])
# np.average(new_data[name])


# In[ ]:


# # Remove the outlier from the whole dataset
# def remove_outlier(df):
#     low = .25
#     high = .75
#     quant_df = df.quantile([low, high])
#     for name in list(df.columns):
#         if is_numeric_dtype(df[name]):
#             df[name] = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
#     return df

# df= remove_outlier(new_data)
# plt.figure(figsize=(15,7))
# plt.xticks(rotation='vertical')
# s = df.boxplot()
# s.set_xticklabels(df.columns,rotation=90)


# In[ ]:





# In[ ]:


# df['system_active_memory'], final_df['system_active_memory']
# for col in new_data.columns:
#     if new_data[col].isna():
#         new_data.drop(columns=col, inplace=True)


# In[ ]:


# np.average(new_data['Cpu Working Time_fe'])


# In[ ]:


# from sklearn.cluster import KMeans
# wcss = []

# X = np.array(new_data)

# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X.reshape(-1,1))
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
# y_kmeans = kmeans.fit_predict(new_data)
# new_data['y'] = y_kmeans

# sb.countplot(x=y_kmeans, data=new_data)


# In[33]:


plt.figure(figsize=(10,10))
cor = new_data.corr() #Calculate the correlation of the above variables
sb.heatmap(cor, square = True) #Plot the correlation as heat map


# In[ ]:


new_data[['cpu_percent','y']].head(10)


# In[ ]:


# df_cluster = final_df
# df_cluster.shape, final_df.shape
# d = final_df


# In[ ]:


# kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
# y_kmeans = kmeans.fit_predict(final_df)
# final_df['y'] = y_kmeans

# sb.countplot(x=y_kmeans, data=final_df)


# In[ ]:


# plt.figure(figsize=(10,10))
# sb.heatmap(final_df.corr())


# In[ ]:


# plt.figure(figsize=(15,8))
# sb.boxplot(data=final_df)
# plt.xticks(rotation=90)


# In[ ]:


#import important libraries.
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pandas.api.types import is_numeric_dtype
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score
# def remove_outlier(df):
#     low = .25
#     high = .75
#     quant_df = df.quantile([low, high])
#     for name in list(df.columns):
#         if is_numeric_dtype(df[name]):
#             df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
#     return df

# df= remove_outlier(final_df)
# sns.boxplot(df)


# In[ ]:


# for col in final_df.columns:
# #     plt.hist(x=col)
#     sb.distplot(final_df[col])
#     plt.xticks(rotation=90)
#     plt.show()


# In[ ]:


# from sklearn.preprocessing import StandardScaler

# # No Feature Scaling as library does it
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X = sc_X.fit_transform(final_df)


# In[ ]:


# from sklearn.cluster import KMeans
# wcss = []
# X = np.array(final_df)

# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
# y_kmeans = kmeans.fit_predict(final_df)
# final_df['y'] = y_kmeans

# sb.countplot(x=y_kmeans, data=final_df)

