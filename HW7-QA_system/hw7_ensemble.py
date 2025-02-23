import os
import pandas as pd
import random
global max_value
def my_filtering_function(pair):
    global max_value
    key, value = pair
    if value < max_value:
        return False  # filter pair out of the dictionary
    else:
        return True  # keep pair in the filtered dictionary
 
 
grades = {'John': 7.8, 'Mary': 9.0, 'Matt': 8.6, 'Michael': 9.5}
 

 

path = "./result/"
dirs = os.listdir( path )
#dirs=['result_luhua_chinese_pretrain_mrc_macbert_large.csv']
dfs=[]
for d in dirs:
    p=path+d
    df=pd.read_csv(p)
    dfs.append(df)
id=[]
anses=[]
#3524
print(dirs)
greatIndex=dirs.index('result_luhua_chinese_pretrain_mrc_macbert_large.csv')
print(dirs[greatIndex])
for i in range(3524):
    temp={}
    for df in dfs:
        try:
            temp[df['Answer'][i]]+=1
        except:
            temp[df['Answer'][i]]=0
    max_value=max(temp.values())
    ans = dict(filter(my_filtering_function, temp.items()))
    keys=list(ans.keys())
    rand=random.randint(0,len(keys)-1)
    if len(keys)!=1 :
        anses.append(dfs[greatIndex]['Answer'][i])
        #anses.append(keys[rand].replace("[CLS]",""))
    else:
        try:
            anses.append(keys[rand].replace("[CLS]",""))
        except:
            anses.append(keys[rand])
    id.append(i)
    #anses.append(keys[rand])

dfNew = pd.DataFrame({'ID': id, 'Answer': anses})
dfNew.to_csv("ensemble.csv",index=False)
print("ensemble done!")
