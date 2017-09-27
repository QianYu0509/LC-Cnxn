import os
os.chdir('/Users/Desktop')
import scipy as sp  
import pandas as pd
import numpy as np
from sklearn.datasets import load_files  
from sklearn.cross_validation import train_test_split  
from sklearn.feature_extraction.text import  TfidfVectorizer  

##combine loan data from 2007 to 2015
dataframes = []
for i in ['a','b','c','d']:
    current_df = pd.read_csv("LoanStats3{:}_securev1.csv".format(i), encoding = "ISO-8859-1")
    dataframes.append(current_df)

combined_data = pd.concat(dataframes)
combined_data.to_csv('output.csv')

all_the_data = pd.read_csv('output.csv')

# get the useful data
# first, drop the row that has more than 10 Na values
useful_data = all_the_data.dropna(axis=0, thresh=10)

#drop unnecessary data
useful_data = useful_data.drop(['id', 'last_pymnt_d','url'], axis=1)
#drop features which is not numerical
useful_data = useful_data.drop(['earliest_cr_line', 'emp_title',
                           'issue_d','last_credit_pull_d','purpose','title'], axis=1)
#drop features indicating whether loan is paid off
useful_data = useful_data.drop(['total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int'
                            ,'total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt'], axis=1)
#deal with ratios which has "%" in it
useful_data['int_rate'] = useful_data['int_rate'].replace('%','',regex=True).astype('float')/100
useful_data['revol_util'] = useful_data['revol_util'].replace('%','',regex=True).astype('float')/100

#replace the status with 0 or 1
useful_data = useful_data.replace(['Charged Off','Does not meet the credit policy. Status:Charged Off',
 'Late (31-120 days)','In Grace Period','Late (16-30 days)','Default'], 0)
useful_data = useful_data.replace(['Fully Paid','Does not meet the credit policy. Status:Fully Paid','Current'], 1)


useful_data.to_csv('useful_data.csv')

#save those with no description for later use
description_null = useful_data[useful_data['desc'].isnull()]
description_null['desc_bool'] = 0
description_null.to_csv('part1.csv')

# save those rows with decsription
description = useful_data[useful_data['desc'].notnull()]
description.to_csv('desc_not_null.csv')

#devide the data into 2 classes: pos, neg    count[pos]=93677, count[neg]=17090  total=110767
des_pos = description[description['loan_status']==1]
des_pos.to_csv('desc_pos.csv')

des_neg = description[description['loan_status']==0]
des_neg.to_csv('desc_neg.csv')

des_pos_data = des_pos['desc'] 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
 
#将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频 
vectorizer=CountVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english',min_df=0.05)

#统计每个词语的tf-idf权值
transformer = TfidfTransformer()

#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
tfidf_pos = transformer.fit_transform(vectorizer.fit_transform(des_pos_data))  
wordlist_pos = vectorizer.get_feature_names()#获取词袋模型中的所有词  
print(wordlist_pos)
print(len(wordlist_pos))
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weight_pos = tfidf_pos.toarray()

tfidfDict_pos = {}
for i in range(len(weight_pos)):
    for j in range(len(wordlist_pos)):
        getWord = wordlist_pos[j]
        getValue = weight_pos[i][j]
        if getValue != 0:
            if getWord in tfidfDict_pos:
                tfidfDict_pos[getWord] += float(getValue)
            else:
                tfidfDict_pos.update({getWord:getValue})
sorted_tfidf_pos = sorted(tfidfDict_pos.items(), key = lambda d:d[1],reverse = True)

fw1 = open('pos_result.txt','w')
for i in sorted_tfidf_pos:
    if(not(i[0].isdigit()) and i[0]!='br'and i[0]!='borrower' and i[0]!='added'):
        fw1.write(i[0] + '\t' + str(i[1]) +'\n')

des_neg_data = des_neg['desc'] 

tfidf_neg = transformer.fit_transform(vectorizer.fit_transform(des_neg_data))  
wordlist_neg = vectorizer.get_feature_names()#获取词袋模型中的所有词  
print(wordlist_neg)
print(len(wordlist_neg))
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weight_neg = tfidf_neg.toarray()

tfidfDict_neg = {}
for i in range(len(weight_neg)):
    for j in range(len(wordlist_neg)):
        getWord = wordlist_neg[j]
        getValue = weight_neg[i][j]
        if getValue != 0:
            if getWord in tfidfDict_neg:
                tfidfDict_neg[getWord] += float(getValue)
            else:
                tfidfDict_neg.update({getWord:getValue})             
                
sorted_tfidf_neg = sorted(tfidfDict_neg.items(), key = lambda d:d[1],reverse = True)

fw2 = open('neg_result.txt','w')
for i in sorted_tfidf_neg:
    if(not(i[0].isdigit()) and i[0]!='br'and i[0]!='borrower' and i[0]!='added'):
        fw2.write(i[0] + '\t' + str(i[1]) +'\n')

tfidfDict = {}
for i in sorted_tfidf_pos:
    if i[0] in tfidfDict_neg:
        tfidfDict.update({i[0]:abs(tfidfDict_pos[i[0]]/len(des_pos_data) - tfidfDict_neg[i[0]]/len(des_neg_data))}) 
        
#print(tfidfDict)
sorted_tfidf = sorted(tfidfDict.items(), key = lambda d:d[1],reverse = True)

count = 0
key_word = []
for i in sorted_tfidf:
    if(not(i[0].isdigit()) and i[0]!='br'and i[0]!='borrower' and i[0]!='added' and count<20):
        count = count+1
        key_word.append(i[0])
        print(i[0] + '\t' + str(i[1]))

fw3 = open('key_word.txt','w')
for i in key_word:
    fw3.write(i +'\n')

# get the index of the columns in weight_pos that should be chosen
pos_index = []
for word in key_word:
    for j in range(len(wordlist_pos)):
        if(wordlist_pos[j] == word):
            pos_index.append(j)
print(pos_index)

pos_exist = []
for i in range(len(weight_pos)):
    pos_exist.append(0)
for i in range(len(weight_pos)):
    for j in pos_index:
        getValue = weight_pos[i][j]
        if getValue != 0:
            pos_exist[i] = 1
            
des_pos['desc_bool'] = pos_exist
des_pos.to_csv('part2.csv')

neg_index = []
for word in key_word:
    for j in range(len(wordlist_neg)):
        if(wordlist_neg[j] == word):
            neg_index.append(j)
print(neg_index)

neg_exist = []
for i in range(len(weight_neg)):
    neg_exist.append(0)
for i in range(len(weight_neg)):
    for j in neg_index:
        getValue = weight_neg[i][j]
        if getValue != 0:
            neg_exist[i] = 1
            

des_neg['desc_bool'] = neg_exist
des_neg.to_csv('part3.csv')


dataframes = []
for i in ['1','2','3']:
    current_df = pd.read_csv("part{:}.csv".format(i))
    dataframes.append(current_df)

combined_data = pd.concat(dataframes)
combined_data.to_csv('useful_data_with_boolean_desc.csv')


#上面是对description进行处理和置换，下面的是在处理其它的属性
num_valid = combined_data.count()
print(num_valid)

ratio = num_valid / float(389225)#389225 is the number of 'id'
ratio = ratio[ratio > 0.9]
ratio = ratio[ratio <= 1.01]
#only keep the features which have >90% data
data_new = combined_data[ratio.keys()]

#drop any loan has empty feature
data_new = data_new.dropna(axis=0, how='any', thresh=None, subset=None,inplace=False)

##begin processing the categorial feature
categorial_features = ['addr_state', 'application_type', 'emp_length', 'grade', 'home_ownership','initial_list_status'
                      , 'pymnt_plan', 'sub_grade', 'term', 'verification_status']
for feature in list(data_new.columns):
    if feature in categorial_features:
        data_new[feature] = data_new[feature].astype('category')
#this step named 'one-hot encoding'
loan_withzip = pd.get_dummies(data_new,columns = categorial_features )

#deal with the zip
zip_income = pd.read_csv('output_zipdata.csv')
#get first three digits of the zipcode such as "618" from "618xx", then change its datatype to int
loan_withzip['zip_code']=loan_withzip['zip_code'].apply(lambda x: x[:3])
loan_withzip['zip_3']=loan_withzip.zip_code.astype(int)

new_data_withzip = pd.merge(loan_withzip,zip_income,on='zip_3').sort_values(by=['Unnamed: 0'])
#I don't know where "Unnamed: 0" comes from...but I tried several times and found when ordered by this feature
#and merge, the result seems true

#drop the zipcode column
temp_data = new_data_withzip.drop(['zip_code','zip_3'], axis=1)

temp_data.to_csv('final_data.csv')

final_data = pd.read_csv('final_data.csv')
t = final_data['loan_status']

#change the dataframe to numpy array, which can be processed by sklearn package
label_numpy = np.array(t)
data_numpy = np.array(final_data.drop(['loan_status'], axis=1))

#Logistic regression
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(data_numpy, label_numpy, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data_train, label_train)
#regular score
regular_score = model.score(data_test, label_test)
print("Regular score is", regular_score)
#fancy score created by paper
predict_result = model.predict(data_test)
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(label_test, predict_result)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

sensitivity = float(TP)/float(TP+FN)
specificity = float(TN)/float(TN+FP)
G = np.sqrt(sensitivity*specificity)
print("G score is", G)
