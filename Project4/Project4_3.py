import json
import datetime, time
import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_predict, cross_val_score

f = open('tweet_data/tweets_#gohawks.txt', 'r', encoding = 'utf-8')

line = f.readline()

tweets = []


while len(line)!=0:
    tweet = json.loads(line)
    tweets.append(tweet)
    line = f.readline()    

tweet_0=tweets[0]
tweet_100=tweets[100]
time_hour = np.zeros(len(tweets)).tolist()
user_mentioned=[]
entities_url=[]
metrics_momentum=[]
metrics_influential= []
for i in range(len(tweets)):
    time_hour_temp=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(tweets[i]['firstpost_date']))
    time_hour[i]=(int(time_hour_temp[3])-4)*12*31*24+(int(time_hour_temp[5])*10+int(time_hour_temp[6])-1)*31*24+int(time_hour_temp[8])*10*24+int(time_hour_temp[9])*24+int(time_hour_temp[11])*10+int(time_hour_temp[12])
    if len(tweets[i]['tweet']['entities']['user_mentions'])>1:
        user_mentioned.append(tweets[i])
    if len(tweets[i]['tweet']['entities']['urls'])>1:
        entities_url.append(tweets[i])
    if tweets[i]['metrics']['momentum']>1:
        metrics_momentum.append(tweets[i]['metrics']['momentum'])
    if tweets[i]['metrics']['citations']['influential']>1:
        metrics_influential.append(tweets[i]['metrics']['citations']['influential'])



#start_time = tweets[0]['firstpost_date']
#end_time = tweets[-1]['firstpost_date']
index_len = (time_hour[-1] - time_hour[0])+1
tweet_num = np.zeros(index_len).tolist()
retweet_num = np.zeros(index_len).tolist()
follower_num = np.zeros(index_len).tolist()
max_follower_num = np.zeros(index_len).tolist()
time_of_the_day = np.zeros(index_len).tolist()
mention_num = np.zeros(index_len).tolist()
author_num = np.zeros(index_len).tolist()
uniqueAuthors_temp = np.zeros(index_len).tolist()
Cooccurrence_times = np.zeros(index_len).tolist()
ranking_scores = np.zeros(index_len).tolist()
impression = np.zeros(index_len).tolist()
reply_num = np.zeros(index_len).tolist()
favorite_count = np.zeros(index_len).tolist()

for i in range(index_len):
    uniqueAuthors_temp[i] = []
'''
time_0 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(tweets[100]['citation_date']))
time_1 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(tweets[100]['firstpost_date']))
'''

'''
uniqueAuthors = []
for i in range(len(tweets)):
    if tweets[i]['author']['name'] in uniqueAuthors:
        continue
    else:
        uniqueAuthors.append(tweets[i]['author']['name'])


uniqueOriginal =  []
for i in range(len(tweets)):
    if tweets[i]['original_author']['name'] in uniqueOriginal:
        continue
    else:
        uniqueOriginal.append(tweets[i]['original_author']['name'])
'''
#feature generation
for i in range(len(tweets)):
    hour_index = time_hour[i]-time_hour[0]
    tweet_num[hour_index] += 1
    retweet_num[hour_index] += tweets[i]['metrics']['citations']['total']
    reply_num[hour_index] += tweets[i]['metrics']['citations']['replies']
    follower_num[hour_index] += tweets[i]['author']['followers']
    max_follower_num[hour_index] = max(max_follower_num[hour_index], tweets[i]['author']['followers'])
    mention_num[hour_index] += len(tweets[i]['tweet']['entities']['user_mentions'])
    ranking_scores[hour_index] += tweets[i]['metrics']['ranking_score']
    favorite_count[hour_index] += tweets[i]['tweet']['favorite_count']
    impression[hour_index] += tweets[i]['metrics']['impressions']
    if len(tweets[i]['tweet']['entities']['hashtags'])>=2:
        Cooccurrence_times[hour_index] += 1
    if tweets[i]['author']['name'] in uniqueAuthors_temp[hour_index]:
        continue
    else:
        uniqueAuthors_temp[hour_index].append(tweets[i]['author']['name'])
        author_num[hour_index] += 1

for i in range(len(time_of_the_day)):
    time_of_the_day[i] = (time_hour[0]+i)%24

for i in range(len(time_of_the_day)):
    if time_of_the_day[i] == 0:
        time_of_the_day[i] = 24

X_train = np.array([tweet_num[:-1], retweet_num[:-1], follower_num[:-1], max_follower_num[:-1], time_of_the_day[:-1], mention_num[:-1], author_num[:-1], Cooccurrence_times[:-1] ,ranking_scores[:-1] , reply_num[:-1], favorite_count[:-1], impression[:-1], np.ones(len(tweet_num)-1)])
X_train = X_train.T
Y_train = np.array(tweet_num[1:])


#training
print(sm.OLS(Y_train, X_train).fit().summary())

result=sm.OLS(Y_train, X_train).fit().predict()

#rf=LinearRegression()
#rf.fit(X_train,Y_train)
#predicted = predicted = cross_val_predict(rf, X_train, Y_train, cv=10)
mse = mean_squared_error(result,Y_train)
rsme=math.sqrt(mse)


fig,ax = plt.subplots()
ax.scatter(Y_train, result)
ax.plot([0,4000],[0,4000],'k--',lw=4) ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()

residuals=abs(Y_train-result)
fig2 = plt.subplot()
plt.scatter(result,residuals)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

'''
plt.figure()
index = np.arange(index_len).tolist()
width = 1
p = plt.bar(index, tweet_num, width)
plt.xlabel('Time')
plt.ylabel('Number of Tweets per hour')
plt.title('Number of Tweets per hour for #superbowl')
plt.show()
'''