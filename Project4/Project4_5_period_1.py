
import json
import datetime, time
import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_predict, cross_val_score
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error

f = open('tweets_#gopatriots.txt', 'r', encoding = 'utf-8')

line = f.readline()

tweets = []


while len(line)!=0:
    tweet = json.loads(line)
    tweets.append(tweet)
    line = f.readline()    


time_hour = np.zeros(len(tweets)).tolist()
user_mentioned=[]
entities_url=[]
metrics_momentum=[]
metrics_influential= []
for i in range(len(tweets)):
    time_hour_temp=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(tweets[i]['firstpost_date']))
    time_hour[i]=(int(time_hour_temp[3])-4)*12*31*24+(int(time_hour_temp[5])*10+int(time_hour_temp[6])-1)*31*24+int(time_hour_temp[8])*10*24+int(time_hour_temp[9])*24+int(time_hour_temp[11])*10+int(time_hour_temp[12])
    if time_hour[i]>=12*31*24+1*31*24+24+8 and time_hour[i-1] <12*31*24+1*31*24+24+8:
        start_i_2 = i
        end_i_1 = i-1
    if time_hour[i]>=12*31*24+1*31*24+24+20 and time_hour[i-1]<12*31*24+1*31*24+24+20:
        start_i_3 = i
        end_i_2 = i-1
    
end_i_3=i
start_i_1=0

start_1=time_hour[0]
end_1=12*31*24+1*31*24+24+8-1
start_2=12*31*24+1*31*24+24+8
end_2=12*31*24+1*31*24+24+20-1
start_3=12*31*24+1*31*24+24+20
end_3=time_hour[-1]

index_len_1 = (end_1 - start_1)+1
index_len_2 = (end_2 - start_2)+1
index_len_3 = (end_3 - start_3)+1




#feature generation
def gen_feature(index_len, start_time, start_i, end_i):
    #feature initial
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
    #feature_generation
    for i in range(start_i,end_i+1):
        hour_index = time_hour[i]-start_time
        tweet_num[hour_index] += 1
#        retweet_num[hour_index] += tweets[i]['metrics']['citations']['total']
#        reply_num[hour_index] += tweets[i]['metrics']['citations']['replies']
#        follower_num[hour_index] += tweets[i]['author']['followers']
#        max_follower_num[hour_index] = max(max_follower_num[hour_index], tweets[i]['author']['followers'])
#        mention_num[hour_index] += len(tweets[i]['tweet']['entities']['user_mentions'])
        ranking_scores[hour_index] += tweets[i]['metrics']['ranking_score']
#        favorite_count[hour_index] += tweets[i]['tweet']['favorite_count']
#        impression[hour_index] += tweets[i]['metrics']['impressions']
        if len(tweets[i]['tweet']['entities']['hashtags'])>=2:
            Cooccurrence_times[hour_index] += 1
#        if tweets[i]['author']['name'] in uniqueAuthors_temp[hour_index]:
#            continue
#        else:
#            uniqueAuthors_temp[hour_index].append(tweets[i]['author']['name'])
#            author_num[hour_index] += 1

    for i in range(len(time_of_the_day)):
        time_of_the_day[i] = (start_time+i)%24

    for i in range(len(time_of_the_day)):
        if time_of_the_day[i] == 0:
            time_of_the_day[i] = 24

    X_train = np.array([tweet_num[:-1], Cooccurrence_times[:-1] ,ranking_scores[:-1], np.ones(len(tweet_num)-1)])
    X_train = X_train.T
    Y_train = np.array(tweet_num[1:])
    return X_train,Y_train

rf=LinearRegression()
#training Before Feb. 1, 8:00 a.m.
X_train,Y_train = gen_feature(index_len_1,start_1,start_i_1,end_i_1)
rf.fit(X_train,Y_train)
f = open('sample8_period1.txt', 'r', encoding = 'utf-8')
line = f.readline()
tweets = []
while len(line)!=0:
    tweet = json.loads(line)
    tweets.append(tweet)
    line = f.readline()    

time_hour = np.zeros(len(tweets)).tolist()
for i in range(len(tweets)):
    time_hour_temp=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(tweets[i]['firstpost_date']))
    time_hour[i]=(int(time_hour_temp[3])-4)*12*31*24+(int(time_hour_temp[5])*10+int(time_hour_temp[6])-1)*31*24+int(time_hour_temp[8])*10*24+int(time_hour_temp[9])*24+int(time_hour_temp[11])*10+int(time_hour_temp[12])
tweet_num=np.zeros(6)
Cooccurrence_times=np.zeros(6)
ranking_scores=np.zeros(6)
for i in range(len(tweets)):
    time = time_hour[i]-time_hour[0]+1
    for j in range(1,7):
        if time==j:
            tweet_num[j-1] += 1
            ranking_scores[j-1] += 1
            if len(tweets[i]['tweet']['entities']['hashtags'])>=2:
                Cooccurrence_times[j-1] += 1
X_test=np.array([tweet_num[:],Cooccurrence_times[:],ranking_scores[:],np.ones(6)])
X_test=X_test.T
predicted=rf.predict(X_test)
print(mean_absolute_error(predicted[0:4],tweet_num[1:5]))
print(predicted[5])

