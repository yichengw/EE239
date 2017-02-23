import json
import datetime, time
import matplotlib.pyplot as plt
import math
import numpy as np

f = open('tweets_#gopatriots.txt', 'r', encoding = 'utf-8')

line = f.readline()

tweets = []
followers = []
retweets = []
while len(line)!=0:
    tweet = json.loads(line)
    tweets.append(tweet)
    line = f.readline()    

start_time = tweets[0]['firstpost_date']
end_time = tweets[-1]['firstpost_date']
index_len = math.ceil((end_time - start_time)/3600)
tweet_num = np.zeros(index_len).tolist()

for i in range(len(tweets)):
    tweet_num[int((tweets[i]['firstpost_date'] - start_time)/3600)] += 1
    retweets.append(tweets[i]['metrics']['citations']['total'])
    followers.append(tweets[i]['author']['followers'])


avg_tweets = len(tweets)/((end_time - start_time)/3600)
avg_retweets = sum(retweets)/len(tweets)
avg_followers = sum(followers)/len(tweets)

print('Average number of tweets per hour is:')
print(avg_tweets)
print('Average number of followers is:')
print(avg_followers)
print('Average number of retweets is:')
print(avg_retweets)


plt.figure()
index = np.arange(index_len).tolist()
width = 1
p = plt.bar(index, tweet_num, width)
plt.xlabel('Time')
plt.ylabel('Number of Tweets per hour')
plt.title('Number of Tweets per hour for #superbowl')
plt.show()
