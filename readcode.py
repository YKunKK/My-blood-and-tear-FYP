from twitter import Twitter, OAuth
import os
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

try:
    import json
except ImportError:
    import simplejson as json


ACCESS_TOKEN = '785462826819125248-xK61XzvjpJ42vQ34lnPBqdQpLy9Wg5q'
ACCESS_SECRET = 'lOKmipo9hdkBOaxKTAhAGbDF2ZVVgstUHFW1E4RcMbAKA'
CONSUMER_KEY = 'tyVSSCSswB5Xe2OsGqUsxJrSD'
CONSUMER_SECRET = 'IRedaRPyKKG1FLvTPgrN7CAdx9vSsyldIUPJrutp1wxeNLqE8z'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
t = Twitter(auth=oauth)


def extract_info(tweet, refer, tt_id ,tt_reply,tt_quote ,tt_retweet ,discount_index,tt_text ,tt_hashtags ,tt_location_post , tt_location_user):
    if refer <=3 and ('text' in tweet):
        tt_id.append(tweet['id_str'])
        if tweet['truncated']:
            try:
                tt_text.append(tweet['extended_tweet']['full_text'])
            except:
                tt_text.append(tweet['text'])
        else:
            tt_text.append(tweet['text'])

        hashtags = []
        for hashtag in tweet['entities']['hashtags']:
            hashtags.append(hashtag['text'])
        tt_hashtags.append([hashtags])
        tt_location_post.append(tweet['place'])
        tt_location_user.append(tweet['user']['location'])

        # need to trace back to original one and discount both info
        tt_reply.append(tweet['in_reply_to_status_id_str'] if tweet['in_reply_to_status_id_str'] != None else False)
        if tweet['is_quote_status'] == True:
            try:
                tt_quote.append(tweet['quoted_status'])
            except:
                try:
                    tt_quote.append(tweet['quoted_status_id_str'])
                except:
                    tt_quote.append(False)
        else:
            tt_quote.append(False)

        tt_retweet.append(tweet['retweeted_status'] if 'retweeted_status' in tweet else False)  # a json variable

        if tt_reply[-1] == False and tt_quote[-1]==False and tt_retweet[-1]==False:
            discount_index.append((0.2**(refer))*1)
        else:
            discount_index.append((0.2**(refer))*0.8)

            # previous twitter call may be denied
            if tt_reply[-1] != False:
                try:
                    #trace back by tweet id
                    tweet1 = t.statuses.show(_id=tt_reply[-1])
                    extract_info(tweet1, refer+1, tt_id ,tt_reply,tt_quote ,tt_retweet ,discount_index,tt_text ,tt_hashtags ,tt_location_post , tt_location_user)
                except:
                    pass


            if tt_quote[-1] != False:
                try:
                    extract_info(tt_quote[-1], refer+1, tt_id ,tt_reply,tt_quote ,tt_retweet ,discount_index,tt_text ,tt_hashtags ,tt_location_post , tt_location_user)
                except TypeError:
                    tweet1 = t.statuses.show(_id=tt_quote[-1])
                    extract_info(tweet1, refer + 1, tt_id ,tt_reply,tt_quote ,tt_retweet ,discount_index,tt_text ,tt_hashtags ,tt_location_post , tt_location_user)
                except:
                    pass

            if tt_retweet[-1] !=False:
                extract_info(tt_retweet[-1], refer+1, tt_id ,tt_reply,tt_quote ,tt_retweet ,discount_index,tt_text ,tt_hashtags ,tt_location_post , tt_location_user)


def one_day(filename):
    tt_id = []  # tweet id sequence, string

    # a tweet may be a reply / quote /retweet/forawrd /repost another tweet so may refer back to the original tweet
    ## (we may retrieve the referred tweet and discount that information)

    tt_reply = []  # tweet reply to another tweet, need discount, status_id saved, str or None
    tt_quote = []  # tweet quoting another tweet, retrieve the quoted one and discount both, False or json type
    tt_retweet = []  # tweet retweeting another one
    discount_index = []  # we give original tweet 0.2, new posts 0.8

    tt_text = []  # tweet content sequence
    tt_hashtags = []  # tweet hashtag
    tt_location_post = []  # tweet posted location
    tt_location_user = []  # tweet poster account location

    tweets_file = open(path + filename, "r")
    count = 0
    for line in tweets_file:
        count += 1
        print count

        try:
            tweets = json.loads(line.strip())

            # examine if duplicate tweets
            if tweets['id_str'] in tt_id:
                print 'duplicate'
                print tweets['id_str']

            elif 'text' not in tweets:
                print 'no text detected'

            else:
                extract_info(tweet=tweets, refer=0, tt_id=tt_id, tt_reply=tt_reply, tt_quote=tt_quote,
                             tt_retweet=tt_retweet, discount_index=discount_index, tt_text=tt_text,
                             tt_hashtags=tt_hashtags, tt_location_post=tt_location_post,
                             tt_location_user=tt_location_user)


        except:
            print 'error'
            continue

    tweets_file.close()

    d = {'id': tt_id, 'discount': discount_index, 'text': tt_text, 'hashtag': tt_hashtags,
             'post_location': tt_location_post, 'user_location': tt_location_user}

    return d


#------------------------------------------------------------------------------------------
#can not use parallel computation

path = '/Users/Kun/Desktop/Dropbox/research/twitter_data/'
combine_path = '/Users/Kun/Desktop/Dropbox/research/twitter_data/combine_'
num_cores = multiprocessing.cpu_count()
date=[]

for i in [16, 17, 18]:
    date.append('2016_12_%s' % str(i))
    file_index = []
    for filename in os.listdir(path):
        if filename.startswith(date[-1]):
            print filename
            file_index.append(filename)

    results = Parallel(n_jobs=num_cores)(delayed(one_day)(filename) for filename in file_index)
    full= pd.DataFrame(None, columns ={'id': None, 'discount': None, 'text': None, 'hashtag': None,
             'post_location': None, 'user_location': None})
    for w in range(len(results)):
        full = pd.concat([full, pd.DataFrame(results[w])], axis=0)

    pd.DataFrame(full).to_csv(path_or_buf=combine_path + date[-1] + '.csv', encoding='utf-8')





