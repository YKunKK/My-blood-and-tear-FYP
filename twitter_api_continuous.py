#reference: http://socialmedia-class.org/twittertutorial.html
import time
# Import the necessary package to process data in JSON format
import json


# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = '785462826819125248-xK61XzvjpJ42vQ34lnPBqdQpLy9Wg5q'
ACCESS_SECRET = 'lOKmipo9hdkBOaxKTAhAGbDF2ZVVgstUHFW1E4RcMbAKA'
CONSUMER_KEY = 'tyVSSCSswB5Xe2OsGqUsxJrSD'
CONSUMER_SECRET = 'IRedaRPyKKG1FLvTPgrN7CAdx9vSsyldIUPJrutp1wxeNLqE8z'



for i in range(24):

    filename = '%s_%s_%s_%s_%s_%s' % time.localtime()[0:6]
    print filename
    #filename = 'testing'
    f = open('%s.txt' % filename, 'w')
    oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    # Initiate the connection to Twitter Streaming API
    twitter_stream = TwitterStream(auth=oauth)

    # Get a sample of the public data following through Twitter
    # we only process english language posts
    iterator = twitter_stream.statuses.sample(language='en')


    # Print each tweet in the stream to the screen
    tweet_count = 1000
    for tweet in iterator:
        tweet_count -= 1
        # Twitter Python Tool wraps the data returned by Twitter
        # as a TwitterDictResponse object.
        # We convert it back to the JSON format to print/score

        #print json.dumps(tweet)
        f.write(json.dumps(tweet))
        f.write('\n')

        # The command below will do pretty printing for JSON data, try it out
        #print json.dumps(tweet, indent=4)


        if tweet_count <= 0:
            break

    f.close()

    time.sleep(3600*2)