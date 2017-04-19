import json
import os


#path = 'D:/Dropbox/research/FYP/1.1twitter_api/problematic_dataset/'
path = '/Users/Kun/Desktop/Dropbox/research/FYP/1.1twitter_api/problematic_dataset/'
revise_path = '/Users/Kun/Desktop/Dropbox/research/twitter_data/'

for filename in os.listdir(path):

    tweets_file = open(path + filename, "r")

    dat = tweets_file.readline()
    dat2 = tweets_file.readline()
    tweets_file.close()

    if dat2 != '':
        print 'check' + filename
    else:
        print 'problematic dateset '+ filename
        slicing = dat
        n = 0
        index_begin, index_end =0, 0
        new_file = open(revise_path+filename, "w")
        while len(slicing)>0 and n<=1000:
            index_begin = slicing.find('{"favorited":')
            index_end = slicing[12:].find('{"favorited"')
            if index_end == -1:
                break
            index_end_check = slicing[:(index_end+12)].rfind('"in_reply_to_user_id"')
            if index_end_check>index_end+12-100:
                line = slicing[index_begin:(index_end+12)]
                new_file.write(line)
                new_file.write('\n')
                n +=1
            else:
                line = ''
            slicing = slicing[(index_end+12):]
            print n


        new_file.close()

################################################
