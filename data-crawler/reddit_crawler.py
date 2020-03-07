#!/usr/bin/env python
# coding: utf-8

import praw
import json
import pandas as pd


reddit = praw.Reddit(client_id='nrYllY_Ws3f0QA',
                     client_secret='EQS-keemsE2DGDm9MjimY18IIgs',
                     user_agent='script by /u/Segment__Fault for academic usage')


def get_subreddit_list():
    subreddit_list = []
    for subreddit in reddit.user.subreddits(limit=None):
        subreddit_list.append(subreddit.display_name)
    return subreddit_list



subreddit_list = get_subreddit_list()


def get_redditor_list(subreddit_name, limit):
    redditor_set = set()
    for submission in reddit.subreddit(subreddit_name).hot(limit=limit):
        if submission is not None and submission.author is not None:
            redditor_set.add(submission.author.name)
    return list(redditor_set)


def get_redditor_subreddit_tuple_list(subreddit_list, limit):
    total_list = []
    next_progress = 0
    i=0
    for subreddit_name in subreddit_list:
        total_list += [(i, subreddit_name) for i in get_redditor_list(subreddit_name, limit)]
        i+=1
        if ((i/len(subreddit_list)*100)>=next_progress):
            print("{}% finished {}/{}".format(next_progress,i,len(subreddit_list)))
            next_progress += 5
    return total_list


redditor_subreddit_tuple_list = get_redditor_subreddit_tuple_list(subreddit_list,100)


redditor_dict = {}
popular_redditor_set = set()
for i in redditor_subreddit_tuple_list:
    redditor_name = i[0]
    subreddit_name = i[1]
    if redditor_name in redditor_dict:
        redditor_dict[redditor_name].append(subreddit_name)
        if len(redditor_dict[redditor_name]) >=3:
            popular_redditor_set.add(redditor_name)
    else:
        redditor_dict[redditor_name] = [subreddit_name]



popular_redditor_dict={}
for i in popular_redditor_set:
    popular_redditor_dict[i] = set(redditor_dict[i])


del popular_redditor_dict['AutoModerator']



serializable_types = {int, str, bool, dict, float, list, tuple}
def make_serializable(x):
    if 'subreddit' in x:
        x['subreddit'] = x['subreddit'].display_name
    if 'author' in x:
        x['author'] = x['author'].name
    if '_replies' in x:
        x['_replies'] = [i.id for i in x['_replies']]
    key_to_delete = []
    for k,v in x.items():
        if v is not None and type(v) not in serializable_types:
            key_to_delete.append(k)
    for k in key_to_delete:
        del x[k]
    return x


def get_new_submissions(redditor_name, limit):
    d = []
    for c in reddit.redditor(redditor_name).submissions.new(limit=limit):
        d.append(make_serializable(c.__dict__.copy()))
    return d

def get_hot_comments(redditor_name, limit):
    d = []
    for c in reddit.redditor(redditor_name).comments.hot(limit=limit):
        d.append(make_serializable(c.__dict__.copy()))
    return d


print(len(popular_redditor_dict))


total_comments= []
i =0
next_progress = 0
for k,v in popular_redditor_dict.items():
    redditor_comments = get_hot_comments(k, 500)
    total_comments += redditor_comments
    i+=1
    if ((i/len(popular_redditor_dict)*100)>=next_progress):
        print("{}% finished {}/{}".format(next_progress, i, len(popular_redditor_dict)))
        next_progress += 5
        with open("comments", "w") as f:
            f.write(json.dumps(total_comments))


with open("comments", "w") as f:
    f.write(json.dumps(total_comments))


with open("comments", "w") as f:
    f.write(json.dumps(total_comments))
pd.read_json("comments").to_csv("comments.csv")


def get_statistics(total_comments):
    subreddit_count = {}
    subreddit_author_dict = {}
    author_count = {}
    author_subreddit_dict = {}
    
    for i in total_comments:
        subreddit_name = i['subreddit']
        author_name = i['author']
        author_count[author_name] = author_count.get(author_name, 0) + 1
        subreddit_count[subreddit_name] = subreddit_count.get(subreddit_name,0)+1
        if subreddit_name in subreddit_author_dict:
            subreddit_author_dict[subreddit_name].add(author_name)
        else:
            subreddit_author_dict[subreddit_name] = set([author_name])
        if author_name in author_subreddit_dict:
            author_subreddit_dict[author_name].add(subreddit_name)
        else:
            author_subreddit_dict[author_name] = set([subreddit_name])
            
    subreddit_author_count = {}
    author_subreddit_count = {}
    for k,v in subreddit_author_dict.items():
        subreddit_author_count[k] = len(v)
    for k,v in author_subreddit_dict.items():
        author_subreddit_count[k] = len(v)
    
    return subreddit_count, subreddit_author_count, author_count, author_subreddit_count



subreddit_count, subreddit_author_count, author_count, author_subreddit_count = get_statistics(total_comments)
filtered_subreddit_set = set([k for k,v in subreddit_author_count.items() if v >=4])


tmp_filtered_total_comments = []
for i in total_comments:
    if i['subreddit'] in filtered_subreddit_set:
        tmp_filtered_total_comments.append(i)


subreddit_count, subreddit_author_count, author_count, author_subreddit_count = get_statistics(tmp_filtered_total_comments)
filtered_author_set = set([k for k,v in author_subreddit_count.items() if v >=3])

filtered_total_comments = []
for i in tmp_filtered_total_comments:
    if i['author'] in filtered_author_set:
        filtered_total_comments.append(i)


subreddit_count, subreddit_author_count, author_count, author_subreddit_count = get_statistics(filtered_total_comments)

with open("filtered_comments", "w") as f:
    f.write(json.dumps(filtered_total_comments))
pd.read_json("filtered_comments").to_csv("filtered_comments.csv")

subreddit_count, subreddit_author_count, author_count, author_subreddit_count = get_statistics(filtered_total_comments)

with open("subreddit_count", "w") as f:
    f.write(json.dumps([(k,v) for k,v in subreddit_count.items()]))
pd.read_json("subreddit_count").to_csv("subreddit_count.csv")

with open("subreddit_author_count", "w") as f:
    f.write(json.dumps([(k,v) for k,v in subreddit_author_count.items()]))
pd.read_json("subreddit_author_count").to_csv("subreddit_author_count.csv")

subreddit_author_list = [v for k,v in subreddit_author_count.items()]
author_subreddit_list = [v for k,v in author_subreddit_count.items()]
print( "A sample dataset with {} comments, including {} authors and {} groups\nEach group has at least {} distinct authors, {} authors in average\nEach author appears at {} groups at least, {} groups in average".format(
    len(filtered_total_comments),len(author_count), len(subreddit_count), min(subreddit_author_list), sum(subreddit_author_list)/len(subreddit_author_list),min(author_subreddit_list), sum(author_subreddit_list)/len(author_subreddit_list)))


print(len(total_comments))
