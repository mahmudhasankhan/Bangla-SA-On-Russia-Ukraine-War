from googleapiclient.discovery import build
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import os

youtube = build('youtube', 'v3', developerKey=os.environ["YOUTUBE_API_KEY"])

box = [['Name', 'Comment', 'Time', 'Likes', 'Reply Count']]
# youtube video ids
code_lang = [
    {"id": "nbQVPSvGnmQ"},
    {"id": "HaJCwY9YBQ0"},
    {"id": "AjzMrDla0OA"},
    {"id": "ZaudmfidJBU"},
    {"id": "aUnwTNrDAD8"},
    {"id": "jWGwRW5tbqA"},
    {"id": "4wKVfyds4YU"},
    {"id": "FYlg-7AsGlY"},
    {"id": "JHVX2sUFdPk"},
    {"id": "T65Xg4tbuoE"},
    {"id": "HoiEu7F2OWg"},
    {"id": "pekBo-ynIhI"},
    {"id": "Dj89023bI64"},
    {"id": "VbQzh1t6Qfo"},
    {"id": "ToX4p1LnNrU"}
]

for id_code in code_lang:
    def scrape_comments_with_replies():
        data = youtube.commentThreads().list(part='snippet', videoId=id_code['id'], maxResults='100',
                                             textFormat="plainText").execute()

        for i in data["items"]:
            name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
            comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
            published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
            likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
            replies = i["snippet"]['totalReplyCount']

            box.append([name, comment, published_at, likes, replies])

            totalReplyCount = i["snippet"]['totalReplyCount']

            if totalReplyCount > 0:

                parent = i["snippet"]['topLevelComment']["id"]

                data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                textFormat="plainText").execute()

                for i in data2["items"]:
                    name = i["snippet"]["authorDisplayName"]
                    comment = i["snippet"]["textDisplay"]
                    published_at = i["snippet"]['publishedAt']
                    likes = i["snippet"]['likeCount']
                    replies = ""

                    box.append([name, comment, published_at, likes, replies])

        while ("nextPageToken" in data):

            data = youtube.commentThreads().list(part='snippet', videoId=id_code['id'], pageToken=data["nextPageToken"],
                                                 maxResults='100', textFormat="plainText").execute()

            for i in data["items"]:
                name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
                comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
                published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
                likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
                replies = i["snippet"]['totalReplyCount']

                box.append([name, comment, published_at, likes, replies])

                totalReplyCount = i["snippet"]['totalReplyCount']

                if totalReplyCount > 0:

                    parent = i["snippet"]['topLevelComment']["id"]

                    data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                    textFormat="plainText").execute()

                    for i in data2["items"]:
                        name = i["snippet"]["authorDisplayName"]
                        comment = i["snippet"]["textDisplay"]
                        published_at = i["snippet"]['publishedAt']
                        likes = i["snippet"]['likeCount']
                        replies = ''

                        box.append(
                            [name, comment, published_at, likes, replies])

        df = pd.DataFrame({'Name': [i[0] for i in box], 'Comment': [i[1] for i in box], 'Time': [i[2] for i in box],
                           'Likes': [i[3] for i in box], 'Reply Count': [i[4] for i in box]})

        sql_vids = pd.DataFrame([])

        sql_vids = sql_vids.append(df, ignore_index=True)

        sql_vids.to_csv(
            '../Dataset/Raw-Data/youtube-comments-bbc.csv', index=False, header=False)

    scrape_comments_with_replies()
