import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
import random
# Webページを取得して解析する
df = pd.read_csv('data/lyrics.csv')
for i in range(20):

    print(id)
    id = random.randint(1,292534)
    while id in df['ID']:
        id = id + 1
    load_url = "https://www.uta-net.com/song/" + str(id)
    html = requests.get(load_url)
    soup = BeautifulSoup(html.content, "html.parser")

    # HTML全体を表示する
    try:
        lyrics = soup.select("#kashi_area")[0].text
        title = soup.select("#view_kashi")[0].find("h2").text
        name = soup.select("#view_kashi")[0].find_all("span", itemprop="byArtist name")[0].text
        df=df.append({'ID' : id, 'Title' :title ,'Artists':name , 'lyrics' : lyrics} , ignore_index=True)
    except:
        continue
    time.sleep(1)
print(df)
df.to_csv('data/lyrics.csv', index=False)