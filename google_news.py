from newspaper import Article
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import datetime
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
sid = SentimentIntensityAnalyzer()



def sentiment(text):
    try:
        chunks = text.split('\n')
        scores = []
        length = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 30:

                if len(chunk) > 1000:
                    epochs = int(len(chunk) / 500)
                    character_per_epochs = int(len(chunk) / epochs)
                    start = 0

                    for i in range(1, epochs + 1):
                        segment = chunk[start:(i * character_per_epochs)]
                        start = i * character_per_epochs
                    # print(segment)

                        vader_score = sid.polarity_scores(segment)
                    # print('Vader Score')
                    # print(vader_score)
                        length.append(len(segment))
                        scores.append(vader_score['compound'])


                else:
                # print(chunk)

                    vader_score = sid.polarity_scores(chunk)
                # print('Vader Score')
                # print(vader_score)
                    length.append(len(chunk))
                    scores.append(vader_score['compound'])

        scores = np.array(scores, dtype=np.float)
        length = np.array(length, dtype=np.float)
        length = length / length.max()
        scores = scores * length
        return scores.mean()
    except:
        return 0



url_string='https://www.google.com/search?q=microsoft&tbm=nws&tbs=sbd:1'

req = Request(url_string,headers={'User-Agent': 'Chrome/86.0.4240.198'})
page = urlopen(req).read()
page = BeautifulSoup(page, 'lxml')
news_links = page.find_all('div', {'class': 'kCrYT'})
link_list = []
time_now = datetime.datetime.now()
for link in news_links:
            
    try:
        time = link.find('span').text
        web = link.find('a')['href']
        web = web.split('q=')[1]
        web = web.split('&sa')[0]
        link_list.append(web)
        if 'mins' in time or 'min' in time:
            backward = time.split(' ')[0]
            delta = datetime.timedelta(minutes=int(backward))
            time_of_article = time_now - delta
        elif 'hour' in time or 'hours' in time:
            backward = time.split(' ')[0]
            delta = datetime.timedelta(hours=int(backward))
            time_of_article = time_now - delta
        elif 'day' in time or 'days' in time:
            backward = time.split(' ')[0]
            delta = datetime.timedelta(days=int(backward))
            time_of_article = time_now - delta
        elif 'week' in time or 'weeks' in time:
            backward = time.split(' ')[0]
            delta = datetime.timedelta(weeks=int(backward))
            time_of_article = time_now - delta
    except:
        pass
               
print(link_list)

for web in link_list:               
    article = Article(web)
    article.download()
    article.parse()
    article.nlp()
    text = article.text
            #print(text)


                            #print(time_of_article.strftime('%Y-%m-%d %H:%M'))
                                
                        #print(web)
                               

    score=sentiment(text)
    print(score)

                        #print('======================')

                          
          
                
