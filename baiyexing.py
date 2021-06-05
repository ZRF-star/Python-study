import array
import re
from tkinter import _flatten

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB  #导入朴素贝叶斯分类器
from sklearn.model_selection import train_test_split #导入自动生成训练集和测试集的模块train_test_split
import jieba
import wordcloud
from wordcloud import ImageColorGenerator, STOPWORDS

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)#禁止自动换行

f = pd.read_excel("白夜行1.xlsx")

# 提取星级
f['star'] = f.rating_num.str.extract(r'(\d)')

# 删除列
f = f.drop(['rating_num', 'user_url', 'comment_time'], axis=1)
# 异常值
#f['content'] = f.content.replace('🤨', '微笑')
print(f)

#数据预处理
#中文分词
contents = f['content']
contents = list(contents)
contents1 = list()
for content in contents:#去除标点符号
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    content = comp.sub('', content)
    contents1.append(content)
print("中文分词：")
print(contents1)

#分词
stopwords = open("stopwords.txt", encoding="utf-8")
stopwords1 = list()
for stopword in stopwords.readlines():
    curLine = stopword.strip().split(" ")
    stopwords1.append(curLine)
stopwords1 = list(_flatten(stopwords1))#二维转一维
print("停用词：")
print(stopwords1)

new_Series = pd.Series()
#处理停用词
new_list = list()
for content1 in contents1:
    ls = list(jieba.cut_for_search(content1))
    ls = [w for w in ls if w not in stopwords1]
    txt = " ".join(ls)
    new_list.append(txt)
print("去除停用词：")
print(new_list)
new_Series = pd.Series(new_list)
f['content'] = new_Series

star = list(f['star'])
star1 = list()
for s in star:
    if s == s:
        star1.append(int(s))
    else:
        star1.append(0)

print("=============================可视化=======================")
# 计数
star_num = f.star.value_counts()
star_num = star_num.sort_index()
print(star_num)
print("====================词云图=====================")
print("===========绘制饼图================")
matplotlib.rcParams['font.family'] = 'Kaiti'#让中文字体正常显示
labels = '1星', '2星', '3星', '4星', '5星'
sizes = list(star_num)
plt.pie(sizes, explode=None, labels=labels, autopct='%1.1f%%')
plt.title("豆瓣评分比例")
plt.axis('equal')
plt.show()
print("===================================三个词云图=========================")
text1 = f[(f.star=='4')|(f.star=='5')]['content']
positive = list(text1)
file = open("positive.txt", 'a')
for i in range(len(positive)):
    s = str(positive[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
alice_coloring = np.array(Image.open('people-flower.jpg'))
f_w = open("positive.txt", "r")
t = f_w.read()
ls_w = jieba.lcut(t)
txt_w = " ".join(ls_w)
w = wordcloud.WordCloud(font_path="msyh.ttc", background_color="black",
                        mask=alice_coloring, collocations=False,
                        stopwords=['一点', '这部','片子','小时','半小时',
                                   '好好','电视剧','人物','之间','电视']
                        )
w.generate(t)
image_color = ImageColorGenerator(alice_coloring)
plt.imshow(w, interpolation='bilinear')
plt.title("正向评分原因")
plt.axis('off')
plt.show()
w.to_file("positive.png")
print("==================负向评分原因===============")
text2 = f[(f.star=='1')|(f.star=='2')]['content']
negative = list(text2)
file = open("negative.txt", 'a')
for i in range(len(negative)):
    s = str(negative[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
alice_coloring = np.array(Image.open('people-flower.jpg'))
f_w = open("negative.txt", "r")
t = f_w.read()
ls_w = jieba.lcut(t)
txt_w = " ".join(ls_w)
w = wordcloud.WordCloud(font_path="msyh.ttc", background_color="black",
                        mask=alice_coloring, collocations=False,
                        stopwords=['一点', '这部','片子','小时','半小时',
                                   '好好','电视剧','人物','之间','小说','看过','电视']
                        )
w.generate(t)
image_color = ImageColorGenerator(alice_coloring)
plt.imshow(w, interpolation='bilinear')
plt.title("负向评分原因")
plt.axis('off')
plt.show()
w.to_file("negative.png")
print("=============================中评原因======================")
text3 = f[(f.star=='3')]['content']
medium = list(text3)
file = open("medium.txt", 'a')
for i in range(len(medium)):
    s = str(medium[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
alice_coloring = np.array(Image.open('people-flower.jpg'))
f_w = open("medium.txt", "r")
t = f_w.read()
ls_w = jieba.lcut(t)
txt_w = " ".join(ls_w)
w = wordcloud.WordCloud(font_path="msyh.ttc", background_color="black",
                        mask=alice_coloring, collocations=False,
                        stopwords=['一点', '这部','片子','小时','半小时',
                                   '好好','电视剧','人物','之间','小说','看过',
                                   '电视','看过','故事']
                        )
w.generate(t)
image_color = ImageColorGenerator(alice_coloring)
plt.imshow(w, interpolation='bilinear')
plt.title("中评评分原因")
plt.axis('off')
plt.show()
w.to_file("medium.png")
print("==================================================================================")
star2 = list()
for st in star1:
    if st > 3:
        st = 1
        star2.append(st)
    else:
        st = 0
        star2.append(st)
f['star'] = star2
print(f['star'])
print(f)

print("=============================朴素贝叶斯======================")
clf = MultinomialNB()
x = f['content']
x = list(x)
y = f['star']
y = list(y)
n = len(x)//8
x_train, y_train = x[n:], y[n:]
x_train = pd.Series(x_train)
y_train = pd.Series(y_train)
x_test, y_test = x[:n], y[:n]
x_test = pd.Series(x_test)
y_test = pd.Series(y_test)


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
count_vec = CountVectorizer(max_df=0.8, min_df=3)
tfidf_vec = TfidfVectorizer()

def MNB_Classifier():
    return Pipeline([
        ('count_vec', CountVectorizer()),
        ('mnb', MultinomialNB())
    ])
mnbc_clf = MNB_Classifier()
# 进行训练
print("Start training...")
mnbc_clf.fit(x_train, y_train)
print("training done!")
answer_b = mnbc_clf.predict(x_test)
print("0：差评和中评；1：好评")
print(answer_b)
print("Prediction done!")
#准确率测试
accuracy=metrics.accuracy_score(y_test,answer_b)
print('准确率：'+str(accuracy))
print("The classification report for b:")
print(classification_report(y_test, answer_b))






