class Animal():
    def __init__(self, name, category):
        self.name = name
        self.category = category

class Dog(Animal):
    def __init__(self, name, category, color):
        super().__init__(name, category)
        self.color = color
def re_match_():
    import re
    s = '对方在 “团购1群”中加我好友，他的微信号是wxid_12345，昵称”xyz”，我的微信号 “lz1234”，名字叫“石头”。'
    wxh = re.compile(r'微信号')
    wxid = re.compile(r'[0-9a-zA-Z_]+')
    re_wxh = [i.span() for i in wxh.finditer(s)]
    re_wxid = [i.span() for i in wxid.finditer(s)]
    def bs(t):
        l, r = 0, len(re_wxid)
        while l < r:
            m = (l+r) >> 1
            if re_wxid[m][0] >= t:
                r = m
            else:
                l = m + 1
        return l
    res = []
    for i,j in re_wxh:
        _id = bs(j)
        _id = re_wxid[_id]
        res.append({
            'start': _id[0],
            'end': _id[1],
            'value': s[_id[0]: _id[1]],
            'type': '微信号'
        })
    print(res)
def get_all_combs(array, target):
    al = len(array)
    def bs(t, l=0, r=al):
        while l < r:
            m = (l+r) >> 1
            if array[m] >= t:
                r = m
            else:
                l = m + 1
        return l
    res = []
    for i, t in enumerate(array):
        j = bs(target-t, i+1)
        if j!=al and array[j]==target-t:
            res.append([array[i], array[j]])
    return res
def find_num(array, num):
    nl = len(array)
    if array[-1] < array[0]:
        l, r = 0, nl-1
        while r-l>1:
            m = (l+r) >> 1
            if array[m] > array[l]:
                l = m
            else:
                r = m
        array = array[r:] + array[:r]
        idx = r
    else:
        idx = 0
    l, r = 0, nl
    while l < r:
        m = (l+r) >> 1
        if array[m] >= num:
            r = m
        else:
            l = m + 1
    if l!=nl and array[l]==num:
        if l >= nl-idx:
            l -= nl-idx
        else:
            l += idx
        return l
    return -1
# re_match_()
import string
import random
import numpy as np
import pandas as pd
##%% md
# 随机生成一些名字和分数
##%%
cnt = 100
name = set()
while len(name) < cnt:
    name.add(''.join(random.choice(string.ascii_lowercase) for _ in range(5)))
name = list(name)

df_score = pd.DataFrame({'name': name, 'score': np.random.randint(80, 100, cnt)})
df_score.head()
##%% md
# 给随机名字分配班级
##%%
classes = ['A', 'B', 'C']
df_class = pd.DataFrame({'name': name, 'class': [random.choice(classes) for _ in range(cnt)]})
df_class = df_class.sample(frac=1).reset_index(drop=True)
df_class.head()
##%% md
# 题目 1： 按照名字合并分数和班级
##%%
df_all = pd.merge(df_score, df_class, on='name')
print(df_all)
##%% md
# 题目 2： 取出 A 班的成绩表，按照分数降序排序
##%%
df_A_score = df_all.loc[df_all['class']=='A']
df_A_score.sort_values(by='score', inplace=True, ascending=False)
print(df_A_score)
##%% md
# 题目 3： 计算 A、B、C 班的平均分
##%%
classes = ['A', 'B', 'C']
res = [] # 一次为A、B、C三班的平均值
for c in classes:
    res.append(df_all.loc[df_all['class']==c]['score'].mean())
print(res)
print('done')
##%%