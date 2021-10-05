import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# sms-spam-corpus.csv
print("Введите название файла:")
filename = str(input())

emails = pd.read_csv(filename, encoding='windows-1252')
email_count = len(emails) # кол-вл всех увед
ham = emails[emails.v1 == "ham"]
ham_email_count = len(ham) # ham
spam = emails[emails.v1 == "spam"]
spam_email_count = len(spam) # spam

# стоп-слова
en_stops = []
with open('stopwords.txt') as f:
    for line in f:
        en_stops.append(line.rstrip('\n'))

# ham без чисел цифр спец символов в одном регистре
char = ['\d+','\W+','_']
ham = ham.v2.replace(char, " ", regex=True)
ham = pd.DataFrame(ham, columns=['v2'])
ham = ham['v2'].str.lower()

ham = pd.DataFrame(ham, columns=['v2'])
arr_ham = ham.to_numpy()
arr_h = []

all_ham = []
dict_ham = []
ham_len = 0
ham_count = 0

# удаляем стоп-слова
for h in arr_ham:
    h1 = str(h)
    ham_len+= len(h1)-4
    lst = h1.split(' ')
    lst[0] = lst[0][2:]
    lst[len(lst) - 1] = lst[len(lst) - 1][:-2]
    k=1
    while(k>0):
        k=0
        for s in lst:
            if s in en_stops:
                lst.remove(s)
                k+=1
    h = ' '.join(lst)
    arr_h.append(h)

    for s in lst:
        if s!="":
            all_ham.append(s)
            if s not in dict_ham:
                dict_ham.append(s)

# распределение по длине
max_ham_string = max(dict_ham, key=len)
ham_len = [0]*len(max_ham_string)

dict_ham_count = [0] * len(dict_ham)
for s in all_ham:
    indx = dict_ham.index(s)
    dict_ham_count[indx]+=1
    ham_len[len(s)-1]+=1

# print(ham_len)
s=0
for n in ham_len:
    s+=n*(ham_len.index(n)+1)

# средняя длина слова
avg_h1 = round(s / len(all_ham),2)
avg_h = round(s / len(all_ham))
# print(avg_h1)

sum_len_h = 0 # длина всех ham уведомлений
for m in arr_h:
    sum_len_h+=len(m)
max_notif_ham = len(max(arr_h, key=len)) # макс длина уведомления
avg_notif_ham = round(sum_len_h / len(arr_h))
avg_notif_ham1 = round((sum_len_h / len(arr_h)),2)

ham_notif = [0]*max_notif_ham
for n in arr_h:
    ham_notif[len(n)-1]+=1

# записываем ham обратно в файл
df_ham = pd.DataFrame(arr_h, columns=['v2'])
df_ham.to_csv('output/ham_after.csv', index=False, encoding='windows-1252')

ham_dictionary = {'Word':dict_ham, 'Count':dict_ham_count}
df_dict_ham = pd.DataFrame(ham_dictionary)
df_dict_ham.to_csv('output/ham_dictionary.csv', index=False, encoding='windows-1252')

# 20 чаще встречаемых слов ham
all_ham_w_c = len(all_ham) # всего слов
ham20 = df_dict_ham.sort_values('Count', ascending=False)
ham20 = ham20.head(20)
ham20_1 = ham20['Word'].tolist()
ham20_2 = ham20['Count'].tolist()
ham20_3 = []
for n in ham20_2:
    ham20_3.append(round((n/all_ham_w_c),3))

# print(all_ham_w_c)
# print(ham20_1)
# print(ham20_2)
# print(ham20_3)

# spam без чисел цифр спец символов в одном регистре
spam = spam.v2.replace(char," ", regex=True)
spam = pd.DataFrame(spam, columns=['v2'])
spam = spam['v2'].str.lower()

spam = pd.DataFrame(spam, columns=['v2'])
arr_spam = spam.to_numpy()
arr_s = []

all_spam = []
dict_spam = []

# удаляем стоп-слова в спаме
for s in arr_spam:
    s1 = str(s)
    lst = s1.split(' ')
    lst[0] = lst[0][2:]
    lst[len(lst) - 1] = lst[len(lst) - 1][:-2]
    k=1
    while(k>0):
        k=0
        for w in lst:
            if w in en_stops:
                lst.remove(w)
                k+=1
    s = ' '.join(lst)
    arr_s.append(s)

    for w in lst:
        if w != "":
            all_spam.append(w)
            if w not in dict_spam:
                dict_spam.append(w)

# распределение по длине spam
max_spam_string = max(dict_spam, key=len)
spam_len = [0]*len(max_spam_string)

dict_spam_count = [0] * len(dict_spam)
for s in all_spam:
    indx = dict_spam.index(s)
    dict_spam_count[indx]+=1
    spam_len[len(s) - 1] += 1

# print(spam_len)
s=0
for n in spam_len:
    s+=n*(spam_len.index(n)+1)

avg_s1 = round(s / len(all_spam),2)
avg_s = round(s / len(all_spam))
# print(avg_s1)

sum_len_s = 0 # длина всех всех spam уведомлений
for m in arr_s:
    sum_len_s+=len(m)
max_notif_spam = len(max(arr_s, key=len)) # макс длина уведомления
avg_notif_spam = round(sum_len_s / len(arr_s))
avg_notif_spam1 = round((sum_len_s / len(arr_s)),2)

spam_notif = [0]*max_notif_spam
for n in arr_s:
    spam_notif[len(n)-1]+=1

# графики распределения длины слов
fig1, ax1 = plt.subplots()

hml = []
# print(all_ham)
for n in ham_len:
    n = round(n / len(all_ham),4)
    hml.append(n)

# print(hml)

x1 = np.arange(1, len(max_ham_string)+1)
# ax1.bar(x1, height=(ham_len))
ax1.bar(x1, height=(hml))
ax1.set_xlabel('Длина слов')
ax1.set_ylabel('Количество')
ax1.set_title('Распределение по длине слов Ham - Средняя длина'+" "+str(avg_h1)+"("+str(avg_h)+")")
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
for x, y in zip(x1, ham_len):
    if (y>0):
        ax1.text(x, y + 0.05, '%d' % y, ha='center', va = 'bottom', rotation=90, fontsize=7)
fig1.set_size_inches(14,10)
fig1.savefig('output/fig1.png', dpi=100)

fig2, ax2 = plt.subplots()
x2 = np.arange(1, len(max_spam_string)+1)
ax2.bar(x2, height=spam_len)
ax2.set_xlabel('Длина слов')
ax2.set_ylabel('Количество')
ax2.set_title('Распределение по длине слов Spam - Средняя длина'+" "+str(avg_s1)+"("+str(avg_s)+")")
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
for x, y in zip(x2, spam_len):
    if (y>0):
        ax2.text(x, y + 0.05, '%d' % y, ha='center', va = 'bottom', rotation=90, fontsize=7)
fig2.set_size_inches(14,10)
fig2.savefig('output/fig2.png', dpi=100)

# графики распределения длины уведомлений
fig3, ax3 = plt.subplots()

x3 = np.arange(1, max_notif_ham+1)
ax3.bar(x3, height=ham_notif)
ax3.set_xlabel('Длина уведомлений')
ax3.set_ylabel('Количество')
ax3.set_title('Распределение по длине уведомлений Ham - Средняя длина'+" "+str(avg_notif_ham1)+"("+str(avg_notif_ham)+")")
for x, y in zip(x3, ham_notif):
    if (y>0):
        ax3.text(x, y + 0.05, '%d' % y, ha='center', va = 'bottom', rotation=90, fontsize=7)
fig3.set_size_inches(40,10)
fig3.savefig('output/fig3.png', dpi=100)
# plt.show()

fig4, ax4 = plt.subplots()

x4 = np.arange(1, max_notif_spam+1)
ax4.bar(x4, height=spam_notif)
ax4.set_xlabel('Длина уведомлений')
ax4.set_ylabel('Количество')
ax4.set_title('Распределение по длине уведомлений Spam - Средняя длина'+" "+str(avg_notif_spam1)+"("+str(avg_notif_spam)+")")
for x, y in zip(x4, spam_notif):
    if (y>0):
        ax4.text(x, y + 0.05, '%d' % y, ha='center', va = 'bottom', rotation=90, fontsize=7)
fig4.set_size_inches(40,10)
fig4.savefig('output/fig4.png', dpi=100)

# записыаем спам-соо обратно в файл
df_spam = pd.DataFrame(arr_s, columns=['v2'])
df_spam.to_csv('output/spam_after.csv', index=False, encoding='windows-1252')

spam_dictionary = {'Word':dict_spam, 'Count':dict_spam_count}
df_dict_spam= pd.DataFrame(spam_dictionary)
df_dict_spam.to_csv('output/spam_dictionary.csv', index=False, encoding='windows-1252')

# 20 чаще встречаемых слов spam
all_spam_w_c = len(all_spam) # всего слов
spam20 = df_dict_spam.sort_values('Count', ascending=False)
spam20 = spam20.head(20)
spam20_1 = spam20['Word'].tolist()
spam20_2 = spam20['Count'].tolist()
spam20_3 = []
for n in spam20_2:
    spam20_3.append(round((n/all_spam_w_c),3))

# print(all_spam_w_c)
# print(spam20_1)
# print(spam20_2)
# print(spam20_3)

# частотный анализ
fig5, ax5 = plt.subplots()

x5 = np.arange(20)
ax5.bar(x5, height=ham20_3)
ax5.set_xlabel('Слова')
ax5.set_ylabel('Частота')
ax5.set_title('Частотный анализ Ham')
ax5.set_xticks(x5)
ax5.set_xticklabels(ham20_1)
fig5.set_size_inches(20,10)
fig5.savefig('output/fig5.png', dpi=100)

fig6, ax6 = plt.subplots()

x6 = np.arange(20)
ax6.bar(x6, height=spam20_3)
ax6.set_xlabel('Слова')
ax6.set_ylabel('Частота')
ax6.set_title('Частотный анализ Spam')
ax6.set_xticks(x6)
ax6.set_xticklabels(spam20_1)
fig6.set_size_inches(20,10)
fig6.savefig('output/fig6.png', dpi=100)

import re

# Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's

# наивный бай. метод
print("Введите сообщение для анализа:")
notif = str(input())
# удаляем все спец символы и цифры
notif = re.sub(r'[^\w\s]+|[\d]+',r'',notif).strip()
notif_ar = notif.split()
# удаляем стоп-слова
k=1
while(k>0):
    k=0
    for s in notif_ar:
        if s in en_stops:
            notif_ar.remove(s)
            k+=1

Pspam=spam_email_count/email_count
for s in notif_ar:
    if s not in dict_spam:
        all_spam_w_c+=1

for s in notif_ar:
    if s in dict_spam:
        i = dict_spam.index(s)
        Pspam*=dict_spam_count[i]/all_spam_w_c
    else:
        Pspam*=(1/all_spam_w_c)

Pham=ham_email_count/email_count
for s in notif_ar:
    if s not in dict_ham:
        all_ham_w_c+=1

for s in notif_ar:
    if s in dict_ham:
        i = dict_spam.index(s)
        Pham*=dict_ham_count[i]/all_ham_w_c
    else:
        Pham*=1/all_ham_w_c

print("Вероятность, что уведомление - Spam:")
print(Pspam)
print("Вероятность, что уведомление - Ham:")
print(Pham)

if (Pspam > Pham):
    print("Вероятней, что уведомление - Spam")
else:
    print("Вероятней, что уведмление - Ham")
