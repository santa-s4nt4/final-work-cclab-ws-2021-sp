# %%
import pandas as pd
import MeCab
import time

# %%
data = pd.read_csv(
    "/home/santanaruse/dev/PyTorch_workshop_2021/final_work/tweet/tweets.csv", engine="python")
data["text"] = data["text"].astype(str)

# %%
data = data[~data["text"].str.contains('@')]
data = data[~data["text"].str.contains('#')]
data = data[~data["text"].str.contains('http')]
data = data[~data["text"].str.contains('RT')]

# %%
pd.DataFrame(data).to_csv(
    '/home/santanaruse/dev/PyTorch_workshop_2021/final_work/tweet/tweets_text.txt', index=False, header=None)

# %%
input_file = '/home/santanaruse/dev/PyTorch_workshop_2021/final_work/tweet/tweets_text.txt'

f = open(input_file, 'r')
text_data = f.read()
mecab = MeCab.Tagger("-Owakati")
text = mecab.parse(text_data)
mecab.parse('')

timestr = time.strftime('%Y%m%d-%H%M%S')

out_file_name = "/home/santanaruse/dev/PyTorch_workshop_2021/final_work/tweet/wakachi_" + timestr + ".txt"
with open(out_file_name, 'w') as f:
    f.write(text)


# %%

m = MeCab.Tagger("-Owakati")
timestr = time.strftime('%Y%m%d-%H%M%S')
for line in open('/home/santanaruse/dev/PyTorch_workshop_2021/final_work/tweet_data/tweets_text.txt', 'r'):
    words = m.parse(line)
    words = words.rstrip('\n')
    print(words)

out_file_name = "/home/santanaruse/dev/PyTorch_workshop_2021/final_work/tweet_data/wakachi_" + timestr + ".txt"
with open(out_file_name, 'w') as f:
    f.write(words)

# %%
