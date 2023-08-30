import os
import pandas as pd
import openai
import re
import json

##  steps for data preparing, finetuning, getting inference etc. from the finetuned model.

openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = openai.api_key

# NLTK（www.nltk.org）是在处理预料库、分类文本、分析语言结构等多项操作中最长遇到的包。其收集的大量公开数据集、模型上提供了全面、易用的接口，涵盖了分词、词性标注(Part-Of-Speech tag, POS-tag)、命名实体识别(Named Entity Recognition, NER)、句法分析(Syntactic Parse)等各项 NLP 领域的功能。
# 原文链接：https://blog.csdn.net/justlpf/article/details/121707391

import nltk
# 如果需要更新这个，打开comment
#nltk.download('stopwords')
from nltk.corpus import stopwords
stpwrds = set(stopwords.words('english'))

# open("data.json", "w").close()
def clean_text(text):
    new_text = [word for word in text.split(" ") if word not in stpwrds]
    return " ".join(new_text)

# 这个数据是从哪里来的?
data = pd.read_csv("cleaned_data.csv")
data.dropna(inplace = True)
data = data.drop_duplicates()

class FinetuningChatGPT:
    def Init(self):

        pass

    def TryPrompt(self):
        completion = openai.Completion()
        question = "can i preorder a playstation"
        prom = f'\nCustomer: {question}\nAgent:'
        print(prom)
        # print(prom)
        response = completion.create(
            model="GPT-Turbo3.5",
            prompt=prom, stop=["\nCustomer"], temperature=0.3,
            top_p=1, best_of=1,
            max_tokens=150
        )
        # print(response)
        print(response.choices[0].text.strip())

    def HandleData(self):
        test = []
        old_id = 0
        d = dict()
        i = 0
        with open("data.json", 'a') as f:
            while (i <= data.shape[0]):
                new_id = data.loc[i]["ID"]
                st = "{\"prompt\": \""
                new_data = data[data["ID"] == new_id]
                prom = clean_text(new_data["question"].values[0])
                st = "{\"prompt\": \"" + prom + "\\n\\n###\\n"
                for ids in range(new_data.shape[0]):
                    if ids == 0:
                        st += "\\nCustomer: " + new_data.iloc[ids]["question"] + "\\nAgent: "
                        st += "\""
                        st += "," + "\"completion\"" + ":" + "\" " + new_data.iloc[ids]["response"] + "\\n\""
                        st += "}"
                        st += "\n"
                        f.write(st)
                    else:
                        st = "{\"prompt\": \"" + prom + "\\n\\n###\\n"
                        for sub_ids in range(ids + 1):
                            if sub_ids == 0:
                                st += "\\nCustomer: " + new_data.iloc[sub_ids]["question"] + "\\nAgent: " + \
                                      new_data.iloc[sub_ids]["response"] + "\\n"
                            elif sub_ids != ids:
                                st += "Customer: " + new_data.iloc[sub_ids]["question"] + "\\nAgent: " + new_data.iloc[sub_ids][
                                    "response"] + "\\n"
                            else:
                                st += "Customer: " + new_data.iloc[ids]["question"] + "\\nAgent: "
                                st += "\""
                                st += "," + "\"completion\"" + ":" + "\" " + new_data.iloc[ids]["response"] + "\\n\""
                                st += "}"
                                st += "\n"
                                f.write(st)
                    i += 1



# !openai tools fine_tunes.prepare_data -f "data.json"
#!openai api fine_tunes.create -t "data_prepared.jsonl" -m "ada"

