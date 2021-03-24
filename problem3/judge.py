# torch与BERT用于提取留言与回复的语义向量表示
import torch
from bert.tokenization import BertTokenizer
from bert.modeling import BertModel
import math
import numpy as np
import os
import re
import datetime
import xlrd  # 处理Excel表格
import logging

logging.basicConfig(level=logging.INFO)

# hanlp工具用于命名实体识别
import stanza

zh_nlp = stanza.Pipeline('zh')


# 计算两个向量之间的余弦相似度
def cosinesimilarity(vectorx, vectory):
    total = 0
    xsize = 0
    ysize = 0
    for i in range(0, len(vectorx)):
        total = total + float(vectorx[i]) * float(vectory[i])
        xsize = xsize + float(vectorx[i]) * float(vectorx[i])
        ysize = ysize + float(vectory[i]) * float(vectory[i])
    xsize = math.sqrt(xsize)
    ysize = math.sqrt(ysize)
    if ysize == 0:
        return 0
    else:
        return total / xsize / ysize


# Load pre-trained model tokenizer (vocabulary)
"加载预训练模型的分词器、词汇表"
tokenizer = BertTokenizer.from_pretrained('/home/hezoujie/Models/roberta_pytorch')
# tokenizer = BertTokenizer.from_pretrained('/home/hezoujie/Models/bert_uncased_pytorch')

# '加载预训练模型'
model = BertModel.from_pretrained('/home/hezoujie/Models/roberta_pytorch')  # BERT中文模型的路径
# model = BertModel.from_pretrained('/home/hezoujie/Models/bert_uncased_pytorch')  # BERT中文模型的路径
# 模型下载地址https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz


# '''模型设置成评估模式，去除dropout（随机停止更新部分参数）的模块
# 因为在评估时，有可重复的结果是很重要的'''
model.eval()

file = r"附件4.xlsx"
file1 = xlrd.open_workbook(file)
ret1 = file1.sheet_by_index(0)

out = r"06答复意见质量评价结果.txt"  # 在上传的附件中，已经整理为excel文件
out1 = open(out, "w", encoding="utf8")
out1.write("留言编号\t最终得分\t相似度分\t长度分\tNE分\t关键词分\t联系方式分\t时间分\n")

for j in range(1, 2817):
    code = str(ret1.cell(j, 0).value).strip()  # 留言编号
    topic = str(ret1.cell(j, 2).value).strip()
    note = str(ret1.cell(j, 4).value).strip()
    note1 = re.sub("\s", "", note)  # 去除空白字符

    all_txt = topic + note1[:100]  # 留言的字符表示
    # print(all_txt)

    note_string = ""  # h
    for i in all_txt:
        if i not in [" ", "\n", "\t"]:
            note_string = note_string + i + " "
    sent1 = note_string.strip()

    sent1 = "[CLS] " + sent1 + " [SEP]"
    tokenized_text = tokenizer.tokenize(sent1)
    # print(tokenized_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    # print (segments_ids)

    tokens_tensor = torch.tensor([indexed_tokens])

    print(tokens_tensor)

    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        # 当网络中的某一个tensor不需要梯度时，可以使用torch.no_grad()来处理。
        # See the models docstrings for the detail of the inputs
        try:
            outputs = model(tokens_tensor, segments_tensors)
            encoded_layers = outputs[0]  # 第一个元素，是bert模型的最后一层的隐藏状态。
            # print(encoded_layers.size())
            token_vecs = encoded_layers[0]  # 得到一句话里每个词的768维向量的矩阵。
            # print(token_vecs.size())
            sentence_embedding = torch.mean(token_vecs[0], dim=0)
            # print(sentence_embedding.size())
            note_vec = sentence_embedding.numpy().tolist()  # 留言的向量表示，list格式
        except:
            print("提取向量有误")
            continue

    reply = str(ret1.cell(j, 5).value).strip()
    reply1 = re.sub("\s", "", reply)
    reply_string = ""

    for i in reply1[:127]:
        if i not in [" ", "\n", "\t"]:
            reply_string = reply_string + i + " "
    sent2 = reply_string.strip()

    sent2 = "[CLS] " + sent2 + " [SEP]"
    tokenized_text = tokenizer.tokenize(sent2)
    # print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(
        tokenized_text)  # Map the token strings to their vocabulary indeces.
    segments_ids = [1] * len(tokenized_text)
    # print (segments_ids)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        # 当网络中的某一个tensor不需要梯度时，可以使用torch.no_grad()来处理。
        # See the models docstrings for the detail of the inputs
        try:
            outputs = model(tokens_tensor, segments_tensors)
            encoded_layers = outputs[0]  # 第一个元素，是bert模型的最后一层的隐藏状态。
            # print(encoded_layers.size())
            token_vecs = encoded_layers[0]  # 得到一句话里每个词的768维向量的矩阵。
            # print(token_vecs.size())
            sentence_embedding = torch.mean(token_vecs[0], dim=0)
            # print(sentence_embedding.size())
            reply_vec = sentence_embedding.numpy().tolist()  # 留言的向量表示，list格式
        except:
            print("提取向量有误")
            continue

    # print(len(reply_vec))
    cosine_score = cosinesimilarity(reply_vec, note_vec) * 100  # 指标1：得到留言与回复向量余弦相似度

    note_list = list(all_txt)

    all_list = []
    all_list.append(note_list)
    doc = zh_nlp(all_txt)

    note_NE = []  # 留言中识别的命名实体的列表

    for ent in doc.sentences[0].ents:  # 每一个i是一个元组
        if not ent:
            continue
        else:
            if ent.text not in note_NE:
                note_NE.append(ent.text)

    count = 0  # 回复拥有的与留言相同的命名实体个数
    for i in note_NE:
        if i in reply1:
            count = count + 1

    if len(note_NE) == 0:
        NE_scope = 100
    else:
        NE_scope = float(count) / len(note_NE) * 100  # 指标2：回复对留言中命名实体的覆盖率分数

    key_list = ["答复", "依法", "咨询", "收悉", "调整", "保证", "及时", "反映", "办理", "处理", "解决", "通知", "开展",
                "拨打", "详询", "支持", "商议", "改造", "监督", "理解", "督促", "规定", "工作", "建议", "意见", "尽快",
                "核实", "建设", "鉴定", "调查", "研究", "积极", "加强", "力度", "确保", "要求", "论证", "确认", "贯彻", "落实", "查处", "整改", "办理",
                "核查", "执法", "整治", "检查", "指导", "提供", "务必", "跟进", "检测", "按照", "核定", "行动", "查处", "严格", "把关", "保障", "巡查",
                "重视", "规划", "调查", "查明", "部门", "行政", "协议", "处置", "妥善", "请求", "报告", "争取", "按照", "请示", "获批", "受理", "统筹",
                "会商", "建设", "解释", "劝诫", "责令", "热线", "电话", "联系", "打击"]
    # print("关键词个数:", len(key_list))

    key_count = 0  # 回复中关键词的个数

    for i in key_list:
        if i in reply1:
            key_count = key_count + 1

    key_score = float(key_count) / len(key_list) * 100  # 指标3：回复中关键词的覆盖率分数

    length_score = 0  # 指标4：回复的长度
    if len(reply1) < 30:
        length_score = 0
    else:
        length_score = 30 + (len(reply1) - 30) / 10
        if length_score > 100:
            length_score = 100

    if ("0731-" in reply1) or ("0000-" in reply1):
        phone_score = 100
    else:
        phone_score = 0  # 指标5：联系方式的有无

    # 获取留言的发布日期，与得到回复的日期
    release_date = str(ret1.cell(j, 3).value).strip().split(" ")[0]
    reply_date = str(ret1.cell(j, 6).value).strip().split(" ")[0]

    release_date1 = re.sub("/", "-", release_date)
    reply_date1 = re.sub("/", "-", reply_date)

    a1 = release_date1.split("-")
    b1 = reply_date1.split("-")

    if len(a1) < 3 or len(b1) < 3:
        time_score = 0
    else:
        release_year = release_date1.split("-")[0]
        release_month = release_date1.split("-")[1]
        release_day = release_date1.split("-")[2]

        reply_year = reply_date1.split("-")[0]
        reply_month = reply_date1.split("-")[1]
        reply_day = reply_date1.split("-")[2]

        release_y_m_d = datetime.date(int(release_year), int(release_month), int(release_day))
        reply_y_m_d = datetime.date(int(reply_year), int(reply_month), int(reply_day))

        gap_day = int((reply_y_m_d - release_y_m_d).days)

        # 指标6：回复的即时性time_score
        if gap_day <= 14:
            time_score = 100
        else:
            time_score = 0

    final_score = cosine_score * 0.25 + NE_scope * 0.15 + length_score * 0.15 + key_score * 0.15 + phone_score * 0.10 + time_score * 0.20

    if final_score < 0:
        final_score = 0  # 最低分为0分

    key_words = ['已解决', '已处理', '妥善解决', '妥善处理']
    for key in key_words:
        if key in reply:
            key_score += 30
            final_score += 30
            break

    out1.write(str(code) + "\t" + str(final_score) + "\t" + str(cosine_score) + "\t" + str(length_score) + "\t" + str(
        NE_scope) + "\t" + str(key_score) + "\t" + str(phone_score) + "\t" + str(time_score) + "\n")
