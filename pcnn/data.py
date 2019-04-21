import numpy as np
from pcnn.myconfig import Config
import re
from pcnn import datautils
import pandas as pd


class Data(object):
    '''need
        句子id 所有单词转化成id
        ent1pos 实体1的位置
        ent2pos 实体2的位置
        相对位置1
        相对位置2
    '''

    def __init__(self,file_path,vec_txt=None,vec_dict=None):

        #测试or训练文件的路径
        self.file_path=file_path
        #词到向量txt文件
        self.vec_txt=vec_txt
        #词到向量npz文件
        self.vec_dict=vec_dict
        #dict
        self.dict=None
        self.length=None
        self.config=Config()

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
                print("add:"+str(self.length))

        return self.length

    def __iter__(self):
        if self.config.USE_SEM:
            for i,line in enumerate(pd.read_csv(self.file_path).values):
                # 如果超出最大迭代次数,退出读取
                if (Config.max_iter != None and i >= Config.max_iter):
                    break

                # 替换semevaltask部分
                if line[4]==0:
                    entity_1 = line[1]
                    entity_2 = line[2]
                else:
                    entity_2 = line[1]
                    entity_1 = line[2]

                try:
                    relation_id = int(line[5])
                except Exception as e:
                    print(e)
                    #print(relation_id)
                sentence = line[7]

                # 转化成id
                sentence_id = []
                pos1 = []
                pos2 = []

                try:
                    # 获取实体相对位置
                    ent1_index = sentence.index(entity_1)
                    ent2_index = sentence.index(entity_2)
                except:
                    continue
                for idx, word in enumerate(sentence):
                    position1 = datautils.pos_constrain(idx - ent1_index)
                    position2 = datautils.pos_constrain(idx - ent2_index)
                    # 将单词转化成id
                    word = self.config.processing_word(word)
                    sentence_id.append(word)
                    pos1.append(position1)
                    pos2.append(position2)
                # 确保三段长度一致
                assert len(sentence_id) == len(pos1) == len(pos2)

                yield sentence_id, pos1, pos2, (ent1_index, ent2_index), relation_id
        else:
            with open(self.file_path) as file:
                for i,line in enumerate(file):
                    #如果超出最大迭代次数,退出读取
                    if(Config.max_iter!=None and i>=Config.max_iter):
                        break
                    #解析文本内容
                    line = re.sub('###END###', '</s>', line)
                    line = line.strip()
                    line_sp=line.split()
                    #读取属性值,替换semevaltask部分
                    entity_1=line_sp[2]
                    entity_2=line_sp[3]
                    relation=line_sp[4]
                    sentence=line_sp[5:]

                    #转化成id
                    sentence_id=[]
                    pos1=[]
                    pos2=[]

                    try:
                        #将relation从文字转化成id
                        relation_id=self.config.processing_relation(relation)
                    except Exception as e:
                        print(e)
                        print("关系查询出错")
                        continue
                    try:
                        #获取实体相对位置
                        ent1_index=sentence.index(entity_1)
                        ent2_index=sentence.index(entity_2)
                    except :
                        continue
                    for idx, word in enumerate(sentence):
                        position1 = datautils.pos_constrain(idx - ent1_index)
                        position2 = datautils.pos_constrain(idx - ent2_index)
                        #将单词转化成id
                        word = self.config.processing_word(word)
                        sentence_id.append(word)
                        pos1.append(position1)
                        pos2.append(position2)
                    #确保三段长度一致
                    assert len(sentence_id)==len(pos1)==len(pos2)

                    yield sentence_id,pos1,pos2,(ent1_index,ent2_index),relation_id

