from pcnn.myconfig import Config
from pcnn.data import Data
from pcnn.model import PCNN_Model
from pcnn.datautils import pos_constrain ,minibatches

#自主输入句子和实体，判断关系
def open_test_shell(model,config):
    words_total = []
    pos1_total = []
    pos2_total = []
    ent_pos_total = []
    #将命令输入的文字转化成数据
    def get_predict_data(seq,ent1,ent2):

        ent1_pos = seq.index(ent1)
        ent2_pos = seq.index(ent2)
        words, pos1_ids, pos2_ids = [], [], []

        for idx, word in enumerate(seq):
            words.append(word)
            pos1 = pos_constrain(idx - ent1_pos)
            pos1_ids.append(pos1)
            pos2 = pos_constrain(idx - ent2_pos)
            pos2_ids.append(pos2)

        words = [config.processing_word(w) for w in words]

        ent_pos = [ent1_pos, ent2_pos,len(seq)]
        ent_pos.sort()

        words_total.append(words)
        pos1_total.append(pos1_ids)
        pos2_total.append(pos2_ids)
        ent_pos_total.append(ent_pos)

    while True:
        #清空输入集合
        words_total.clear()
        pos1_total.clear()
        pos2_total.clear()
        ent_pos_total.clear()

        sentence=input("输入句子>>")
        ent1=input("输入实体1>>")
        ent2=input("输入实体2>>")

        seq=sentence.split()

        if ent1 not in seq or ent2 not in seq:
            print("实体不在句子中")
            continue
        get_predict_data(seq,ent1,ent2)
        get_predict_data(seq,ent1,ent2)

        #预测
        result=model.predict_batch(words_total,pos1_total,pos2_total,ent_pos_total,[[0]])

        with open(config.filename_relation) as f:
            for id,line in enumerate(f.readlines()):
                if id==result[1]:
                    print(line)


if __name__ == '__main__':
    my_config=Config()

    model=PCNN_Model()
    #加载模型文件
    model.build_model()
    model.restore_session(my_config.dir_model+"/model_epoch19.ckpt")

    #测试数据集
    test_data=Data(my_config.TEST_PATH)
    model.run_evaluate(test_data)

    #打开测试命令
    open_test_shell(model,my_config)
