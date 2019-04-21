from pcnn.general_utils import get_logger
import numpy as np
from pcnn.datautils import load_vocab, get_processing_word

class Config(object):

    #是否使用SEM数据集
    USE_SEM = False
    #是否使用多实例学习
    USE_MIL = True

    def __init__(self):
        self.load()

    def load(self):
        """加载这些用于转化文本到id的文件模型
        """
        # 1. 词库
        self.vocab_words     = load_vocab(self.filename_words)
        self.vocab_relations = load_vocab(self.filename_relation)

        # 2. get processing functions that map str -> id
        self.processing_word     = get_processing_word(self.vocab_words, UNK = "<UNK>")
        self.processing_relation = get_processing_word(self.vocab_relations, UNK='NA')

    '''配置'''
    '''训练参数'''
    #学习率
    learning_rate=0.01
    #batch_size
    batch_size=100
    # n_epoch 训练多少次
    n_epoch = 40
    # max_iter
    max_iter = None

    '''测试参数'''
    #测试多少组
    test_size=10
    #每一组测试多少案例
    test_batch_size=5000

    '''embbeding层参数'''

    train_word_embeddings = True
    #位置向量的占位
    dim_pos=5
    #位置向量的偏置
    pos_biases=60
    #对长短不一的句子进行补全时的填充
    pad_tok=499

    '''卷积层参数'''
    #每层多少个卷积核
    feature_map=430
    #一维卷积核大小
    kernel_size=3

    '''其他参数'''
    #图的输出 (for tensorboard)
    graph_output = "./graph"
    dropout = 0.5


    # 训练集txt文件
    TRAIN_PATH = "./data/new-train.txt"
    # 测试集　txt文件
    TEST_PATH = "./data/new-test.txt"

    # word2id转化文件
    filename_words = "./data/words.txt"
    filename_relation = "./data/new-relation.txt"

    # semevaltask 数据集
    SEM_TRAIN_DATA = "./data/sem/train.csv"
    SEM_TEXT_DATA = "./data/sem/test.csv"

    #模型保存地址
    dir_model="./result"
    #多少epoch保存一次
    save_epoch=2


    # 词向量数组
    VEC_DICT_PATH="./vec/vec.npy"
    #词向量文本
    VEC_DICT_TXT="./vec/vec.txt"

    path_log =  "./log.txt"
    #模型存储路径
    restore_model = "./result/early_best.ckpt"
    # 创建logger
    logger = get_logger(path_log)

    '''type_count 分类有多少种
        total_num 训练集总个数
    '''
    if USE_SEM:
        type_count=10
        total_num=8000
    else:
        type_count=9
        total_num=1043586


