import tensorflow as tf
import os
from pcnn.datautils import minibatches,pad_sequences,shuffle_data,to_piece ,to_bags
import numpy as np
from pcnn.myconfig import Config
from pcnn.general_utils import Progbar ,get_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PCNN_Model(object):
    def __init__(self):
        self.config=Config()
        # 创建logger
        self.logger = get_logger(self.config.path_log)

    def add_placeholder(self):
        '''定义所需的placeholder'''
        self.word_ids_left = tf.placeholder(tf.int32, shape=[None, None])
        self.word_ids_mid = tf.placeholder(tf.int32, shape=[None, None])
        self.word_ids_right = tf.placeholder(tf.int32, shape=[None, None])

        #左子句的位置向量
        self.pos_l_1 = tf.placeholder(tf.int32, shape=[None, None])
        self.pos_l_2 = tf.placeholder(tf.int32, shape=[None, None])

        #中子句的位置向量
        self.pos_m_1 = tf.placeholder(tf.int32, shape=[None, None])
        self.pos_m_2 = tf.placeholder(tf.int32, shape=[None, None])

        #右子句的位置向量
        self.pos_r_1 = tf.placeholder(tf.int32, shape=[None, None])
        self.pos_r_2 = tf.placeholder(tf.int32, shape=[None, None])

        #输出的标签值
        self.label = tf.placeholder(tf.int32,shape=[None,1])

        # 超参数
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
    def get_feed_dict(self,words_id,pos1,pos2,entpos,relation,lr=None,dr=None):

        #切分句子为左中右三个部分
        width=self.config.kernel_size-1
        words_id_left_set, words_id_mid_set, words_id_right_set = to_piece(words_id, entpos, width)
        pos1_left_set, pos1_mid_set, pos1_right_set = to_piece(pos1, entpos, width)
        pos2_left_set, pos2_mid_set, pos2_right_set = to_piece(pos2, entpos, width)

        assert len(words_id_left_set)==len(pos1_left_set)==len(pos2_left_set)
        assert len(words_id_mid_set) == len(pos1_mid_set) == len(pos2_mid_set)
        assert len(words_id_right_set) == len(pos1_right_set) == len(pos2_right_set)
        #print(pad_sequences(words_id_left_set))
        #增加relation维度
        relation=np.expand_dims(relation,axis=1)

        feed_dict={
            self.word_ids_left:pad_sequences(words_id_left_set),
            self.word_ids_mid:pad_sequences(words_id_mid_set),
            self.word_ids_right:pad_sequences(words_id_right_set),

            self.pos_l_1:pad_sequences(pos1_left_set,pad_tok=self.config.pad_tok),
            self.pos_m_1:pad_sequences(pos1_mid_set,pad_tok=self.config.pad_tok),
            self.pos_r_1:pad_sequences(pos1_right_set,pad_tok=self.config.pad_tok),

            self.pos_l_2: pad_sequences(pos2_left_set,pad_tok=self.config.pad_tok),
            self.pos_m_2: pad_sequences(pos2_mid_set,pad_tok=self.config.pad_tok),
            self.pos_r_2: pad_sequences(pos2_right_set,pad_tok=self.config.pad_tok),

            self.label:np.asarray(relation).reshape((-1, 1))
        }
        if lr is not None:
            feed_dict[self.learning_rate]=lr
        else:
            feed_dict[self.learning_rate]=0.001

        if dr is not None:
            feed_dict[self.dropout]=dr
        else:
            feed_dict[self.dropout]=1
        return feed_dict

    #初始化得嵌入层
    def get_sentence_emb(self,word_ids,pos_1,pos_2):
        with tf.variable_scope("words", reuse=tf.AUTO_REUSE):
            #定义word嵌入层
            _word_embeddings = tf.Variable(
                np.load(self.config.VEC_DICT_PATH),
                dtype=tf.float32,
                trainable=self.config.train_word_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,word_ids)
            self.word_emb=word_embeddings

        #定义pos1嵌入层
        with tf.variable_scope("pos1", reuse=tf.AUTO_REUSE):
            _pos1_embeddings = tf.get_variable(
                dtype=tf.float32,
                shape=[500, self.config.dim_pos],name="pos1_emb")
            pos1_embeddings=tf.nn.embedding_lookup(_pos1_embeddings,pos_1)
            self.pos1_emb = pos1_embeddings

        #定义pos2嵌入层
        with tf.variable_scope("pos2", reuse=tf.AUTO_REUSE):
            _pos2_embeddings = tf.get_variable(
                dtype=tf.float32,
                shape=[500, self.config.dim_pos],name="pos2_emb")
            pos2_embeddings = tf.nn.embedding_lookup(_pos2_embeddings, pos_2)
            self.pos2_emb = pos2_embeddings

        word_emb_shape = word_embeddings.get_shape().as_list()
        pos1_emb_shape = pos1_embeddings.get_shape().as_list()
        pos2_emb_shape = pos2_embeddings.get_shape().as_list()

        #print(word_emb_shape)
        #print(pos1_emb_shape)
        #print(pos2_emb_shape)
        sentence_embeddings = tf.concat([word_embeddings, \
                                         pos1_embeddings, pos2_embeddings], 2)
        #增加一个维度
        # (batch_size, batch中最长句的长度, 句中单词向量表示, dimension, 1)
        #[None, None, 60, 1]
        sentence_embeddings=tf.expand_dims(sentence_embeddings,-1)

        #print(sentence_embeddings.get_shape().as_list())
        return sentence_embeddings

    def add_convolution_op(self, sentence_embeddings):
        """定义卷积层和最大池化层
        输入:嵌入层
        输出:最大池化过的层
        """
        '''定义卷积层
        filter:卷积核个数
        kernel_size:卷积核大小
        '''
        with tf.variable_scope("cov1", reuse=tf.AUTO_REUSE) as  scope:
            _conv = tf.layers.conv2d(
                inputs=sentence_embeddings,
                filters=self.config.feature_map,
                kernel_size=[3, 50+2*self.config.dim_pos],
                strides=(1, 50+2*self.config.dim_pos),
                padding="same",
                name=scope.name
            )

        _conv_shape = _conv.get_shape().as_list()
        #print("卷积层大小:"+str(_conv_shape))
        assert _conv_shape[2] == 1
        sen_emb_shape = sentence_embeddings.get_shape().as_list()
        conv = tf.squeeze(_conv, [2])
        #定义池化层
        maxpool = tf.reduce_max(conv, axis=1, keepdims=True)
        maxpool_shape = maxpool.get_shape().as_list()

        assert maxpool_shape[1] == 1
        maxpool = tf.squeeze(maxpool)
        # shape = (batch_size, feature_maps, 1)
        maxpool = tf.expand_dims(maxpool, -1)

        return maxpool

    def Piece_Wise_CNN(self):
        '''定义pcnn网络结构'''
        #右句向量
        emb_right=self.get_sentence_emb(self.word_ids_right,self.pos_r_1,self.pos_r_2)
        #中句向量
        emb_mid=self.get_sentence_emb(self.word_ids_mid,self.pos_m_1,self.pos_m_2)
        #左句向量
        emb_left=self.get_sentence_emb(self.word_ids_left,self.pos_l_1,self.pos_l_2)
        #最大池化输出
        # shape = (batch_size, feature_maps, 1)
        max_pool_out_left=self.add_convolution_op(emb_left)
        max_pool_out_mid = self.add_convolution_op(emb_mid)
        max_pool_out_right = self.add_convolution_op(emb_right)

        #连接所有的池化层
        # shape = (batch_size, feature_maps, 3)
        _maxpool = tf.concat([max_pool_out_left, max_pool_out_mid, max_pool_out_right], 2)
        #全连接
        # shape = (batch_size, 900)
        maxpool_flat = tf.reshape(_maxpool, [-1, 3 * self.config.feature_map])
        #激活函数
        # shape = (batch_size, 900)
        _gvector = tf.tanh(maxpool_flat)
        #输入dropout层
        #shape (batch_size ,900)
        self.gvector=tf.nn.dropout(_gvector,self.dropout)

    def add_pred_op(self):
        """Defines self.logits and self.relations_pred
        获得每一种关系的预测可能性
        获得最后预测的种类
        """
        #连接最后的权值和偏置
        with tf.variable_scope("proj"):
            W1 = tf.get_variable("W1", dtype=tf.float32,
                    shape=[3*self.config.feature_map, self.config.type_count])

            b = tf.get_variable("b", dtype=tf.float32,
                    shape=[self.config.type_count], initializer=tf.zeros_initializer())
        #与最大池化层运算,得到53个预测值
        pred = tf.matmul(self.gvector, W1) + b

        # shape = (batch_size, nrelations)
        self.logits = tf.reshape(pred, [-1, self.config.type_count])


    def add_loss_op(self):
        """定义损失函数"""
        # 使用softmax分类器
        # shape = (batch_size, nrelations))
        self.softmax_tensor = tf.nn.softmax(self.logits)
        # shape = (batch_size, 1)　求出概率最大的那一个数
        relations_pred = tf.argmax(self.softmax_tensor, axis=1)
        # shape = (batch_size, )　求出其下标
        self.relations_pred = tf.reshape(relations_pred, [-1])

        # 扩充维数，将（5，）扩充为（5,1），里面的内容不变
        batch_size = tf.size(self.label)
        # 扩充维数。由于batch_size=(5, )
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        # 将indices和labels在第二维连接
        concated = tf.concat([indices, self.label], 1)
        self.onehot_labels = tf.sparse_to_dense(
            concated, tf.stack([batch_size, self.config.type_count]), 1.0, 0.0)

        #交叉熵
        cross_entropy=-tf.reduce_sum(self.onehot_labels* tf.log(self.softmax_tensor), reduction_indices=[1])
        self.loss=tf.reduce_mean(cross_entropy)

        '''losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits, labels=self.label)
        self.loss = tf.reduce_mean(losses)'''
        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def train_op(self):
        #损失值
        loss = self.loss
        #学习率
        learning_rate=self.learning_rate
        #随机梯度下降
        with tf.variable_scope("train_step"):
            self.trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    #for tensorboard
    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        print(self.merged)
        self.file_writer = tf.summary.FileWriter(self.config.graph_output, self.sess.graph)

    def run_epoch(self,train,test,epoch_num):
        '''最小训练单位'''
        #一次跑多少数据
        batch_size=self.config.batch_size
        # 在每个epoch内跑多少个batch_size
        nbatches = (self.config.total_num + batch_size - 1) // batch_size
        print("nbatches:{}".format(nbatches))
        #nbatches=1043586

        print('*' * 200)
        #定义进度条
        #prog = Progbar(target=nbatches)

        for i ,data in enumerate(minibatches(train,batch_size)):
            #每200个batch测试准确度
            if (i%200)==1 and i is not 1:
                self.run_evaluate(test)

            #若使用多示例学习
            if self.config.USE_MIL:
                #print("正在转化成bag...")
                word_ids, pos1_ids, pos2_ids, entpos, relation_id = [], [], [], [], []
                #将batch数据分成多个bag,每个bag为一种关系
                word_bags, pos1_bags, pos2_bags, pos_bags, y_bags, num_bags = to_bags(data)
                #遍历包
                for j in range(num_bags):
                    if len(word_bags[j])==1:
                        continue
                    #提取每个包中的关系
                    relation = y_bags[j][0]
                    #获得feeddict
                    fd = self.get_feed_dict(word_bags[j], pos1_bags[j], pos2_bags[j], pos_bags[j],[[0]])
                    #获得softmax向量

                    softmax = self.sess.run(self.softmax_tensor, feed_dict=fd)

                    scores = softmax[:, relation]
                    idx = scores.argmax(axis=0)

                    #加入训练数据
                    word_ids.append(word_bags[j][idx])
                    pos1_ids.append(pos1_bags[j][idx])
                    pos2_ids.append(pos2_bags[j][idx])
                    entpos.append(pos_bags[j][idx])
                    relation_id.append(y_bags[j][idx])

            else:
                # 获得数据
                word_ids,  pos1_ids, pos2_ids,entpos, relation_id = data

            '''跑一下'''
            # 获得feeddict
            feed_dict=self.get_feed_dict(word_ids,pos1_ids,pos2_ids,entpos,relation_id,lr=self.config.learning_rate,dr=0.5)

            #训练网络！
            _,loss,summary=self.sess.run([self.trainer,self.loss,self.merged],feed_dict=feed_dict)
            #print(self.sess.run(self.softmax_tensor,feed_dict=feed_dict))
            #print(self.sess.run(self.logits,feed_dict=feed_dict))
            #print(self.sess.run(self.label, feed_dict=feed_dict))
            #print(self.sess.run(self.relations_pred,feed_dict=feed_dict))
            #使用进度条
            #prog.update(i + 1, [("train loss", loss)])
            # 打印loss
            if (i % 20) == 0 or i==0:
                print("batch:{}".format(i)+"    loss:{}".format(loss))

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch_num * nbatches + i)

    def train(self,train,test):
        self.add_summary()
        '''训练模型－－－－－入口函数'''
        for epoch_num in range(self.config.n_epoch):
            #第几次训练
            print("epoch:"+str(epoch_num))

            #打乱数据集
            shuffle_data(self.config.TRAIN_PATH)
            #训练一次
            self.run_epoch(train,test,epoch_num)

            # 保存模型
            if (epoch_num +1)% self.config.save_epoch == 0 :
                try:
                    self.save_session("/model_epoch_" + str(epoch_num) + ".ckpt")
                except Exception as e:
                    print(str(e) + "保存模型出错")
                else:
                    print("保存模型成功!")

    def log_trainable(self):
        """打印变量信息
        """
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            self.logger.info("Variable: {}".format(k))
            self.logger.info("Shape: {}".format(v.shape))

    def init_session(self):
        '''config = tf.ConfigProto(device_count={"CPU":12},
                                inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=1)'''
        self.logger.info("Initializing tf session")
        #创建sess
        self.sess = tf.Session()
        #初始化sess
        self.sess.run(tf.global_variables_initializer())
        #创建saver
        self.saver = tf.train.Saver()

    #构建pcnn模型
    def build_model(self):
        #添加占位器
        self.add_placeholder()
        #定义网络结构
        self.Piece_Wise_CNN()
        #预测模块　
        self.add_pred_op()
        #损失模块
        self.add_loss_op()
        #获得训练器
        self.train_op()
        #初始化session
        self.init_session()
        self.log_trainable()

    '''预测模块'''
    def predict_batch(self, word_ids, pos1_ids, pos2_ids, entities,relation_id):
        """
        在训练中对模型进行预测
        """
        #获得feeddict
        fd = self.get_feed_dict(word_ids, pos1_ids, pos2_ids, entities,relation_id,lr=self.config.learning_rate, dr=1.0)
        #跑模型，输出预测向量(batchsize,53)
        relations_pred = self.sess.run(self.relations_pred, feed_dict=fd)
        return relations_pred

    '''返回精度'''
    def run_evaluate(self, test):
        """计算模型在测试集上的表现
        """
        #打乱测试集
        shuffle_data(self.config.TEST_PATH)
        #正确的和预测的标签
        y_true, y_pred = [], []
        #跑batch
        for i,data in enumerate(minibatches(test, self.config.test_batch_size)):
            word_ids, pos1_ids, pos2_ids, entpos,relation_id = data

            #进行一次预测
            relations_pred = self.predict_batch(word_ids, pos1_ids, pos2_ids, entpos,relation_id)

            assert len(relations_pred) == len(relation_id)
            y_true += relation_id
            y_pred += relations_pred.tolist()
            if i==self.config.test_size:
                break

        #print("true:" + str(y_true))
        #print("pre:" + str(y_pred))
        acc = accuracy_score(y_true, y_pred)
        p   = precision_score(y_true, y_pred, average='macro')
        r   = recall_score(y_true, y_pred, average='macro')
        f1  = f1_score(y_true, y_pred, average='macro')

        print({"acc":acc, "p":p, "r":r, "f1":f1})
        return {"acc":acc, "p":p, "r":r, "f1":f1}

    def restore_session(self, dir_model):
        """加载模型到session
        """
        self.logger.info("加载训练好的模型，位置在 {}...".format(dir_model))
        self.saver.restore(self.sess, dir_model)

    def save_session(self, model_name):
        """保存session"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        saved_name = self.config.dir_model + model_name
        self.saver.save(self.sess, saved_name)

    def close_session(self):
        """关闭session"""
        self.sess.close()
if __name__ == '__main__':
    PCNN_Model().build_model()