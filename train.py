# -*- coding: utf-8 -*
import sys
#sys.path.append('/dataa2-2t/molamola/SemEvalTask')
from pcnn.model import PCNN_Model
from pcnn.data import Data
from pcnn.myconfig import Config as MyConfig
from pcnn.datautils import shuffle_data


if __name__ == '__main__':
    # 创建配置实例
    my_config = MyConfig()

    # 获得模型
    model = PCNN_Model()
    # 构建模型
    model.build_model()

    if not my_config.USE_SEM:
        #获得训练数据
        train_data=Data(my_config.TRAIN_PATH)
        #获得测试数据
        test_data=Data(my_config.TEST_PATH)
    else:
        train_data=Data(my_config.SEM_TRAIN_DATA)
        test_data=Data(my_config.SEM_TEXT_DATA)

    #训练！
    model.train(train_data,test_data)