# -*- coding: utf-8 -*
'''分析日志并进行绘图'''
import json
import matplotlib.pyplot as plt
import sys
#匹配各个需要数据集
EPOCH_PART="epoch:"
BATCH_PART="batch:"
LOSS_PART="loss:"
EVA_PART="{'acc'"
'''
精确度:dict
'''
def get_evaluate_data(file):
    '''获得所有的测试信息，以dict结构保存'''
    dict_list=[]
    with open(file) as f:
        for line in f.readlines():

            if EVA_PART in line:
                #字符串转化成json
                _json=json.dumps(line)
                #再转化成字典
                dict=eval(json.loads(_json))
                dict_list.append(dict)

        return dict_list

def get_loss_data(file):
    '''获得loss，以list结构保存'''
    loss_list=[]
    with open(file) as f:
        for line in f.readlines():

            if LOSS_PART in line:

                loss=line[line.index(LOSS_PART)+5:-1]
                loss_list.append(loss)
        return loss_list

def draw_eva_picture(eva_dict_list=None,loss_list=None,test_index=None):
    '''画出evaluate的折线图'''
    # 建立对象
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    if eva_dict_list is not None:
        ax.set_title('evaluate:'+test_index)
        for dict in eva_dict_list:
            #print(dict)
            eva_dict=dict['result']
            #x1轴坐标
            x=range(0,len(eva_dict))
            #y2轴坐标
            y=[e[test_index] for e in eva_dict]

            # 画图
            plt.plot(x, y, 'o-', label=dict['method'],markersize=3)

        plt.legend()
        plt.show()
        return

    if loss_list is not None:
        ax.set_title("loss")
        for loss in loss_list:
            loss_data=loss['result']
            # x1轴坐标
            x = range(0, len(loss_data))

            # y2轴坐标
            y = [float(l) for l in loss_data]

            # 画图
            plt.plot(x, y, 'o-', label=loss['method'],markersize=1)
        plt.legend()
        plt.show()
        return
    print("无输入,自动退出")

if __name__ == '__main__':
    arg=sys.argv[1]

    if arg == "--eva":
        # 输入的命令列表
        command_list = sys.argv[2:-1]
    elif arg == "--loss":
        command_list=sys.argv[2:]

    #选择指标
    test_index=sys.argv[-1]
    if not len(command_list)%2 is 0:
        print("command error:确认文件或标签成对出现")

    eva_dict_list = []
    loss_list = []
    for idx in range(int(len(command_list)/2)):
        #log文件名
        index_1=idx*2
        #log标签名
        index_2=idx*2+1
        file_name=command_list[index_1]
        tag_name=command_list[index_2]

        #获得估计dict
        if arg=="--eva":
            eva_dict=get_evaluate_data(file_name)
            eva_dict_dict = {"method": tag_name, "result": eva_dict}
            eva_dict_list.append(eva_dict_dict)

        elif arg=="--loss":
            loss_data = get_loss_data(file_name)
            loss_dict = {"method": tag_name, "result": loss_data}
            loss_list.append(loss_dict)

    # eva测试结果
    if arg=="--eva":
        draw_eva_picture(eva_dict_list=eva_dict_list,test_index=test_index)

    # loss的测试结果
    elif arg == "--loss":
        draw_eva_picture(loss_list=loss_list)




