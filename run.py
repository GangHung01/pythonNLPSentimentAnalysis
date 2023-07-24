from AffectiveAnalysis.Analysis_Sentence import Analysis_Sentence, sim_sen_Result
from AffectiveAnalysis.data_process_show import data_process_show
from AffectiveAnalysis.similarSentence import Similar_Sentence

if __name__ == '__main__':
    #展示饼图，直方图，词云图
    data_process_show()
    #与‘蚊帐放下后落不到地板，容易进蚊子，’相似评论的五个用户
    Similar_Sentence()
    #训练并保存感情分析预测模型
    Analysis_Sentence()
    #进行感情分析
    while True:
        print('请输入评论:(1表退出)')
        sentence = input("请输入评论:")
        if sentence == '1':
            break
        sim_sen_Result(sentence)


