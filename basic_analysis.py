import pandas as pd
import matplotlib.pyplot as plt

def roll(codes):
    df_list=[pd.read_csv(f'pre2_data/{i}.csv')for i in codes]
    for p,df in enumerate(df_list):
        df['MA_30'] = df['close'].rolling(window=30).mean()

    plt.figure(figsize=(10, 6))

    # 绘制的30日滚动平均值曲线
    n = 0
    for df in df_list:
        plt.plot(df['MA_30'], label=codes[n])
        n += 1

    # 添加标题、图例和标签
    plt.title('30-Day Rolling Average of Forex Pairs')
    plt.xlabel('Date')
    plt.ylabel('30-Day Rolling Average')
    plt.legend()
    # 显示图表
    plt.show()

    # ------  单个绘制
    n = 0
    for df in df_list:
        plt.plot(df['MA_30'], label=codes[n])
        n += 1

        # 添加标题、图例和标签
        plt.title('30-Day Rolling Average of Forex Pairs')
        plt.xlabel('Date')
        plt.ylabel('30-Day Rolling Average')
        plt.legend()
        # 显示图表
        plt.show()
def basic(codes):
    # 导入库
    import seaborn as sns
    import pandas as pd
    from matplotlib import pyplot as plt

    # 创建一个空的DataFrame，用于存储所有外汇对的数据
    all_data = []

    # 读取每个外汇对的数据并合并到all_data中
    series_list = []
    for i in codes:
        data = pd.read_csv(f'pre2_data/{i}.csv', index_col='date')

        # 计算每个外汇对的日收益率
        returns = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
        data['return']=returns
        all_data.append(data)
        series_list.append(returns)

    result = pd.concat(series_list, axis=1)
    #result.index = pd.read_csv(f'pre2_data/{codes[0]}.csv')['']
    result.columns = codes

    # #第一行无上一行的数据无法计算,所以去掉
    result = result.iloc[1:, :]


   # print(result.head())
    # 对日收益率之间的相关系数
    corr_matrix = result[codes].corr()

    # 打印相关系数矩阵
    print("相关系数矩阵:")
    print(corr_matrix)

    # 制作散点矩阵图，展示不同股票对的日收益率关系，对角线展示KDE密度分布
    sns.set(style="ticks")
    sns.pairplot(result[codes], diag_kind="kde")
    plt.show()


    variance_list = []
    for i_df in all_data:
        # 计算'close'列的最大值和最小值
        min_value = i_df['close'].min()
        max_value = i_df['close'].max()

        # 将'close'列的每个值标准化到0-1之间
        normalized_close = (i_df['close'] - min_value) / (max_value - min_value)
        # 计算标准化后数据的标准差
        std_normalized = normalized_close.std()
        variance_list.append(std_normalized)
    print('标准差')
    print(codes)
    print(variance_list)

    year_return_list=[]
    for i_df in all_data:
        first_close_value = i_df['close'].iloc[0]
        end_close_value = i_df['close'].iloc[-1]
        year_return=(end_close_value-first_close_value)/first_close_value
        year_return_list.append(year_return)
    print('year_return')
    print(year_return_list)

def pre_10_return(codes):
    all_data = []
    for i in codes:
        data = pd.read_csv(f'pre2_data/{i}.csv', index_col='date')
        # 计算每个外汇对的日收益率
        returns = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
        data['return'] = returns
        all_data.append(data)
    return10_list=[]
    for i in all_data:
        return10=i['return'].tail(10).mean()
        return10_list.append(return10)
    print('预测的十日回报率平均值')
    print(return10_list)


