import os.path
import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import detail_predict
import basic_analysis

def get_stock_datas(codes):
    # 获取当前日期和时间
    now = datetime.now()
    # 将当前日期格式化为"YYYYMMDD"
    date_one_year_ago = now - timedelta(days=365)
    startdate=date_one_year_ago.strftime("%Y%m%d")
    date_str = now.strftime("%Y%m%d")
    print('当前日期：',date_str)
    #爬取每只股票一年前至今的数据
    for i in  codes:
        get_stock_data(i, startdate, date_str, 'data\\')



def get_stock_data(code, date1, date2, filename, length=-1):
	pro=ts.pro_api()
	df = pro.daily(ts_code=code, start_date=date1, end_date=date2)
	df1 = pd.DataFrame(df)

	path = code + '_1.csv'
	df1.to_csv(os.path.join(filename, path), index=False) #保存原始副本

	df1 = df1[['trade_date','open', 'high', 'low', 'close', 'vol', 'pct_chg']]
	df1 = df1.sort_values(by='trade_date')
	trade_date_column = df['trade_date']
	df1['trade_date'] = trade_date_column.apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))

	print('共有%s天数据' % len(df1))
	if len(df1) >= length:
		path = code + '.csv'
		df1.to_csv(os.path.join(filename, path),index=False)
		print(path)
	else:
		print('无数据')  # 没有数据则输出显示提示
        
def all_lstm(codes):
    df_list=[]
    acys=[]
    for i in codes:
        df,acy=detail_predict.lstm_detail(i)
        df_list.append(df)
        acys.append(acy)
    return df_list,acys


def show_pre_data_close(codes,acys=[],isAcy=False):
    #codes = ['300750.SZ', '000001.SZ']  # 用逗号分隔的代码列表
    dataframes = []
    for code in codes:
        df = pd.read_csv(f'pre2_data\\{code}.csv')
        dataframes.append(df)


    # 设置figsize以确保子图能够适应3x3的网格
    fig, axes = plt.subplots(len(dataframes), 1, figsize=(10, 5 * len(dataframes)))
    # 假设每个DataFrame的长度都是20
    datalen = 90
    k=int(datalen*0.07)
    for i, df in enumerate(dataframes):
        close = df['close'].iloc[-datalen:]
        date = df['date'].iloc[-datalen:].tolist()
        date=[i[5:] for i in date]
        # 绘制前10个数据点的折线图，颜色为蓝色
        axes[i].plot(range(1, datalen - 10 + 1), close[:datalen - 10], color='blue')
        # 绘制后10个数据点的折线图，颜色为绿色
        axes[i].plot(range(datalen - 10, datalen + 1), close[datalen - 10 - 1:], color='green')
        # 设置x轴的刻度，从1到20
        x_range_future = np.arange(len(date))
        axes[i].set_xticks(x_range_future[::k], date[::k])
        # 设置子图的标题
        if isAcy:
            axes[i].set_title(f'{codes[i]},average accuracy: {acys[i]:.3f}')
        else:
            axes[i].set_title(f'{codes[i]}')

    plt.savefig('close_all.png')
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # 显示图表
    plt.show()

def cheak(codes):
    need=[]
    for i in codes:
        df=pd.read_csv(f'data\\{i}.csv')
        if len(df)>90:
           need.append(i)
        else:
            print(i,'股票数据不符合')
    print(need)
    return need

def readSelect():
    with open('selectStock.txt',mode='r',encoding='utf-8') as f:
        selectcodes=[i.strip() for i in f.readlines()]
        return selectcodes

if __name__ == '__main__':
     #codes = readSelect()
     codes = ['000066.SZ', '001339.SZ', '002197.SZ', '002351.SZ', '002528.SZ', '002577.SZ']
     print(codes)
    # 第二步，股票数据获取
    # 爬取每只股票一年前至今的数据
     get_stock_datas(codes)
    # 筛选不符合要求的股票，例子中没有不符合的
     codes=cheak(codes)
    # 股票预测
     df_list,acys=all_lstm(codes)
    # 有精确率需要与预测同时运行
     show_pre_data_close(codes,acys,isAcy=True)

    #show_pre_data_close(codes)

    # 30日滚动平均值曲线
     basic_analysis.roll(codes)
    # 其他基本数据
     basic_analysis.basic(codes)
    # 预测的十日回报率平均值
     basic_analysis.pre_10_return(codes)

     detail_predict.lstm_detail('000066.SZ',simulation_size=5,predictDay=30,isShow=True)


