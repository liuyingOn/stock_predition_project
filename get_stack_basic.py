import tushare as ts
# 初始化pro接口
pro = ts.pro_api()
# 获取数据
df = pro.stock_basic(exchange='SZSE', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# 保存数据到CSV文件
df.to_csv('stock_basic_data.csv', index=False)