import sys
import warnings
# 忽略未指定的警告
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm


# 定义LSTM模型类
class Model:
    def __init__(
            self,
            learning_rate,
            num_layers,
            size,
            size_layer,
            output_size,
            forget_bias=0.1,
    ):
        # 定义LSTM单元
        def lstm_cell(size_layer):
            return tf.compat.v1.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)
        # 创建多层LSTM结构
        rnn_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple=False,
        )
        # 定义输入和输出占位符
        self.X = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, None, size))
        self.Y = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, output_size))

        # 定义丢弃层
        drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            rnn_cells, output_keep_prob=forget_bias
        )
        # 定义隐藏层和输出层
        self.hidden_layer = tf.compat.v1.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.compat.v1.nn.dynamic_rnn(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        self.logits = tf.compat.v1.layers.dense(self.outputs[-1], output_size)
        # 定义成本函数和优化器
        self.cost = tf.compat.v1.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
# 定义LSTM模型的详细功能
def lstm_detail(code, simulation_size=1, predictDay=10, isShow=False):
    tf.compat.v1.reset_default_graph()  # 重置计算图
    # 设置Seaborn样式
    sns.set()
    # 设置TensorFlow的随机种子
    #tf.compat.v1.random.set_random_seed(1234)
    # 读取股票数据
    df = pd.read_csv(f'data/{code}.csv')
    # 创建数据标准化器
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index

    # 转换数据为标准化形式
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = pd.DataFrame(df_log)
    # 定义LSTM模型的参数
    num_layers = 1
    size_layer = 128
    timestamp = 5
    epoch = 100
    dropout_rate = 0.8
    learning_rate = 0.01

    df_train = df_log


    # 禁用TensorFlow的 eager execution
    tf.compat.v1.disable_eager_execution()


    # 计算准确率的函数
    def calculate_accuracy(real, predict):
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    # 平滑信号的函数
    def anchor(signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer
    import os
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 创建LSTM模型实例
    modelnn = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )

    # 预测函数
    def forecast():
        # 重置默认图
        tf.compat.v1.reset_default_graph

        # 创建会话
        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.compat.v1.global_variables_initializer())
        # 原始日期列表
        date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
        # 进度条
        pbar = tqdm(range(epoch), desc='train loop')
        for i in pbar:
            # 初始化隐藏状态
            init_value = np.zeros((1, num_layers * 2 * size_layer))
            # 总损失和总准确率列表
            total_loss, total_acc = [], []
            # 遍历数据批次
            for k in range(0, df_train.shape[0] - 1, timestamp):
                index = min(k + timestamp, df_train.shape[0] - 1)
                batch_x = np.expand_dims(
                    df_train.iloc[k: index, :].values, axis=0
                )
                batch_y = df_train.iloc[k + 1: index + 1, :].values
                # 运行前向和反向传播
                logits, last_state, _, loss = sess.run(
                    [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                    feed_dict={
                        modelnn.X: batch_x,
                        modelnn.Y: batch_y,
                        modelnn.hidden_layer: init_value,
                    },
                )
                init_value = last_state
                # 更新总损失和准确率
                total_loss.append(loss)
                total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
            # 设置进度条标签
            pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))
        # 预测未来数据
        future_day = predictDay

        output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
        output_predict[0] = df_train.iloc[0]
        upper_b = (df_train.shape[0] // timestamp) * timestamp
        init_value = np.zeros((1, num_layers * 2 * size_layer))

        for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict={
                    modelnn.X: np.expand_dims(
                        df_train.iloc[k: k + timestamp], axis=0
                    ),
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[k + 1: k + timestamp + 1] = out_logits
        # 处理剩余数据
        if upper_b != df_train.shape[0]:
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict={
                    modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis=0),
                    modelnn.hidden_layer: init_value,
                },
            )
            output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
            # 更新未来天数
            future_day -= 1
            # 更新日期列表
            date_ori.append(date_ori[-1] + timedelta(days=1))
        # 更新隐藏状态
        init_value = last_state
        # 预测剩余未来数据
        for i in range(future_day):
            o = output_predict[-future_day - timestamp + i:-future_day + i]
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict={
                    modelnn.X: np.expand_dims(o, axis=0),
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[-future_day + i] = out_logits[-1]
            date_ori.append(date_ori[-1] + timedelta(days=1))
        # 逆标准化预测数据
        output_predict = minmax.inverse_transform(output_predict)
        # 平滑预测数据
        deep_future = anchor(output_predict[:, 0], 0.4)
        return deep_future

    # 进行模拟预测
    results = []
    for i in range(simulation_size):
        print(f'{code} simulation {i+1}')
        results.append(forecast())

    # 更新日期列表
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    for i in range(predictDay):
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()

    # 测结果
    accepted_results = []
    for r in results:
            accepted_results.append(r)
    # 计算准确率
    accuracies = [calculate_accuracy(df['close'].values, r[:-predictDay]) for r in accepted_results]

    # 绘图
    if isShow:
        plt.figure(figsize = (15, 5))
        for no, r in enumerate(accepted_results):
            plt.plot(r, label = 'forecast %d'%(no + 1))
        plt.plot(df['close'], label = 'true trend', c = 'black')

        plt.legend()
        plt.title('average accuracy: %.4f' % (np.mean(accuracies)))
        # 设置x轴刻度
        x_range_future = np.arange(len(results[0]))
        plt.xticks(x_range_future[::30], date_ori[::30])

        plt.show()

    accepted_series = pd.Series(accepted_results[0], name='close')
    # 将 date_ori 转换成 Series 并指定列名
    date_series = pd.Series(date_ori, name='date')

    # 将 Series 合并成 DataFrame
    df = pd.concat([date_series, accepted_series], axis=1)

    df.to_csv(f'pre2_data/{code}.csv', index=False)
    return df, np.mean(accuracies)






