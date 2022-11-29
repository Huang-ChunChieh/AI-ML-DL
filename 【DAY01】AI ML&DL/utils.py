import mplfinance as mpf # 畫K線套件
from datetime import datetime
import pandas as pd # 表格處理套件
import matplotlib.pyplot as plt # 畫圖套件
import matplotlib.colors as mcolors # 畫圖塗顏色的設計
import numpy as np

def getStockPrice(code, start_date=None):
    """
    code 字串格式的股票代碼，ex: "2330" 
    start_date 從什麼時間開始抓，"2021-01-03"
    """
    all_stock_data = pd.read_csv('./data/twstock_2018_2021.txt', encoding='cp950')
    all_stock_data = all_stock_data.rename(columns={
        '證券代碼': "code",
        '簡稱': 'name',
        '年月日': 'date',
        '開盤價(元)': 'open',
        '最高價(元)': 'high',
        '最低價(元)': 'low',
        '收盤價(元)': 'close',
        '成交量(千股)': 'volume'
    })
    all_stock_data['code'] = all_stock_data['code'].apply(lambda x: str(x).strip())
    all_stock_data['name'] = all_stock_data['name'].apply(lambda x: str(x).strip())

    target_stock_data = all_stock_data.query('code == @code')
    if not target_stock_data.empty:
        target_stock_data['date'] = target_stock_data['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
        target_stock_data.index = target_stock_data['date'] 
        if not start_date is None:
            target_stock_data = target_stock_data.query(f'date >= "{start_date}"')
        return target_stock_data
    else:
        return target_stock_data
    
    
style = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(
        up='red',
        down='green',
        volume='in'
    ),
    gridstyle="-"
)


def plotStockCandle(
    stock_data,
    subdata_list=[],
    figsize=(40, 10),
    mav=(5, 10, 20),
    volume=True,
    vlines_date =None,
):
    """
    stock_data ： 要畫圖的 pandas dataframe，基本欄位一定要有 open high low close volume
    subdata_list : 想要加在主畫面上的線資料，ex: subdata_list = [line_data_1, line_data_2 ...]
    figsize ： 圖大小，預設 (40, 10)
    mav ： 移動平均線，預設有 (5, 10, 20)
    volume ： 是否呈現成交量，預設 True
    vlines_date ： 給定日期，可以畫垂直線， ex: '2021-01-05'
    """
    subplot_list = [mpf.make_addplot(
        data,
        type='line',
        panel=0
    ) for data in subdata_list]
    
    if not vlines_date is None:
        vlines = dict(vlines=[vlines_date])
        mpf.plot(
            stock_data,
            type='candle',
            figsize=figsize,
            mav=mav,
            volume=volume,
            style=style,
            addplot=subplot_list,
            vlines=vlines
        )
    else:
        mpf.plot(
            stock_data,
            type='candle',
            figsize=figsize,
            mav=mav,
            volume=volume,
            style=style,
            addplot=subplot_list
        )


def plotIrisScatter(the_iris_x, the_iris_y, figsize=(10,8)):
    """
    
    """
    target_set_list = [
        {
            'target_value': 0,
            'color': 'red',
            'marker': 's'
        },
            {
            'target_value': 1,
            'color': 'blue',
            'marker': 'x'
        },
            {
            'target_value': 2,
            'color': 'lightgreen',
            'marker': 'o'
        },
    ]
    target_names = ['setosa', 'versicolor', 'virginica']
    plt.figure(figsize=figsize)
    for target_set in target_set_list:
        target_value =  target_set['target_value']
        plt.scatter(
            the_iris_x[the_iris_y == target_value].iloc[:,0], 
            the_iris_x[the_iris_y == target_value].iloc[:,1], 
            label=target_names[target_value],
            color=target_set['color'], marker=target_set['marker']
        )

    plt.xlabel(the_iris_x.columns[0])
    plt.ylabel(the_iris_x.columns[1])
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plotDecisionRegions(the_train_x, the_train_y, the_test_x, the_test_y, classifier, feature_name_columns, resolution=0.02, figsize=(10,8)):
    all_x = np.vstack((the_train_x, the_test_x))
    all_y = np.hstack((the_train_y, the_test_y))
    y_category = np.unique(all_y)

    # 設定圖示及顏色
    markers = ['s', 'x', 'o']
    colors = ['red', 'blue', 'lightgreen']
    cmap =  mcolors.ListedColormap(colors[:len(y_category)])

    ##　特徵1, 2的最大最小值，+-1 是為了讓邊界出去一點
    x1_min, x1_max = all_x[:, 0].min() - 1, all_x[:, 0].max() + 1
    x2_min, x2_max = all_x[:, 1].min() - 1, all_x[:, 1].max() + 1
    # 產生網狀級的資料
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    # 這裡的classifier要是被訓練過的
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.figure(figsize=figsize)
    # 區域圖
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(y_category):
        plt.scatter(
            x= all_x[all_y == cl, 0], 
            y= all_x[all_y == cl, 1],
            alpha=0.8, 
            c= np.array([cmap(idx)]),
            marker=markers[idx], 
            label=cl
        )

    plt.scatter(the_test_x[:, 0],
                the_test_x[:, 1],
                c='k',
                alpha=1.0,
                linewidths=1,
                marker='o',
                s=25, label='test set')

    plt.xlabel(feature_name_columns[0])
    plt.ylabel(feature_name_columns[1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    pred_y = classifier.predict(the_test_x)

    return pred_y