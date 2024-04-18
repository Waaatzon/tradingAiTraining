
# -- Imports --

# Torch and Sklearn
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Pandas and NumPy
import pandas as pd
import pandas_ta as ta
import numpy as np

# DPG
import dearpygui.dearpygui as dpg

# My imports
from time_series_dataset import TimeSeriesDataset
from lstm import LSTM

# Other
from copy import deepcopy as dc
from scipy.stats import linregress
from backtesting import Strategy, Backtest
from xgboost import XGBClassifier
import time




# LSTM Data
data_LSTM = pd.read_csv('data/US100.cash1.csv')
data_LSTM['datetime'] = pd.to_datetime(data_LSTM['date'] + ' ' + data_LSTM['time'])
data_LSTM = data_LSTM[['datetime', 'close']]

data_KNN = pd.read_csv("data/EURUSD_ecn1440.csv")

data_NN = pd.read_csv("data/EURUSD_ecn1440.csv")
data_NN['Local time'] = pd.to_datetime(data_NN['date'] + ' ' + data_NN['time'])
data_NN.drop(['date', 'time'], axis=1, inplace=True)
data_NN = data_NN[['Local time', 'open', 'high', 'low', 'close', 'volume']]

def show_us100():
    """
    Shows US100 data in a window using graph.

    Parameters:
        None
    Returns:
        None
    """
    data = dc(data_LSTM)
    x = list(range(len(data)))
    y = list(data['close'])
    with dpg.window(label="US100 data",
                    width=800,
                    height=800,
                    pos=(200, 200)):
        with dpg.plot(width=-1,
                        height=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Time")
            with dpg.plot_axis(dpg.mvYAxis, label="Value"):
                dpg.add_line_series(x, y, label="US100")

def show_eurusd():
    """
    Shows EURUSD data in a window using graph.

    Parameters:
        None
    Returns:
        None
    """
    data = dc(data_KNN)
    x = list(range(len(data)))
    y = list(data['close'])
    with dpg.window(label="EURUSD data",
                    width=800,
                    height=800,
                    pos=(200, 200)):
        with dpg.plot(width=-1,
                        height=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Time")
            with dpg.plot_axis(dpg.mvYAxis, label="Value"):
                dpg.add_line_series(x, y, label="EURUSD")

device_items = ["CPU"]
if torch.cuda.is_available():
    device_items.append("GPU")

def prepare_data_for_LSTM(data: pd.DataFrame,
                          steps: int = 7) -> pd.DataFrame:
    """
    Prepares data for the LSTM.
    Makes new dataframe, this dataframe contains datetime, target value and number of values in previous days according to steps parameter.

    Parameters:
        data: pd.DataFrame
        steps: int; Default 7
    Returns:
        pd.DataFrame
    """
    data = dc(data)
    data.set_index('datetime', inplace=True)
    for i in range(1, steps + 1):
        data[f'close-{i}'] = data['close'].shift(i)
    data.dropna(inplace=True)
    return data

def get_slope_KNN(array: list) -> float:
    """
    Calculates slope of the array, used for averages.

    Parameters:
        array: list
    Returns:
        float
    """
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope

def target_trend_KNN(barsupfront: int,
                     df1: pd.DataFrame,
                     pipdiff: float,
                     SLTPRatio: float) -> list:
    """
    Calculates the target trend.

    Parameters:
        barsupfront: int
        df1: pd.DataFrame
        pipdiff: float
        SLTPRatio: float
    Returns:
        list
    """
    length = len(df1)
    high = list(df1['high'])
    low = list(df1['low'])
    close = list(df1['close'])
    open = list(df1['open'])
    trendcat = [None] * length
    
    for line in range (0,length-barsupfront-2):
        valueOpenLow = 0
        valueOpenHigh = 0
        for i in range(1,barsupfront+2):
            value1 = open[line+1]-low[line+i]
            value2 = open[line+1]-high[line+i]
            valueOpenLow = max(value1, valueOpenLow)
            valueOpenHigh = min(value2, valueOpenHigh)

            if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                trendcat[line] = 1 #-1 downtrend
                break
            elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
                trendcat[line] = 2 # uptrend
                break
            else:
                trendcat[line] = 0 # no clear trend
            
    return trendcat

def train_part_LSTM(epoch: int,
                    model: torch.nn.Module,
                    loss_fn: torch.nn.Module,
                    optimizer: torch.optim,
                    train_dataloader: torch.utils.data.DataLoader,
                    device: str) -> None:
    """
    Training loop for LSTM model.

    Parameters:
        epoch: int,
        model: torch.nn.Module
        loss_fn: torch.nn.Module
        optimizer: torch.optim
        train_dataloader: torch.utils.data.DataLoader
    Returns:
        None
    """
    model.train()
    if dpg.does_item_exist("LSTM_console"):
        dpg.add_text(f"\nEpoch: {epoch}", parent="LSTM_console")
        dpg.set_y_scroll("LSTM_console", dpg.get_y_scroll_max("LSTM_console")+500)
    running_loss = 0
    for batch_i, batch in enumerate(train_dataloader):
        x_batch = batch[0]
        y_batch = batch[1]

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        y_preds = model(x_batch)
        loss = loss_fn(y_preds, y_batch)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_i % 100 == 0:
            true_loss = running_loss / 100
            if dpg.does_item_exist("LSTM_console"):
                dpg.add_text(f"Batch: {batch_i}, Loss: {true_loss:.3f}", parent="LSTM_console")
                dpg.set_y_scroll("LSTM_console", dpg.get_y_scroll_max("LSTM_console")+500)
            running_loss = 0


def test_part_LSTM(model: torch.nn.Module,
                   loss_fn: torch,
                   test_dataloader: torch.utils.data.DataLoader,
                   device: str) -> None:
    """
    Testing loop for LSTM model.

    Parameters:
        model: torch.nn.Module
        loss_fn: torch.nn.Module
        test_dataloader: torch.utils.data.DataLoader
    Returns:
        None
    """
    model.eval()
    running_loss = 0
    for batch_i, batch in enumerate(test_dataloader):
        x_batch = batch[0]
        y_batch = batch[1]

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.inference_mode():
            y_preds = model(x_batch)
            loss = loss_fn(y_preds, y_batch)
            running_loss += loss
    average_loss = running_loss / len(test_dataloader)
    if dpg.does_item_exist("LSTM_console"):
        dpg.add_text(f"Average loss across batches: {average_loss:.3f}", parent="LSTM_console")
        dpg.set_y_scroll("LSTM_console", dpg.get_y_scroll_max("LSTM_console")+500)

def train_LSTM() -> None:
    """
    Basic training function for LSTM model.

    Parameters:
        None
    Returns:
        None
    """
    dpg.disable_item("LSTM_train_button")
    if dpg.does_item_exist("LSTM_learning_process"):
        dpg.delete_item("LSTM_learning_process")
    if dpg.does_item_exist("LSTM_result_graph"):
        dpg.delete_item("LSTM_result_graph")
    with dpg.window(label="LSTM learning process",
                    no_close=True,
                    no_collapse=True,
                    no_resize=True,
                    width=620,
                    height=480,
                    pos=(200, 200),
                    tag="LSTM_learning_process"):
        dpg.add_progress_bar(default_value=0.0,
                                overlay="Training",
                                width=600,
                                height=20,
                                tag="LSTM_progress_bar",
                                pos=(10, 25))
        with dpg.child_window(width=600,
                            height=400,
                            border=True,
                            pos=(10, 55),
                            tag="LSTM_console"):
            dpg.add_text("Training started")


    data=dc(data_LSTM)
    last_data_split_from_input = len(data) - dpg.get_value("LSTM_range_input")
    device = dpg.get_value("LSTM_device_input")
    if device == "GPU":
        device = "cuda"
    else:
        device = "cpu"

    data = data[last_data_split_from_input:]


    past_data_from_input = dpg.get_value("LSTM_past_input")
    prepared_data = prepare_data_for_LSTM(data=data,
                                          steps=past_data_from_input)
    prepared_data = prepared_data.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaled_data = scaler.fit_transform(prepared_data)

    X = rescaled_data[:, 1:]
    X = dc(np.flip(X, axis=1))
    y = rescaled_data[:, 0]


    ratio_from_input = dpg.get_value("LSTM_ratio_input") / 100
    split = int(len(X) * ratio_from_input)

    X_train = X[:split].reshape((-1, past_data_from_input, 1))
    X_test = X[split:].reshape((-1, past_data_from_input, 1))

    y_train = y[:split].reshape((-1, 1))
    y_test = y[split:].reshape((-1, 1))


    X_train = torch.tensor(X_train).to(torch.float32)
    X_test = torch.tensor(X_test).to(torch.float32)
    y_train = torch.tensor(y_train).to(torch.float32)
    y_test = torch.tensor(y_test).to(torch.float32)

    train_dataset = TimeSeriesDataset(X=X_train, y=y_train)
    test_dataset = TimeSeriesDataset(X=X_test, y=y_test)

    batch_size_from_input = dpg.get_value("LSTM_batch_input")
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size_from_input,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size_from_input,
                                 shuffle=False)
    start_training_time = time.time()
    model = LSTM(1, 4, 1, device)
    model.to(device)

    lr_from_input = dpg.get_value("LSTM_lr_input")
    epochs_from_input = dpg.get_value("LSTM_epochs_input")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr_from_input)

    for epoch in range(epochs_from_input):
        train_part_LSTM(epoch=epoch,
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader,
                        device=device)

        test_part_LSTM(model=model,
                       loss_fn=loss_fn,
                       test_dataloader=test_dataloader,
                       device=device)
        
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
    dummies = np.zeros((X_test.shape[0], past_data_from_input+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])
    test_predictions = list(test_predictions.tolist())

    dummies = np.zeros((X_test.shape[0], past_data_from_input+1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    y_test_new = dc(dummies[:, 0])
    y_test_new = list(y_test_new.tolist())
    
    x = list(range(len(y_test_new)))

    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    dpg.add_text(f"\nTraining time: {training_time:.2f} seconds on {device}", parent="LSTM_console")
    dpg.set_y_scroll("LSTM_console", dpg.get_y_scroll_max("LSTM_console")+500)


    with dpg.window(label="LSTM result graph",
                    width=800,
                    height=800,
                    pos=(200, 200),
                    tag="LSTM_result_graph"):
        with dpg.plot(width=-1,
                      height=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Time")
            with dpg.plot_axis(dpg.mvYAxis, label="Value"):
                dpg.add_line_series(x, y_test_new, label="True values")
                dpg.add_line_series(x, test_predictions, label="Predicted values")
    
    dpg.enable_item("LSTM_train_button")
    dpg.configure_item("LSTM_learning_process", no_close=False)


def train_KNN() -> None:
    """
    Training function for KNN model.

    Parameters:
        None
    Returns:
        None
    """
    dpg.set_value("KNN_status", "Training started")
    dpg.disable_item("KNN_train_button")
    data = dc(data_KNN)
    data['ATR'] = data.ta.atr(length=20)
    data['RSI'] = data.ta.rsi()
    data['Average'] = data.ta.midprice(length=1)
    data['MA40'] = data.ta.sma(length=40)
    data['MA80'] = data.ta.sma(length=80)
    data['MA160'] = data.ta.sma(length=160)

    backroll = 6
    data['slopeMA40'] = data['MA40'].rolling(window=backroll).apply(get_slope_KNN, raw=True)
    data['slopeMA80'] = data['MA80'].rolling(window=backroll).apply(get_slope_KNN, raw=True)
    data['slopeMA160'] = data['MA160'].rolling(window=backroll).apply(get_slope_KNN, raw=True)
    data['AverageSlope'] = data['Average'].rolling(window=backroll).apply(get_slope_KNN, raw=True)
    data['RSISlope'] = data['RSI'].rolling(window=backroll).apply(get_slope_KNN, raw=True)

    pipdiff = 500*1e-5
    SLTPRatio = 2
    data['Trend'] = target_trend_KNN(16, data, pipdiff, SLTPRatio)
    data_model= data[['volume', 'ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160',
                      'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope', 'Trend']]
    data_model = data_model.dropna()

    attributes=['ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope']
    X = data_model[attributes]
    y = data_model["Trend"]
    
    train_index = int(0.8 * len(X))
    X_train, X_test = X[:train_index], X[train_index:]
    y_train, y_test = y[:train_index], y[train_index:]

    neighbors = dpg.get_value("KNN_neighbours_input")
    model = KNeighborsClassifier(n_neighbors=neighbors,
                                 weights='uniform',
                                 algorithm='kd_tree',
                                 leaf_size=30,
                                 p=1,
                                 metric='minkowski',
                                 metric_params=None,
                                 n_jobs=1)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    training_result = f"""
Accuracy train: {accuracy_train * 100.0:.2f}%\n
Accuracy test: {accuracy_test * 100.0:.2f}%\n
Trend distribution:\n
    Stand: {data_model['Trend'].value_counts()[0]*100/data_model['Trend'].count():.2f}%\n
    Up: {data_model['Trend'].value_counts()[2]*100/data_model['Trend'].count():.2f}%\n
    Down: {data_model['Trend'].value_counts()[1]*100/data_model['Trend'].count():.2f}%\n
Gambler: 33.33%\n
    """

    dpg.set_value("KNN_status", training_result)
    dpg.enable_item("KNN_train_button")

def train_NN() -> None:
    """
    Training function for Neural Network model.

    Parameters:
        None
    Returns:
        None
    """
    dpg.set_value("NN_status", "Training started")
    dpg.disable_item("NN_train_button")
    def calc_signal() -> pd.Series:
        """Returns signal value of data.\n\nParameters:\n\tNone\nReturns:\n\tpd.Series"""
        return data.signal
    class StratForNN(Strategy):  
        def init(self):
            super().init()
            self.signal = self.I(calc_signal)

        def next(self):
            super().next() 
            if self.signal==2:
                sl1 = self.data.Close[-1] - 600e-4
                tp1 = self.data.Close[-1] + 450e-4
                self.buy(sl=sl1, tp=tp1)
            elif self.signal==1:
                sl1 = self.data.Close[-1] + 600e-4
                tp1 = self.data.Close[-1] - 450e-4
                self.sell(sl=sl1, tp=tp1)

    data = dc(data_NN)
    data=data[data['volume']!=0]
    data.reset_index(drop=True, inplace=True)

    def support(df1: pd.DataFrame,
                l: int,
                n1: int,
                n2: int) -> int:
        """
        Calculates supports.

        Parameters:
            df1: pd.DataFrame
            l: int
            n1: int
            n2: int
        Returns:
            int
        """
        for i in range(l-n1+1, l+1):
            if(df1.low[i]>df1.low[i-1]):
                return 0
        for i in range(l+1,l+n2+1):
            if(df1.low[i]<df1.low[i-1]):
                return 0
        return 1

    def resistance(df1: pd.DataFrame,
                   l: int,
                   n1: int,
                   n2: int) -> int:
        """
        Calculates resistances.

        Parameters:
            df1: pd.DataFrame
            l: int
            n1: int
            n2: int
        Returns:
            int
        """
        for i in range(l-n1+1, l+1):
            if(df1.high[i]<df1.high[i-1]):
                return 0
        for i in range(l+1,l+n2+1):
            if(df1.high[i]>df1.high[i-1]):
                return 0
        return 1
    

    def isEngulfing(l: int) -> int:
        """
        Checkes if trend is engulfing.

        Parameters:
            l: int
        Returns:
            int
        """
        row=l
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<0.000001:
            bodydiff[row]=0.000001      

        bodydiffmin = 0.002
        if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            open[row-1]<close[row-1] and
            open[row]>close[row] and 
            (open[row]-close[row-1])>=-0e-5 and close[row]<open[row-1]): #+0e-5 -5e-5
            return 1

        elif(bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            open[row-1]>close[row-1] and
            open[row]<close[row] and 
            (open[row]-close[row-1])<=+0e-5 and close[row]>open[row-1]):#-0e-5 +5e-5
            return 2
        else:
            return 0
        
    def isStar(l: int) -> int:
        """
        CHeckes for shooting stars (pinbars).

        Parameters:
            l: int
        Returns:
            int
        """
        bodydiffmin = 0.0020
        row=l
        highdiff[row] = high[row]-max(open[row],close[row])
        lowdiff[row] = min(open[row],close[row])-low[row]
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<0.000001:
            bodydiff[row]=0.000001
        ratio1[row] = highdiff[row]/bodydiff[row]
        ratio2[row] = lowdiff[row]/bodydiff[row]

        if (ratio1[row]>1 and lowdiff[row]<0.2*highdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]>close[row]):
            return 1
        elif (ratio2[row]>1 and highdiff[row]<0.2*lowdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]<close[row]):
            return 2
        else:
            return 0
        
    def closeResistance(l,levels,lim):
        if len(levels)==0:
            return 0
        c1 = abs(data.high[l]-min(levels, key=lambda x:abs(x-data.high[l])))<=lim
        c2 = abs(max(data.open[l],data.close[l])-min(levels, key=lambda x:abs(x-data.high[l])))<=lim
        c3 = min(data.open[l],data.close[l])<min(levels, key=lambda x:abs(x-data.high[l]))
        c4 = data.low[l]<min(levels, key=lambda x:abs(x-data.high[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1
        else:
            return 0
        
    def closeSupport(l,levels,lim):
        if len(levels)==0:
            return 0
        c1 = abs(data.low[l]-min(levels, key=lambda x:abs(x-data.low[l])))<=lim
        c2 = abs(min(data.open[l],data.close[l])-min(levels, key=lambda x:abs(x-data.low[l])))<=lim
        c3 = max(data.open[l],data.close[l])>min(levels, key=lambda x:abs(x-data.low[l]))
        c4 = data.high[l]>min(levels, key=lambda x:abs(x-data.low[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1
        else:
            return 0
    
    pipdiff = 250*1e-4
    SLTPRatio = 1
    def mytarget(barsupfront, df1):
        length = len(df1)
        high = list(df1['High'])
        low = list(df1['Low'])
        close = list(df1['Close'])
        open = list(df1['Open'])
        trendcat = [None] * length
        for line in range (0,length-barsupfront-2):
            valueOpenLow = 0
            valueOpenHigh = 0
            for i in range(1,barsupfront+2):
                value1 = open[line+1]-low[line+i]
                value2 = open[line+1]-high[line+i]
                valueOpenLow = max(value1, valueOpenLow)
                valueOpenHigh = min(value2, valueOpenHigh)
                if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ): # down
                    trendcat[line] = 1
                    break
                elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ): # up
                    trendcat[line] = 2
                    break
                else: # stay
                    trendcat[line] = 0
                
        return trendcat

    length = len(data)
    high = list(data['high'])
    low = list(data['low'])
    close = list(data['close'])
    open = list(data['open'])
    bodydiff = [0] * length

    highdiff = [0] * length
    lowdiff = [0] * length
    ratio1 = [0] * length
    ratio2 = [0] * length

    n1=2
    n2=2
    backCandles=30
    signal = [0] * length

    for row in range(backCandles, len(data)-n2):
        ss = []
        rr = []
        for subrow in range(row-backCandles+n1, row+1):
            if support(data, subrow, n1, n2):
                ss.append(data.low[subrow])
            if resistance(data, subrow, n1, n2):
                rr.append(data.high[subrow])

        if ((isEngulfing(row)==1 or isStar(row)==1) and closeResistance(row, rr, 150e-5) ):
            signal[row] = 1
        elif((isEngulfing(row)==2 or isStar(row)==2) and closeSupport(row, ss, 150e-5)):
            signal[row] = 2
        else:
            signal[row] = 0

    data['signal']=signal
    dpg.set_value("NN_status", "Training started\nSignal calculation completed")

    data.columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal']
    bt = Backtest(data, StratForNN, cash=10_000, commission=.00)
    stat = bt.run()
    data['Target'] = mytarget(30, data)

    data["RSI"] = ta.rsi(data.Close, length=16)
    data.dropna(inplace=True)
    data.reset_index(drop=True,inplace=True)

    attributes = ['RSI', 'signal', 'Target']
    data_model= data[attributes].copy()

    data_model['signal'] = pd.Categorical(data_model['signal'])
    data_Dummies = pd.get_dummies(data_model['signal'], prefix = 'signalcategory')
    data_model= data_model.drop(['signal'], axis=1)
    data_model = pd.concat([data_model, data_Dummies], axis=1)


    attributes = ['RSI', 'signalcategory_0', 'signalcategory_1', 'signalcategory_2']
    X = data_model[attributes]
    y = data_model['Target']

    train_pct_index = int(0.7 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

    xgb_enabled = dpg.get_value("NN_XGBoost_input")
    if xgb_enabled:
        model_type = "XGBoost"
        model = XGBClassifier()
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
    else:
        model_type = "MLP"
        random_seed = dpg.get_value("NN_seed_input")
        layers = dpg.get_value("NN_layers_input")
        layers = layers.split()
        layers_feed = []
        for i in layers:
            if not i.isdigit():
                dpg.set_value("NN_layers_input", "20 20 40 40 20")
                layers_feed = [20, 20, 40, 40, 20]
                break
            layers_feed.append(int(i))

        NN = MLPClassifier(hidden_layer_sizes=layers_feed,
                            random_state=random_seed,
                            verbose=0,
                            max_iter=1000,
                            activation='relu')
        NN.fit(X_train, y_train)
        pred_train = NN.predict(X_train)
        pred_test = NN.predict(X_test)
        
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)

    matrix_train = confusion_matrix(y_train, pred_train)
    matrix_test = confusion_matrix(y_test, pred_test)

    report_train = classification_report(y_train, pred_train)
    report_test = classification_report(y_test, pred_test)

    result_text = f"Training completed ({model_type})\nAccuracy train: {acc_train * 100.0:.2f}%\nAccuracy test: {acc_test * 100.0:.2f}%\n{matrix_train}\n{matrix_test}\n{report_train}\n{report_test}"

    dpg.set_value("NN_status", result_text)
    dpg.enable_item("NN_train_button")


def xgboost_updated():
    enabled = dpg.get_value("NN_XGBoost_input")
    if enabled:
        dpg.configure_item("NN_layers_input", show=False)
        dpg.configure_item("NN_seed_input", show=False)
        dpg.configure_item("NN_layers_text", show=False)
        dpg.configure_item("NN_seed_text", show=False)
        dpg.configure_item("NN_layers_tooltip", show=False)
        dpg.configure_item("NN_seed_tooltip", show=False)
        dpg.configure_item("NN_advanced_setings", show=True)
    else:
        dpg.configure_item("NN_layers_input", show=True)
        dpg.configure_item("NN_seed_input", show=True)
        dpg.configure_item("NN_layers_text", show=True)
        dpg.configure_item("NN_seed_text", show=True)
        dpg.configure_item("NN_layers_tooltip", show=True)
        dpg.configure_item("NN_seed_tooltip", show=True)
        dpg.configure_item("NN_advanced_setings", show=False)

# GUI Code -------------------
dpg.create_context()
dpg.create_viewport(title='Trading AI')

with dpg.window(label="LSTM",
                width=620,
                height=460,
                no_close=True,
                no_collapse=True,
                no_resize=True,
                pos=(10, 10)):
    # Device selection
    dpg.add_text("Select device",
                 pos=(10, 25))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Select device to train the model on. Default is CPU")
    dpg.add_combo(items=device_items,
                  default_value="CPU",
                  tag="LSTM_device_input",
                  width=600,
                  pos=(10, 45))
    
    # Data range selection
    lenght_of_data_LSTM = len(data_LSTM)
    dpg.add_text("Select range of data",
                 pos=(10, 75))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("How much data to use for model, default is maximum.")
    dpg.add_input_int(min_value=100,
                      max_value=lenght_of_data_LSTM,
                      min_clamped=True,
                      max_clamped=True,
                      default_value=lenght_of_data_LSTM,
                      tag="LSTM_range_input",
                      width=600,
                      pos=(10, 95))
    
    # Training ratio
    dpg.add_text("Select training / testing ratio",
                 pos=(10, 125))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Ratio between the training data and testing data. Defaul is 80% (80 for training, 20 for testing)")
    dpg.add_drag_int(format="%d%%",
                     min_value=1,
                     max_value=99,
                     default_value=80,
                     tag="LSTM_ratio_input",
                     width=600,
                     pos=(10, 145))
    
    # Batch size
    dpg.add_text("Select batch size",
                 pos=(10, 175))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Batch size, number of data in one batch. Default is 16.")
    dpg.add_input_int(min_value=1,
                      min_clamped=True,
                      default_value=16,
                      tag="LSTM_batch_input",
                      width=600,
                      pos=(10, 195))
    
    # Learning rate
    dpg.add_text("Select learning rate",
                 pos=(10, 225))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Learning rate for the model. Learning rate determines how big steps will the model make to improve itself. Default is 0.001.")
    dpg.add_input_float(format="%.4f",
                        min_value=0.0001,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                        default_value=0.001,
                        tag="LSTM_lr_input",
                        width=600,
                        pos=(10, 245))
    
    # Epochs
    dpg.add_text("Select epochs",
                 pos=(10, 275))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Number of epochs. Determines how many times the model will go through the data. Default is 3.")
    dpg.add_input_int(min_value=1,
                      min_clamped=True,
                      default_value=3,
                      tag="LSTM_epochs_input",
                      width=600,
                      pos=(10, 295))
    
    # Time into past
    dpg.add_text("Select time into past",
                 pos=(10, 325))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Number of data values into the past (how much data will it take to calculation). Default is 7.")
    dpg.add_input_int(min_value=1,
                      max_value=100,
                      min_clamped=True,
                      max_clamped=True,
                      default_value=7,
                      tag="LSTM_past_input",
                      width=600,
                      pos=(10, 345))
    
    # Train button
    dpg.add_button(label="Train",
                   callback=train_LSTM,
                   width=600,
                   height=50,
                   pos=(10, 385),
                   tag="LSTM_train_button")
    
with dpg.window(label="KNN",
                width=620,
                height=460,
                no_close=True,
                no_collapse=True,
                no_resize=True,
                pos=(640, 10)):
    # Naighbours selection
    dpg.add_text("Select number of neighbours",
                 pos=(10, 25))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Number of neighbours for the KNN model. Default is 200.")
    dpg.add_input_int(min_value=1,
                      min_clamped=True,
                      default_value=200,
                      max_value=1000,
                      tag="KNN_neighbours_input",
                      width=600,
                      pos=(10, 45))
    # Train button
    dpg.add_button(label="Train",
                   callback=train_KNN,
                   width=600,
                   height=50,
                   pos=(10, 85),
                   tag="KNN_train_button")
    dpg.add_text("",
                 pos=(10, 135),
                 tag="KNN_status")
    
with dpg.window(label="NN",
                width=620,
                height=680,
                no_close=True,
                no_collapse=True,
                no_resize=True,
                pos=(1270, 10)):
    # Training ratio
    dpg.add_text("Select training / testing ratio",
                 pos=(10, 25))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Ratio between the training data and testing data. Defaul is 70% (70 for training, 30 for testing)")
    dpg.add_drag_int(format="%d%%",
                     min_value=1,
                     max_value=99,
                     default_value=70,
                     tag="NN_ratio_input",
                     width=600,
                     pos=(10, 45))
    
    # Enable XGBoost
    dpg.add_text("Enable XGBoost",
                 pos=(10, 75))
    with dpg.tooltip(dpg.last_item()):
        dpg.add_text("Enable XGBoost model. Default is False.")
    dpg.add_checkbox(label="XGBoost",
                    default_value=False,
                    tag="NN_XGBoost_input",
                    pos=(10, 95),
                    callback=xgboost_updated)
    
    # Layers
    dpg.add_text("Complexity of layers",
                 pos=(10, 125),
                 tag="NN_layers_text")
    with dpg.tooltip(dpg.last_item(),
                     tag="NN_layers_tooltip"):
        dpg.add_text("Layers and their sizes. Default is 20 20 40 40 20.")
    dpg.add_input_text(default_value="20 20 40 40 20",
                       tag="NN_layers_input",
                       width=600,
                       pos=(10, 145))

    # Seed
    dpg.add_text("Seed",
                 pos=(10, 175),
                 tag="NN_seed_text")
    with dpg.tooltip(dpg.last_item(),
                     tag="NN_seed_tooltip"):
        dpg.add_text("Seed for the model. Default is 100.")
    dpg.add_input_int(min_value=1,
                      min_clamped=True,
                      default_value=100,
                      tag="NN_seed_input",
                      width=600,
                      pos=(10, 195))
    
    dpg.add_text("Disable XGBoost to enable layers and seed",
                 show=False,
                 tag="NN_advanced_setings",
                 pos=(10, 140))

    # Train button
    dpg.add_button(label="Train",
                   callback=train_NN,
                   width=600,
                   height=50,
                   pos=(10, 235),
                   tag="NN_train_button")
    
    # Status
    dpg.add_text("",
                 pos=(10, 285),
                 tag="NN_status")
    
with dpg.window(label="Data window",
                width=220,
                height=140,
                no_close=True,
                no_collapse=True,
                no_resize=True,
                pos=(10, 480)):
    dpg.add_text("US100 data (LSTM)",
                 pos=(10, 25))
    dpg.add_button(label="US100",
                   callback=show_us100,
                   width=200,
                   height=30,
                   pos=(10, 45))
    dpg.add_text("EURUSD data (KNN and NN)",
                 pos=(10, 80))
    dpg.add_button(label="EURUSD",
                   callback=show_eurusd,
                   width=200,
                   height=30,
                   pos=(10, 100))

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.maximize_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
