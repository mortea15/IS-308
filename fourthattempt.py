# https://github.com/mouradmourafiq/tensorflow-lstm-regression/blob/master/lstm_weather.ipynb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.contrib import learn

from data_processing import load_csvdata
from lstm import lstm_model

LOG_DIR = './ops_logs/predictor'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100


def load_data_frame(filename):
    # load the weather data and make a date
    data_raw = pd.read_csv(filename, dtype={'Date': str})
    data_raw['NumStudents'] = data_raw['NumStudents'].astype(tf.cast(tf.float32))
    years = []
    for index, row in data_raw.iterrows():
        _d = row['Year']
        years.append(_d)

    data_raw['_time'] = pd.Series(years, index=data_raw.index)
    df = pd.DataFrame(data_raw, columns=['_year', 'NumStudents'])
    return df.set_index('_year')


# scale values to reasonable values and convert to float
data_opptak = load_data_frame("opptaksdata.csv")
X, y = load_csvdata(data_opptak, TIMESTEPS, seperate=False)

regressor = learn.SKCompat(learn.Estimator(
    model_fn=lstm_model(
        TIMESTEPS,
        RNN_LAYERS,
        DENSE_LAYERS
    ),
    model_dir=LOG_DIR
))

# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)
regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])

# not used in this example but used for seeing deviations
rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))

score = mean_squared_error(predicted, y['test'])
print("MSE: %f" % score)

# plot the data
all_dates = data_opptak.index.get_values()

fig, ax = plt.subplots(1)
fig.autofmt_xdate()

predicted_values = predicted.flatten()  # already subset
predicted_dates = all_dates[len(all_dates) - len(predicted_values):len(all_dates)]
predicted_series = pd.Series(predicted_values, index=predicted_dates)
plot_predicted, = ax.plot(predicted_series, label='predicted (c)')

test_values = y['test'].flatten()
test_dates = all_dates[len(all_dates) - len(test_values):len(all_dates)]
test_series = pd.Series(test_values, index=test_dates)
plot_test, = ax.plot(test_series, label='2015 (c)')

xfmt = mdates.DateFormatter('%b %d %H')
ax.xaxis.set_major_formatter(xfmt)

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('PDX Weather Predictions for 2016 vs 2015')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
