'''
Use fully connected network to classify candlestick patterns as up, down, flat

'''

import datetime
import random
import matplotlib as mp
import matplotlib.finance as mfinance
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from collections import deque
import numpy as np
import tensorflow as tf

# constants
TICKER = 'SPY'      # the stock we are interested in training for
SLIDING_WINDOW = 20 # 20 days of historical day to predict for...
PREDICT_DAY = 7     # ... trend 7 days after; this # must < sliding_window
TRAIN_BATCH = 5
OUTPUT_FOLDER = 'Output'
IMG_NAME = 'StockImg'
IMG_EXT = '.png'
# global var
startdate = datetime.date(2015, 12, 1)
enddate = datetime.date(2016, 1, 31)

#
# convert the ploted candlestick graph into a numpy 3-D array of RGB
def fig2rgb_array(fig):
    # input: a matplotlib plot figure
    # output: a numpy 3-D array of [y, x, rgb values] i.e. 640x480x3
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    #return np.fromstring(buf, dtype=np.uint8).reshape(ncols, nrows,  3)
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols,  3)

def setPlotfeatures():
    fig, ax = plt.subplots()
    # set size of the plot
    fig_size = plt.rcParams["figure.figsize"]
    dpi = plt.rcParams["figure.dpi"]

    # Set figure width to 8 and height to 6 with dpi of 80 i.e. 640x480 pixels image
    fig_size[0] = 8.0
    fig_size[1] = 6.0
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.dpi"] = 80.0 

    # minimize margins and invisible the axes and frames
    plt.tight_layout() 
    ax.set_frame_on(False) 
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    #ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    return fig, ax

def GenTrainingData(fig, ax, DataStore):
    # get quotes from Yahoo
    quotes = mfinance.quotes_historical_yahoo_ohlc(TICKER, startdate, enddate)
    totalquotesday = len(quotes)
    print("total days of data: " + str(totalquotesday))
    # Loop to process the quotes
    counter = 0
    quotesToDraw = []
    quotesOpens = []
    quotesHighs = []
    quotesLows = []
    quotesCloses = []
    while counter <= (totalquotesday - SLIDING_WINDOW):
        del quotesToDraw[:]
        del quotesOpens[:]
        del quotesHighs[:]
        del quotesLows[:]
        del quotesCloses[:]
        for i in range(0, SLIDING_WINDOW):
            quotesToDraw.append(quotes[counter+i])
            quotesOpens.append(quotes[counter+i][1])
            quotesHighs.append(quotes[counter+i][2])
            quotesLows.append(quotes[counter+i][3])
            quotesCloses.append(quotes[counter+i][4])
        # this version automatically has closing dates in between (follow dates)
        #mfinance.candlestick_ohlc(ax, quotesToDraw, width=0.6) 
        # this version is continuous plot with no closing dates in between
        mfinance.candlestick2_ohlc(ax, quotesOpens, quotesHighs, quotesLows, quotesCloses, width=0.6)
        imgArray = fig2rgb_array(fig)
        DataStore.append((counter, imgArray))
        #TrainNetwork(imgArray)
        #mimage.imsave(IMG_NAME + str(counter) + IMG_EXT, imgArray)
        plt.cla()
        counter += 1
    #plt.imshow(imgArray, interpolation='nearest')
    #plt.show()
    
    
def TrainNetwork(imgArray):
    x = tf.placeholder(tf.float32, [640, 480, 3])
    y = x+1 #dummy operations for now
    with tf.Session() as session:
        result = session.run(y, feed_dict={x: imgArray})
        print(result)    
    
def Run():
    # Set up memory store for the training data (in the form of a seq of
    #  (index 0: the sequence of the data
    #   index 1: the image of the plot
    #   index 2: the one-hot tensor of the trend)
    DataStore = deque()
    # Set plot figures
    fig, ax = setPlotfeatures()
    # Get the training data
    GenTrainingData(fig, ax, DataStore)

    # sample a minibatch to train on
    minibatch = random.sample(DataStore, TRAIN_BATCH)
    y_batch = [d[0] for d in minibatch]
    x_batch = [d[1] for d in minibatch]
    print(y_batch)
    #print(x_batch)

def main():
    Run()

if __name__ == "__main__":
    main()    



