'''
Use fully connected network to classify candlestick patterns as up, down, flat

'''

import datetime
import random
import resource
import os
import matplotlib as mp
import matplotlib.finance as mfinance
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from collections import deque
import numpy as np
import tensorflow as tf

# constants
TICKER = 'SPY'      # the stock we are interested in training for
SLIDING_WINDOW = 50 # days of historical day to predict for...
PREDICT_DAY = 7     # ... trend x days after; this # must < sliding_window
TRAIN_BATCH = 100
TRAIN_ITERATIONS = 100
TREND_NOS = 3       
TREND_FLAT = 0
TREND_UP = 1
TREND_DOWN = 2
OUTPUT_FOLDER = 'Output'
IMG_NAME = 'StockImg'
IMG_EXT = '.png'
IMG_HEIGHT = 1 # inch
IMG_WIDTH = 1 
IMG_DPI = 80
# global var
startdate = datetime.date(2015, 1, 1)
enddate = datetime.date(2016, 1, 31)

# return the current memory usage of the program
def checkMemUsage():
    return str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024) + " MB"

# print a msg with the current timestamp
def printNow(msg):
    print("%s %s." % (msg, datetime.datetime.now()))

# convert the ploted candlestick graph into a numpy 3-D array of RGB
def fig2rgb_array(fig):
    # input: a matplotlib plot figure
    # output: a numpy 3-D array of [y, x, rgb values] 
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    #return np.fromstring(buf, dtype=np.uint8).reshape(ncols, nrows,  3)
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols,  3)

def setPlotfeatures():
    # create a figure and ax instance
    fig, ax = plt.subplots(figsize=(IMG_HEIGHT,IMG_WIDTH),dpi=IMG_DPI)
    # minimize margins and invisible the axes and frames
    plt.tight_layout() 
    ax.set_frame_on(False) 
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    #ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    # Set figure width to 4 and height to 3 with dpi of 80 i.e. 320x240 pixels image
    #fig_size = plt.rcParams["figure.figsize"]
    #dpi = plt.rcParams["figure.dpi"]
    #fig_size[0] = 4.0
    #fig_size[1] = 3.0
    #plt.rcParams["figure.figsize"][0] = 4.0
    #plt.rcParams["figure.figsize"][1] = 3.0
    #plt.rcParams["figure.figsize"] = fig_size
    #plt.rcParams["figure.dpi"] = 80.0 
    #plt.show()
    return fig, ax

def GenTrainingData(fig, ax, DataStore):
    # get quotes from Yahoo
    quotes = mfinance.quotes_historical_yahoo_ohlc(TICKER, startdate, enddate)
    totalquotesday = len(quotes)
    print("total days of data: " + str(totalquotesday))
    # Loop to process the quotes into a sliding window
    #  from which we plot each sliding window into an image
    #  and get the trend of a certain days later
    counter = 0
    #quotesToDraw = []
    quotesOpens = []
    quotesHighs = []
    quotesLows = []
    quotesCloses = []
    trend = np.zeros(TREND_NOS) # one-hot tensor for the trend
    while counter <= (totalquotesday - SLIDING_WINDOW - PREDICT_DAY):
        #del quotesToDraw[:]
        del quotesOpens[:]
        del quotesHighs[:]
        del quotesLows[:]
        del quotesCloses[:]
        for i in range(0, SLIDING_WINDOW):
            #quotesToDraw.append(quotes[counter+i])
            quotesOpens.append(quotes[counter+i][1])
            quotesHighs.append(quotes[counter+i][2])
            quotesLows.append(quotes[counter+i][3])
            quotesCloses.append(quotes[counter+i][4])
        # this version automatically has closing dates in between (follow dates)
        #mfinance.candlestick_ohlc(ax, quotesToDraw, width=0.6) 
        # this version is continuous plot with no closing dates in between
        
        mfinance.candlestick2_ohlc(ax, quotesOpens, quotesHighs, quotesLows, quotesCloses, width=0.6)
        # conver the image plot into a numpy 3D array
        imgArray = fig2rgb_array(fig)
        '''
        print("Using sliding window from day " + str(counter) + " to day " + \
            str(counter+SLIDING_WINDOW-1) + " to predict price on day: " + \
            str(counter+SLIDING_WINDOW+PREDICT_DAY-1))
        print("Closing price on day " + str(counter+SLIDING_WINDOW-1) + " is " + \
            str(quotesCloses[SLIDING_WINDOW-1]))
        print("Closing price on day " + str(counter+SLIDING_WINDOW+PREDICT_DAY-1) + " is " + \
            str(quotes[counter+SLIDING_WINDOW+PREDICT_DAY-1][4]))
        '''
        if quotesCloses[SLIDING_WINDOW-1] == quotes[counter+SLIDING_WINDOW+PREDICT_DAY-1][4]:
            trend[TREND_FLAT] = 1
        if quotesCloses[SLIDING_WINDOW-1] < quotes[counter+SLIDING_WINDOW+PREDICT_DAY-1][4]:
            trend[TREND_UP] = 1
        if quotesCloses[SLIDING_WINDOW-1] > quotes[counter+SLIDING_WINDOW+PREDICT_DAY-1][4]:
            trend[TREND_DOWN] = 1
        #print(trend)
        # then add that 
        DataStore.append((counter, imgArray, trend))
        #mimage.imsave(os.path.join(OUTPUT_FOLDER, IMG_NAME + str(counter) + IMG_EXT), imgArray)
        #plt.imshow(imgArray, interpolation='nearest')
        #plt.show()
        plt.cla()
        print(checkMemUsage())
        trend = [0,0,0]
        counter += 1
    # clear the whole plot release memory 
    plt.clf()

# get a samples from the DataStore to train
def getTrainingBatch(DataStore):
    minibatch = random.sample(DataStore, TRAIN_BATCH)
    batch_key = [d[0] for d in minibatch]
    batch_imgarray = [d[1] for d in minibatch]
    batch_trend = [d[2] for d in minibatch]
    return batch_key, batch_imgarray, batch_trend
    
# Common CNN operations
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    
    
def TrainNetwork(DataStore):
    '''
    x = tf.placeholder(tf.float32, [None, 240, 240, 3])
    y = x*2 #dummy operations for now
    with tf.Session() as session:
        result = session.run(y, feed_dict={x: imgArray})
        print(result)    
    '''
    # input layer
    '''
    3 conv layers + 1 fully connected layer
    input: [None, 80, 80, 3]  
    conv layer 1: 
     in: [80x80x3] 
     kernel: 4x4x3x32, stride 4
     out: [20, 20, 32]
     max pooling 2x2: [10, 10, 32]
    conv layer 2:
     in: [10, 10, 32]
     kernel: 2x2x32x64, stride 2
    out: [5,5,64]
     max pooling 2x2: [3,3,64]
    fully connected layer 1:
     in: [2,2,64]
     reshape: [256x1]
     relu: [256x1]
    output:
     in: [256x]
     matrix multiplier: [3x1]
    '''    
    x = tf.placeholder("float", [None, 80, 80, 3])
    y_ = tf.placeholder(tf.float32, [None, 3])
    # hidden layer 1
    W_conv1 = weight_variable([4, 4, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # hidden layer 2
    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_conv2_flat = tf.reshape(h_conv2, [-1, 1600])
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, 3])
    b_fc2 = bias_variable([3])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    # define the cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(readout))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    #   launch a training session
    sess = tf.Session()
    sess.run(init)
    counter = 0
    while counter < TRAIN_ITERATIONS:
        batch_key, batch_imgarray, batch_trend = getTrainingBatch(DataStore)
        for i in range(TRAIN_BATCH):
            sess.run(train_step, feed_dict={x: batch_imgarray, y_: batch_trend})
        correct_prediction = tf.equal(tf.argmax(readout,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print(accuracy)
        #   output the result
        print(str(counter) + " " + str(sess.run(accuracy, feed_dict={x: batch_imgarray, y_: batch_trend})))
        counter += 1

    '''
    x = tf.placeholder(tf.float32, [TRAIN_BATCH, 19200])
    W = tf.Variable(tf.zeros([19200, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 3])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    init = tf.initialize_all_variables()
    #   launch a training session
    sess = tf.Session()
    sess.run(init)
    batch_key, batch_imgarray, batch_trend = getTrainingBatch(DataStore)
    for i in range(TRAIN_BATCH):
        print(batch_imgarray[i].shape[0])
        print(batch_imgarray[i].shape[1])
        print(batch_imgarray[i].shape[2])
        bi = np.reshape(batch_imgarray[i], batch_imgarray[i].shape[0]*batch_imgarray[i].shape[1]*batch_imgarray[i].shape[2])
        print(bi.shape[0])
        #batch_imgarray[i] = 
    sess.run(train_step, feed_dict={x: bi, y_: batch_trend})
    #for i in range(TRAIN_BATCH):
        #bi = batch_imgarray[i].reshape()
        #sess.run(train_step, feed_dict={x: bi, y_: batch_trend})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accurracy)
    #   output the result
    #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    '''

    
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
    TrainNetwork(DataStore)

    
def main():
    startTime = datetime.datetime.now()
    printNow("Starts: ")
    Run() # main actions
    printNow("Ends: ")
    print("Elaspsed: " + str(datetime.datetime.now() - startTime))

if __name__ == "__main__":
    main()    



