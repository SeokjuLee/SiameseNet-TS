# Seokju Lee, 16.09.27
'''
Visualize log plots
'''
import numpy as np
import pdb
import matplotlib.pyplot as plt
import csv
import argparse
import os


def close_event():
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str)

while 1: 
    log_path_1 = './progress_log_full.csv'
    log_path_2 = './progress_log_summary.csv'
    print '%s' %(log_path_1)

    logFile_1 = open(log_path_1,'r')
    logFile_2 = open(log_path_2,'r')

    out_1 = np.zeros([1, 0])
    out_2 = np.zeros([2, 0])

    idx = -2
    rdr = csv.reader(logFile_1)
    for line in rdr:
        idx = idx + 1
        if idx == -1:
            continue;
        data = line[0].split('\t')
        out_1 = np.concatenate( (out_1, [[np.float(data[0])]]), axis=1 )

    idx = -2
    rdr = csv.reader(logFile_2)
    for line in rdr:
        idx = idx + 1
        if idx == -1:
            continue;
        data = line[0].split('\t')
        out_2 = np.concatenate( (out_2, [[np.float(data[0])],
                                         [np.float(data[1])]]), axis=1 )


    fig = plt.figure(figsize=(8, 8))

    numPlot = 2
    # projName = next(os.walk('.'))[1][0]
    # fig.suptitle(projName, fontsize=10)

    fig.add_subplot(numPlot,1,1)
    plt.plot(out_1[0], label='tr_loss')
    plt.grid()
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.legend(fontsize=9)
    frame = plt.gca()
    # frame.set_ylim([0, 1])

    fig.add_subplot(numPlot,1,2)
    plt.plot(out_2[0], label='tr_loss')
    plt.plot(out_2[1], label='te_loss')
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.legend(fontsize=9)
    frame = plt.gca()
    # frame.set_ylim([0, 1])

    # pdb.set_trace()
    fig.tight_layout()
    fig.savefig('sj_show_plot.png')
    timer = fig.canvas.new_timer(interval = 1000 * 60 * 5) #interval=3000: creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)
    timer.start()

    plt.show()

    # pdb.set_trace()