from __future__ import print_function
import csv
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import cycle
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from pybrain.tools.validation import ModuleValidator, Validator
from sys import stdout


csvfile = open('/Users/CocoZK/Desktop/study/graduate/data/data35.csv', 'wb')
spamwriter = csv.writer(csvfile, delimiter=' ')
data = np.genfromtxt('/Users/CocoZK/Desktop/study/graduate/data/normopen.csv', delimiter=",", dtype=float)
#input area, the lp is to control the how
#  many experiments to do in once and to output the result in one file,
# now suggested to put all small datasets together
# However in terms of test the time for excuting a project, we can't use it, so this time
# It's suggested to be 1
for lp in range(0,1,1):

    kfolds = raw_input("Input a number for the kfolds in cross validation here: ")
    spamwriter.writerow(kfolds)
    trdatf = raw_input("Thfe range of training data here(start) ")
    spamwriter.writerow(trdatf)
    trdats = raw_input("The range of training data here(end) ")
    spamwriter.writerow(trdats)
    tedatf = raw_input("The range of testing data here(start) ")
    spamwriter.writerow(tedatf)
    tedats = raw_input("The range of training data here(end) ")
    spamwriter.writerow(tedats)
    kfolds = int(kfolds)
    trdatf = int(trdatf)
    trdats = int(trdats)
    tedatf = int(tedatf)
    tedats = int(tedats)


    sys.stdout=open("/Users/CocoZK/Desktop/study/graduate/data/data35.csv", "a")
    dat = data[trdatf:trdats]
    num = len(dat)-len(dat)%(kfolds+1)
    dat = dat[0:num]
    cv_data = np.split(dat, kfolds+1)
    eval_err = []
    modnet = []
    hypernet = []
    hypereval = []
    hnum = [1,2,3,4,5,6,7,8,9,10]

    # Building the neural network
    # for thnum in range(hnum):

    for num in hnum:

      #thum=num
       net = buildNetwork(1, num, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True) # 1 input, 5 hidden layer and 1 output

       for i in range(kfolds+1):
         test_data = cv_data[i]
         train_data = []
         for j in range (kfolds+1) :
           train_data.extend(cv_data[j] ) # make the train data continuous data




         print("test///", test_data)
         train_ds = SequentialDataSet(1, 1)
         # The next_sample picks a next number of the begin of the sample
         for sample, next_sample in zip(train_data, cycle(train_data[1:])):
             train_ds.addSample(sample, next_sample)

         test_ds = SequentialDataSet(1, 1)
         # The next_sample picks a next number of the begin of the sample
         for sample, next_sample in zip(test_data, cycle(test_data[1:])):
            test_ds.addSample(sample, next_sample)

         # Training
         trainer = RPropMinusTrainer(net, dataset=train_ds)
         train_errors = []
         # save errors for plotting later
         EPOCHS_PER_CYCLE = 10
         CYCLES = 10
         EPOCHS = EPOCHS_PER_CYCLE * CYCLES

         for a in xrange(CYCLES):
            trainer.trainEpochs(EPOCHS_PER_CYCLE)
            train_errors.append(trainer.testOnData())
            epoch = (a+1) * EPOCHS_PER_CYCLE
            print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
            stdout.flush()

         print("final error for training =", train_errors[-1])
         err_tst = ModuleValidator.validate(Validator.MSE, net, dataset=test_ds)
         eval_err.append(err_tst)
         modnet.append(net)
         print("test_Err", err_tst)

       print(eval_err)
       pmin = eval_err.index(min(eval_err))
       print(pmin)
       net = modnet[pmin]
       hypernet.append(net)
       hypereval.append(min(eval_err))

    hypermin = hypereval.index(min(eval_err))
    net = hypernet[hypermin]
    print("number of hidden layers", hypermin+1)

    #Testing data
    ds = SequentialDataSet(1, 1)
    dat = data[tedatf:tedats]
    # The next_sample picks a next number of the begin of the sample
    for sample, next_sample in zip(dat, cycle(dat[1:])):
        ds.addSample(sample, next_sample)

    print("put into practice:", ModuleValidator.validate(Validator.MSE, net, dataset=ds))

    pred = []

    for sample, target in ds.getSequenceIterator(0):
        pred.append(net.activate(sample))


    b = ds.getSequence(0)
    ax1 = plt.subplot(1,1,1)
    ax1.plot(b[1], label='tar')
    ax1.plot(pred, label='pre')
    ax1.legend(loc=1, ncol=2, shadow=True)
    plt.title('Stock Predicting Result')
    plt.xlabel('Time')
    plt.ylabel('Return of The Stock')
    plt.show()
