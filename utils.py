import csv
import numpy as np
import matplotlib.pyplot as plt

def unifyPaths(IMGpath, path, isCustom):
    splitChar = '\\' if isCustom else '/'

    imgName = IMGpath.split(splitChar)[-1]
    return path + 'IMG/' + imgName

def importInputData(pathToCollectedData, isCustom):
    
    dropRate = 0.5
    correction = 0.2
    sampleList = []

    with open(pathToCollectedData + 'driving_log.csv') as csvfile:
        fieldnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'] if  isCustom else None
        reader = csv.DictReader(csvfile, fieldnames = fieldnames)
        for line in reader:
            sample = {}
            sample['image'] = (unifyPaths(line['center'], pathToCollectedData, isCustom))
            sample['steering']  = float(line['steering'])
            sampleList.append(sample)

            # if float(line['steering']) == 0.0 and random.random() < dropRate:
            #     pass
            # else:
            sample = {}
            sample['image'] = (unifyPaths(line['left'], pathToCollectedData, isCustom))
            sample['steering']  = float(line['steering']) + correction
            sampleList.append(sample)

            sample = {}
            sample['image'] = (unifyPaths(line['right'], pathToCollectedData, isCustom))
            sample['steering']  = float(line['steering']) - correction
            sampleList.append(sample)

    return sampleList


def trimZeroSteeringSamples(samples):
    trimmedSamples =[]
    for sample in samples:
        if sample['steering'] == 0.0 and random.random() < 0.5:
            pass
        else:
            trimmedSamples.append(sample)
    return trimmedSamples


def plotHistogram(data):

    labels = []
    for item in data:
        labels.append(item['steering'])
    #print (labels)
    bins = np.arange(start = -1.0, stop = 1.05, step = 0.05)
    bins = np.concatenate([bins[:20], [-0.001,0.001], bins[21:]])

    hist_vals, _ ,_ = plt.hist(labels, bins = bins)
    plt.show()

    max_hist_val = hist_vals.max()
    max_hist_idx = hist_vals.argmax()
