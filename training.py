from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
from skimage.transform import rotate
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from Utils.nms import nms
from Utils.cbam import cbam_block
seed=888
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
inputSize=30


def pseudo_labeling(final_images, final_samples, k):
    pseudo_y = []
    video_count = 0 
    
    for subject in final_samples:
        for video in subject:
            samples_arr = []
            if (len(video)==0):
                pseudo_y.append([0 for i in range(len(final_images[video_count])-k)]) #Last k frames are ignored
            else:
                pseudo_y_each = [0]*(len(final_images[video_count])-k)
                for ME in video:
                    samples_arr.append(np.arange(ME[0]+1, ME[1]+1))
                for ground_truth_arr in samples_arr: 
                    for index in range(len(pseudo_y_each)):
                        pseudo_arr = np.arange(index, index+k) 
                        # Equivalent to if IoU>0 then y=1, else y=0
                        if (pseudo_y_each[index] < len(np.intersect1d(pseudo_arr, ground_truth_arr))/len(np.union1d(pseudo_arr, ground_truth_arr))):
                            pseudo_y_each[index] = 1 
                pseudo_y.append(pseudo_y_each)
            video_count+=1
    
    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    print('Total frames:', len(pseudo_y))
    return pseudo_y
    
def loso(dataset, pseudo_y, final_images, final_samples, k):
    #To split the dataset by subjects
    y = np.array(pseudo_y)
    videos_len = []
    groupsLabel = y.copy()
    prevIndex = 0
    countVideos = 0
    
    #Get total frames of each video
    for video_index in range(len(final_images)):
      videos_len.append(final_images[video_index].shape[0]-k)
    
    print('Frame Index for each subject:-')
    for video_index in range(len(final_samples)):
      countVideos += len(final_samples[video_index])
      index = sum(videos_len[:countVideos])
      groupsLabel[prevIndex:index] = video_index
      print('Subject', video_index, ':', prevIndex, '->', index)
      prevIndex = index
    
    X = [frame for video in dataset for frame in video]
    print('\nTotal X:', len(X), ', Total y:', len(y))
    return X, y, groupsLabel
            
def shuffling(X, y):
    shuf = list(zip(X, y))
    random.shuffle(shuf)
    X, y = zip(*shuf)
    return list(X), list(y)

def data_augmentation(X, y):
    transformations = {
        0: lambda image: np.fliplr(image),
        1: lambda image: random_noise(image),
        2: lambda image: cv2.GaussianBlur(image, (5,5), 0),
    }
    y1=y.copy()
    for index, label in enumerate(y1):
        if (label==1): #Only augment on expression samples (label=1)
            for augment_type in range(3):
                img_transformed = transformations[augment_type](X[index]).reshape(inputSize,inputSize,3)
                X.append(np.array(img_transformed))
                y.append(1)
    return X, y

def RMES():
    inputsC = layers.Input(shape=(inputSize,inputSize,3))
    inputs1=inputsC[:,:,:,0:1]
    conv1 = layers.Conv2D(3, (3,3), padding='same', activation='relu')(inputs1)
    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv1)
    #pool1 = layers.Dropout(0.2)(pool1)
    # channel 2
    inputs2 = inputsC[:,:,:,1:2] # layers.Input(shape=(inputSize,inputSize,1))
    conv2 = layers.Conv2D(5, (3,3), padding='same', activation='relu')(inputs2)
    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv2)
    #pool2=layers.Dropout(0.2)(pool2)
    # channel 3
    inputs3 = inputsC[:,:,:,2:3] # layers.Input(shape=(inputSize,inputSize,1))
    conv3 = layers.Conv2D(8, (3,3), padding='same', activation='relu')(inputs3)
    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv3)
    #pool3 = layers.Dropout(0.2)(pool3)
    # merge
    merged = layers.Concatenate()([pool1, pool2, pool3])
    # interpretation
    merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(merged)
    flat = layers.Flatten()(merged_pool)
    dense = layers.Dense(400, activation='relu')(flat)
    outputs = layers.Dense(1, activation='linear')(dense)
    #Takes input u,v,s
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    # compile
    sgd = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p, show_plot):
    prev=0
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        preds = []
        gt = []
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        print('Video:', countVideo+videoIndex)
        score_plot = np.array(result[prev:prev+len(dataset[countVideo+videoIndex])]) #Get related frames to each video
        score_plot_agg = score_plot.copy()
        
        #Score aggregation
        for x in range(len(score_plot[k:-k])):
            score_plot_agg[x+k] = score_plot[x:x+2*k].mean()
        score_plot_agg = score_plot_agg[k:-k]
        
        #Plot the result to see the peaks
        #Note for some video the ground truth samples is below frame index 0 due to the effect of aggregation, but no impact to the evaluation
        if(show_plot):
            plt.figure(figsize=(15,4))
            plt.plot(score_plot_agg) 
            plt.xlabel('Frame')
            plt.ylabel('Score')
        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean()) #Moilanen threshold technique
        peaks, _ = find_peaks(score_plot_agg[:,0], height=threshold[0], distance=k)
        if(len(peaks)==0): #Occurs when no peak is detected, simply give a value to pass the exception in mean_average_precision
            preds.append([0, 0, 0, 0, 0, 0]) 
        for peak in peaks:
            preds.append([peak-k, 0, peak+k, 0, 0, 0]) #Extend left and right side of peak by k frames
        for samples in video:
            gt.append([samples[0]-k, 0, samples[1]-k, 0, 0, 0, 0])
            total_gt += 1
            if(show_plot):
                plt.axvline(x=samples[0]-k, color='r')
                plt.axvline(x=samples[1]-k+1, color='r')
                plt.axhline(y=threshold, color='g')
        if(show_plot):
            #plt.show()
            plt.savefig('CASME_sq/output/img_'+str(countVideo+videoIndex)+'.png')
            np.save('CASME_sq/output/arr_'+str(countVideo+videoIndex)+'.npy', score_plot_agg)
        prev += len(dataset[countVideo+videoIndex])
        metric_fn.add(np.array(preds),np.array(gt)) #IoU = 0.5 according to MEGC2020 metrics
    return preds, gt, total_gt

        
def evaluation(preds, gt, total_gt, metric_fn): #Get TP, FP, FN for final evaluation
    TP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = total_gt - TP
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    return TP, FP, FN

def training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset, train, show_plot,
             threshold):
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groupsLabel)
    subject_count = 0
    epochs = 1000
    batch_size = 1000
    total_gt = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    p = threshold
    model = RMES()
    weight_reset = model.get_weights() #Initial weights
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10, restore_best_weights=True)

    for train_index, test_index in logo.split(X, y, groupsLabel): # Leave One Subject Out
        subject_count+=1
        print('Subject : ' + str(subject_count))
        
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index] #Get training set
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index] #Get testing set
        
        print('------Initializing-------') #To reset the model at every LOSO testing
        
        path = 'RMES_Weights/' + dataset_name + '/' + expression_type + '/s' + str(subject_count) + '.hdf5'
        if(train):
            #Downsampling non expression samples the dataset by 1/2 to reduce dataset bias 
            print('Dataset Labels', Counter(y_train))
            unique, uni_count = np.unique(y_train, return_counts=True) 
            rem_count = int(uni_count.max()*0.5)
            
            
            #Randomly remove non expression samples (With label 0) from dataset
            rem_index = random.sample([index for index, i in enumerate(y_train) if i==0], rem_count) 
            rem_index += (index for index, i in enumerate(y_train) if i>0)
            rem_index.sort()
            X_train = [X_train[i] for i in rem_index]
            y_train = [y_train[i] for i in rem_index]
            print('After Downsampling Dataset Labels', Counter(y_train))
            
            #Data augmentation to the micro-expression samples only
            if (expression_type == 'micro-expression'):
                X_train, y_train = data_augmentation(X_train, y_train)
                print('After Augmentation Dataset Labels', Counter(y_train))
                
            #Shuffle the training set
            X_train, y_train = shuffling(X_train, y_train)
            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_train),
                                                              y=y_train)
            class_weight_dict = dict(enumerate(class_weights))

            #print('class weight', str(class_weight_dict))
            model.set_weights(weight_reset) #Reset weights to ensure the model does not have info about current subject

            model.fit(
                np.array(X_train),
                np.array(y_train),
                batch_size,
                epochs,
                verbose=0,
                validation_data=(np.array(X_test), np.array(y_test)),
                shuffle=True,
                callbacks=[callback],
                class_weight=class_weight_dict
            )
            model.save(path)
        else:
            model.load_weights(path)  #Load Pretrained Weights

        result = model.predict(np.array(X_test),verbose=0)

        preds, gt, total_gt = spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p, show_plot)
        TP, FP, FN = evaluation(preds, gt, total_gt, metric_fn)
        
        print('Done Subject', subject_count)
        del X_train, X_test, y_train, y_test, result
    return TP, FP, FN, metric_fn

def final_evaluation(TP, FP, FN, metric_fn):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:", round(metric_fn.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
