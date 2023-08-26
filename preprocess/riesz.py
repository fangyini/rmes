import matlab.engine
import dlib
import subprocess
import glob
import os
import shutil
import natsort
import cv2
from tqdm import tqdm
import numpy as np

def getPhase(path, dataset_name, eng):
    if dataset_name == 'CASME_sq':
        sample_freq = 30
        filter_order = 12
    elif dataset_name == 'SAMMLV':
        sample_freq = 200
        filter_order = 74
    # targetdir, filter,sample_freq, filter_order, level
    phase_cos_sin = eng.PhaseExtraction(path, 'low_pass', matlab.single([sample_freq]),
                                                           matlab.int64([filter_order]), matlab.int64([3]), nargout=1)

    # size: phase: 27,27,2,N
    phase_cos_sin = np.asarray(phase_cos_sin)
    return phase_cos_sin

def riesz(dataset_name, output_folder):
    eng = matlab.engine.start_matlab()
    eng.cd(r'preprocess/RieszPyramid/', nargout=0)
    dir_output = dataset_name + '/' + output_folder
    if os.path.exists(dir_output) == False:
        os.mkdir(dir_output)

    alreadySaved = glob.glob(dir_output+'/*.npy')
    if (dataset_name == 'CASME_sq'):
        # Save the images into folder 'rawpic_crop'
        for subjectName in tqdm(glob.glob(dataset_name + '/rawpic_crop_aligned_openface/*')):
            # Create new directory for each subject
            dir_crop_sub = dataset_name + '/rawpic_crop_aligned_openface/' + str(subjectName.split('/')[-1]) + '/*'
            print('Subject', subjectName.split('/')[-1])
            for vid in glob.glob(dir_crop_sub):
                folder = vid.split('/')[-1]
                if folder in alreadySaved:
                    continue
                path = '../../' + vid + '/' + folder + '_aligned/'
                phase_cos_sin = getPhase(path, dataset_name, eng) # 27, 27, N
                np.save(dir_output + '/' + folder, phase_cos_sin)


if __name__ == '__main__':
    riesz('CASME_sq', 'phase')