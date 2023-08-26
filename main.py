import sys
import argparse
from load_images import *
from load_label import *
from extraction_preprocess import *
from training import *
# Import the os module
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
print('Current working directory: {0}'.format(os.getcwd()))

def main(config):
    # Define the dataset and expression to spot
    dataset_name = config.dataset_name
    expression_type = config.expression_type
    train = config.train
    show_plot = config.show_plot
    
    print(' ------ Spotting', dataset_name, expression_type, '-------')
    
    # Load Images
    #print('\n ------ Croping Images ------')
    #Can comment this out after completed on the dataset specified and intend to try on another expression_type
    #crop_images(dataset_name)
    print("\n ------ Loading Images ------")
    images, subjects, subjectsVideos, videoNames = load_images(dataset_name)
    #images = pickle.load( open( dataset_name + "_images_crop.pkl", "rb" ) )
    
    # Load Ground Truth Label
    print('\n ------ Loading Excel ------')
    codeFinal = load_excel(dataset_name)
    print('\n ------ Loading Ground Truth From Excel ------')
    final_images, final_videos, final_subjects, final_samples, final_names = load_gt(dataset_name, expression_type, images,
                                                                        subjectsVideos, subjects, codeFinal,
                                                                        videoNames)
    print('\n ------ Computing k ------')
    #k = cal_k(dataset_name, expression_type, final_samples)
    k = 6
    
    # Feature Extraction & Pre-processing
    print('\n ------ Feature Extraction & Pre-processing ------')
    dataset = extract_preprocess(final_images, k, final_names, dataset_name, fromSaved=False)
    
    # Pseudo-labeling
    print('\n ------ Pseudo-Labeling ------')
    pseudo_y = pseudo_labeling(final_images, final_samples, k)
    
    # LOSO
    print('\n ------ Leave one Subject Out ------')
    X, y, groupsLabel = loso(dataset, pseudo_y, final_images, final_samples, k)
    
    # Model Training & Evaluation
    # X: len, feature; y: len; groupLabel: len;
    print('\n ------ Training & Testing ------')
    TP, FP, FN, metric_fn = training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset, train,
                                     show_plot, threshold=0.7)
    final_evaluation(TP, FP, FN, metric_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset_name', type=str, default='CASME_sq') # Specify CASME_sq or SAMMLV only
    parser.add_argument('--expression_type', type=str, default='micro-expression') # Specify micro-expression or macro-expression only
    parser.add_argument('--train', type=bool, default=True) #Train or use pre-trained weight for prediction
    parser.add_argument('--show_plot', type=bool, default=True)
    
    config = parser.parse_args()

    main(config)
