import dlib
import subprocess
import glob
import os
import shutil
import natsort
import cv2
from tqdm import tqdm

def crop_images(dataset_name, fe_path):
    if (dataset_name == 'CASME_sq'):
        # Save the images into folder 'rawpic_crop'
        for subjectName in glob.glob(dataset_name + '/rawpic/*'):
            dataset_rawpic = dataset_name + '/rawpic/' + str(subjectName.split('/')[-1]) + '/*'

            # Create new directory for 'rawpic_crop'
            dir_crop = dataset_name + '/rawpic_crop_aligned_openface/'
            if os.path.exists(dir_crop) == False:
                os.mkdir(dir_crop)

            # Create new directory for each subject
            dir_crop_sub = dataset_name + '/rawpic_crop_aligned_openface/' + str(subjectName.split('/')[-1]) + '/'
            if not os.path.exists(dir_crop_sub):
                #shutil.rmtree(dir_crop_sub)
                os.mkdir(dir_crop_sub)
            print('Subject', subjectName.split('/')[-1])
            for vid in glob.glob(dataset_rawpic):
                dir_crop_sub_vid = dir_crop_sub + vid.split('/')[-1]  # Get dir of video
                if os.path.exists(dir_crop_sub_vid):
                    #shutil.rmtree(dir_crop_sub_vid)
                    print('skip')
                    continue
                else:
                    os.mkdir(dir_crop_sub_vid)

                for dir_crop_sub_vid_img in natsort.natsorted(glob.glob(vid + '/img*.jpg')):  # Read images
                    img = dir_crop_sub_vid_img.split('/')[-1]
                    count = img[3:-4]  # Get img num Ex 001,002,...,2021
                    newcount = '0' * (5 - len(count)) + count
                    newname = 'img' + newcount + '.jpg'
                    newpath = dir_crop_sub_vid_img[:-len(img)] + newname
                    subprocess.run(['mv', dir_crop_sub_vid_img, newpath])
                #vid: -fdir
                #-out_dir dir_crop_sub_vid/
                # results: dir_crop_sub_vid/vidfoldername
                # subprocess.run(["ls", "-l"])
                # root: project
                # fe_path -fdir vid -out_dir dir_crop_sub_vid -nomask
                subprocess.run([fe_path, "-fdir", vid, "-out_dir", dir_crop_sub_vid, "-nomask", "-simsize", "224"])

    elif (dataset_name == 'SAMMLV'):
        if not os.path.exists(dataset_name + '/SAMM_longvideos_crop_aligned_openface'):  # Delete dir if exist and create new dir
            #  shutil.rmtree(dataset_name + '/SAMM_longvideos_crop')
            os.mkdir(dataset_name + '/SAMM_longvideos_crop_aligned_openface')

        for vid in glob.glob(dataset_name + '/SAMM_longvideos/*'):
            #sub = vid[-5:-2]
            dir_crop = dataset_name + '/SAMM_longvideos_crop_aligned_openface/' + vid.split('/')[-1]

            if os.path.exists(dir_crop):  # Delete dir if exist and create new dir
                # shutil.rmtree(dir_crop)
                print('skip')
                continue
            else:
                os.mkdir(dir_crop)
            print('Video', vid.split('/')[-1])
            subprocess.run([fe_path, "-fdir", vid, "-out_dir", dir_crop, "-nomask"])


def wrappingRes(dataset_name):
    # check dimensionality of all the folders
    # change file names
    #os.rename('old name', 'new name')
    if (dataset_name == 'CASME_sq'):
        # Save the images into folder 'rawpic_crop'
        for subjectName in glob.glob(dataset_name + '/rawpic/*'):
            dataset_rawpic = dataset_name + '/rawpic/' + str(subjectName.split('/')[-1]) + '/*'

            # Create new directory for 'rawpic_crop'
            dir_crop = dataset_name + '/rawpic_crop_aligned_openface/'
            if os.path.exists(dir_crop) == False:
                print('no such directory: ' + str(dir_crop))

            # Create new directory for each subject
            dir_crop_sub = dataset_name + '/rawpic_crop_aligned_openface/' + str(subjectName.split('/')[-1]) + '/'
            if not os.path.exists(dir_crop_sub):
                print('no such sub directory: ' + str(dir_crop_sub))
            print('Subject', subjectName.split('/')[-1])
            for vid in glob.glob(dataset_rawpic):
                dir_crop_sub_vid = dir_crop_sub + vid.split('/')[-1]  # Get dir of video
                if not os.path.exists(dir_crop_sub_vid):
                    print('no such subsub directory: ' + str(dir_crop_sub_vid))

                original = glob.glob(vid+'/img*.jpg')
                cropped = glob.glob(dir_crop_sub_vid+'/'+vid.split('/')[-1]+'_aligned/frame_det*.bmp')
                if len(original) != len(cropped):
                    print('error: ' +str(dir_crop_sub_vid + '/' + vid.split('/')[-1] + '_aligned/frame_det*.bmp'))
                    print('unmatched number of pics, original vs. cropped: ', len(original), len(cropped))

    elif (dataset_name == 'SAMMLV'):
        if not os.path.exists(dataset_name + '/SAMM_longvideos_crop_aligned_openface'):  # Delete dir if exist and create new dir
            #  shutil.rmtree(dataset_name + '/SAMM_longvideos_crop')
            os.mkdir(dataset_name + '/SAMM_longvideos_crop_aligned_openface')

        for vid in tqdm(glob.glob(dataset_name + '/SAMM_longvideos/*')):
            dir_crop_sub_vid = dataset_name + '/SAMM_longvideos_crop_aligned_openface/' + vid.split('/')[-1]

            if not os.path.exists(dir_crop_sub_vid):
                print('no such subsub directory: ' + str(dir_crop_sub_vid))

            original = glob.glob(vid + '/*.jpg')
            cropped = glob.glob(dir_crop_sub_vid + '/' + vid.split('/')[-1] + '_aligned/frame_det*.bmp')
            if len(original) != len(cropped):
                print('error: ' + str(dir_crop_sub_vid + '/' + vid.split('/')[-1] + '_aligned/frame_det*.bmp'))
                print('unmatched number of pics, original vs. cropped: ', len(original), len(cropped))
    return 0


if __name__ == '__main__':
    #subprocess.run(["ls", "-l"])
    crop_images('CASME_sq', '')
    wrappingRes('CASME_sq')