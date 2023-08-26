clearvars
close all
clc


path='/Users/adia/Documents/HKUST/projects/SoftNet-SpotME-main/CASME_sq/rawpic_crop_aligned_openface/s15/15_0102eatingworms/15_0102eatingworms_aligned';

phase_cos_sin = PhaseExtraction(path,'low_pass', 30, 12, 3);
%a=Amplitude_out(:,:,50);
a1= sum(phase_cos_sin_amp(:,:,1,50:60), [3,4]);
