function [phase_cos_sin_amp] = PhaseExtraction(targetdir, filter,sample_freq, filter_order, level)

total_dir=targetdir;

sample_freq=sample_freq(1);
filter_order=filter_order(1);
myfolderinfo = dir(total_dir);
length_seq=length(myfolderinfo);
init_list = [];
%offset=0;
for k = (3):length_seq
    img_name = myfolderinfo(k,1).name;
    if ~strcmp(img_name(end-3+1:end), 'jpg') && ~strcmp(img_name(end-3+1:end), 'bmp')
        continue
    end
    numb = isstrprop(img_name,'digit');
    init_list(k-2) = str2double(img_name(numb));
end
[A,sorted_list]=sort(init_list);
img_name = myfolderinfo(sorted_list(3)+2,1).name;
if ~strcmp(img_name(end-3+1:end), 'jpg')&& ~strcmp(img_name(end-3+1:end), 'bmp')
    %img_name='img001.jpg';
    img_name='image549651.bmp';
end

dir_img = strcat(total_dir,'/',img_name);
videoFrame = imread(dir_img);
frameSize = size(videoFrame);
frame = 1;
offset=0;


while (frame <= length_seq-2)
    %% Detecting Facial Features
    % Reading Image
    if (frame <= length_seq+2)
        img_name = myfolderinfo(sorted_list(frame)+2,1).name;
        if ~strcmp(img_name(end-3+1:end), 'jpg')&& ~strcmp(img_name(end-3+1:end), 'bmp')
            frame=frame+1;
            offset=offset+1;
            continue
        end
    end
    dir_img = strcat(total_dir,'/',img_name);
    videoFrame = imread(dir_img);
    videoFrame2 = videoFrame;
    if (size(videoFrame,3) == 1)
        videoFrame = cast(videoFrame, 'double')/255;
        videoFrameGray=videoFrame;
    else
        frameNtsc=rgb2ntsc(videoFrame);
        videoFrameGray=frameNtsc(:,:,1);
    end

    grayframes(:, :, frame-offset) = videoFrameGray;
    im_size = size(videoFrameGray);            
    frame = frame + 1; 
end


[phase_cos_sin_amp, ~] =  RieszMagnificationAnalysis(grayframes, 2, 10,...
    sample_freq,20,'fil_ord',filter_order,'sigma',2,'pyr_level',level,'pyr_ini',1, ...
    'filType', filter);


clear grayframes
end