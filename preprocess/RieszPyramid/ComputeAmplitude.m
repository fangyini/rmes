function [Amplitude_out] = ComputeAmplitude(Img_seq, low_cutoff, high_cutoff, sampling_rate,amplification_factor,varargin)
    p = inputParser();
    default_type_filter = 'low_pass'; %If true, use reference filter 
    default_filt_ord = 14;
    default_sigma = 2;
    default_pyr_level = 2;
    default_pyr_level_ini = 0;

    filTypes = {'low_pass','bandpass'}; 
    checkfiltType = @(x) find(ismember(x, filTypes));
    checkfiltOrd = @(x) (isnumeric(x)&&(mod(x, 2)==0));
    addOptional(p, 'fil_ord', default_filt_ord, checkfiltOrd);
    addOptional(p, 'filType', default_type_filter, checkfiltType);
    addOptional(p, 'sigma', default_sigma, @isnumeric);
    addOptional(p, 'pyr_level', default_pyr_level, @isnumeric);
    addOptional(p, 'pyr_ini', default_pyr_level_ini, @isnumeric);

    parse(p, varargin{:});
    filType  = p.Results.filType;
    tfi_ord  = p.Results.fil_ord;
    sigma    = p.Results.sigma;
    pyr_level = p.Results.pyr_level;
    pyr_level_ini = p.Results.pyr_ini;
    nyquist_frequency = sampling_rate/2;

    if strcmp(filType,'bandpass')
        [B, ~] = fir1(tfi_ord, [low_cutoff/nyquist_frequency,...
        high_cutoff/nyquist_frequency],'bandpass',window(@rectwin,tfi_ord+1)); % bandpass
%         [Bf, Af] = butter(1, [low_cutoff/nyquist_frequency, high_cutoff/nyquist_frequency]);
    elseif strcmp(filType,'low_pass')
        [B, ~] = fir1(tfi_ord, high_cutoff/nyquist_frequency,window(@rectwin,tfi_ord+1)); % low pass
%         [Bf, Af] = butter(2, high_cutoff/nyquist_frequency);
    end
    % Computes convolution kernel for spatial blurring kernel used during
    % quaternionic phase denoising step.
%     gaussian_kernel_sd = 2; % px
%     gaussian_kernel = GetGaussianKernel(gaussian_kernel_sd);
    gaussian_kernel = fspecial('gaussian',[2 2],sigma); %todo: change it back

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialization of variables before main loop.
    % This initialization is equivalent to assuming the motions are zero
    % before the video starts.
%     previous_frame = GetFirstFrameFromVideo();
    previous_frame = Img_seq(:,:,1);
    [previous_laplacian_pyramid, previous_riesz_x, previous_riesz_y] = ...
    ComputeRieszPyramid(previous_frame, pyr_level+1);
    number_of_levels = numel(previous_laplacian_pyramid) - 1; % Do not include lowpass residual
    nF = size(Img_seq,3);
%     for k = 1:number_of_levels
    k2 = pyr_level_ini;
%     k2 = number_of_levels - pyr_level;
    filter_flag = false;

    phase_cos = zeros(size(previous_laplacian_pyramid{pyr_level}));
    phase_sin = zeros(size(previous_laplacian_pyramid{pyr_level}));
    phase_cos3 = zeros(size(previous_laplacian_pyramid{pyr_level}));
    phase_sin3 = zeros(size(previous_laplacian_pyramid{pyr_level}));
    
    register0_cos = zeros([size(previous_laplacian_pyramid{pyr_level}) tfi_ord+1]);       
    register0_sin = zeros([size(previous_laplacian_pyramid{pyr_level}) tfi_ord+1]);
    register_cos_flip = zeros([size(previous_laplacian_pyramid{pyr_level}) tfi_ord+1]);       
    register_sin_flip = zeros([size(previous_laplacian_pyramid{pyr_level}) tfi_ord+1]);
    Amplitude_t =  zeros([size(previous_laplacian_pyramid{pyr_level}),nF]);
    
    phase_cos_sin_amp = zeros([size(previous_laplacian_pyramid{pyr_level}) 2, nF]);
    Amplitude_out = zeros([size(previous_laplacian_pyramid{pyr_level}),nF]);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Main loop. It is executed on new frames from the video and runs until
    % stopped.
    z = 0;
    z2 = 1;
    z3 = 2;
    while (z<nF + tfi_ord/2)
        z=z+1;
        if (z<=nF)
            current_frame = Img_seq(:,:,z);
            [current_laplacian_pyramid, current_riesz_x, current_riesz_y] = ...
            ComputeRieszPyramid(current_frame, pyr_level+1);
        end

        if z<=nF
            
           [phase_difference_cos, phase_difference_sin, amplitude] = ...
            ComputePhaseDifferenceAndAmplitude(current_laplacian_pyramid{pyr_level}, ...
            current_riesz_x{pyr_level}, current_riesz_y{pyr_level}, previous_laplacian_pyramid{pyr_level}, ...
            previous_riesz_x{pyr_level}, previous_riesz_y{pyr_level});

            Amplitude_t(:,:,z) = amplitude;
        else 
            z3 = z3+1;
        end

        if (z == tfi_ord/2+1)
            filter_flag = true;
        end

        if (filter_flag)
            Amplitude_out(:,:,z2) = Amplitude_t(:,:,z2);
         end
        
    if (filter_flag)
        z2 = z2+1;
    end
        % Prepare for next iteration of loop
        previous_laplacian_pyramid = current_laplacian_pyramid;
        previous_riesz_x = current_riesz_x;
        previous_riesz_y = current_riesz_y;
    end

end