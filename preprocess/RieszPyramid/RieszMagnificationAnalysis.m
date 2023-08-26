function [phase_cos_sin_amp, Amplitude_out] = RieszMagnificationAnalysis(Img_seq, low_cutoff, high_cutoff, sampling_rate,amplification_factor,varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initializes spatial smoothing kernel and temporal filtering
    % coefficients.
    % Compute an IIR temporal filter coefficients. Butterworth filter could be replaced
    % with any IIR temporal filter. Lower temporal filter order is faster
    % and uses less memory, but is less accurate. See pages 493-532 of
    % Oppenheim and Schafer 3rd ed for more information
%     setPath;
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
    gaussian_kernel = fspecial('gaussian',[2 2],sigma); 

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
    k2 = pyr_level_ini;
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

        %for k = 1:number_of_levels
        % Compute quaternionic phase difference between current Riesz pyramid
        % coefficients and previous Riesz pyramid coefficients.
        if z<=nF
            
           [phase_difference_cos, phase_difference_sin, amplitude] = ...
            ComputePhaseDifferenceAndAmplitude(current_laplacian_pyramid{pyr_level}, ...
            current_riesz_x{pyr_level}, current_riesz_y{pyr_level}, previous_laplacian_pyramid{pyr_level}, ...
            previous_riesz_x{pyr_level}, previous_riesz_y{pyr_level});

            % Adds the quaternionic phase difference to the current value of the quaternionic
            % phase.
            % Computing the current value of the phase in this way is
            % equivalent to phase unwrapping.

            phase_cos = phase_cos + phase_difference_cos;
            phase_sin = phase_sin + phase_difference_sin;
            
            Amplitude_t(:,:,z) = amplitude;
            
            register0_cos = refresh_register(register0_cos,tfi_ord+1);
            register0_cos(:,:,tfi_ord+1) = phase_cos;
            register0_sin = refresh_register(register0_sin,tfi_ord+1);
            register0_sin(:,:,tfi_ord+1) = phase_sin;
        else 
            register0_cos = refresh_register(register0_cos,tfi_ord+1);
            register0_cos(:,:,tfi_ord+1) = register_cos_flip(:,:,z3);
            register0_sin = refresh_register(register0_sin,tfi_ord+1);
            register0_sin(:,:,tfi_ord+1) = register_sin_flip(:,:,z3);
            z3 = z3+1;
        end

        if (z == tfi_ord/2+1)
            register0_cos(:,:,1:tfi_ord/2) = flip(register0_cos(:,:,(tfi_ord/2+2):end),3);
            register0_sin(:,:,1:tfi_ord/2) = flip(register0_sin(:,:,(tfi_ord/2+2):end),3);
            filter_flag = true;
            %%% Slight Modification
            register0_cos(:,:,tfi_ord/2+1) = register0_cos(:,:,tfi_ord/2);
            register0_sin(:,:,tfi_ord/2+1) = register0_sin(:,:,tfi_ord/2);
        elseif (z == nF)
            register_cos_flip = flip(register0_cos,3);
            register_sin_flip = flip(register0_sin,3);
        end

        if (filter_flag)
            phase_filtered_cos = non_causal_FIRfilter(B, register0_cos);
            phase_filtered_sin = non_causal_FIRfilter(B, register0_sin);
           
            phase_filtered_cos = AmplitudeWeightedBlurRiesz(phase_filtered_cos, Amplitude_t(:,:,z2), gaussian_kernel);
            phase_filtered_sin = AmplitudeWeightedBlurRiesz(phase_filtered_sin, Amplitude_t(:,:,z2), gaussian_kernel);

            if (z == tfi_ord/2+1)
                phase_cos_amplify = phase_cos3;
                phase_sin_amplify = phase_sin3;
            else
                phase_cos_amplify = phase_filtered_cos - phase_cos3;
                phase_sin_amplify = phase_filtered_sin - phase_sin3;
            end
            phase_cos3 = phase_filtered_cos;
            phase_sin3 = phase_filtered_sin;
        
            phase_cos_sin_amp(:,:,1,z2) = phase_cos_amplify;
            phase_cos_sin_amp(:,:,2,z2) = phase_sin_amplify;
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