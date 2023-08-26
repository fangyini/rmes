function [phase_filtered_spatial] = AmplitudeWeightedBlurRiesz(phase_filtered_temporal, amplitude, gaussian_kernel)
    denominator = conv2(amplitude, gaussian_kernel, 'same');
    numerator = conv2(phase_filtered_temporal.*amplitude, gaussian_kernel, 'same');
    phase_filtered_spatial = numerator ./ (denominator+eps);
end