function [laplacian_pyramid, riesz_x, riesz_y] = ComputeRieszPyramid(grayscale_frame, level)
    % Compute Riesz pyramid of two dimensional frame. This is done by first
    % computing the laplacian pyramid of the frame and then computing the
    % approximate Riesz transform of each level that is not the lowpass
    % residual. The result is stored as an array of grayscale frames.
    % Corresponding locations in the result correspond to the real,
    % i and j components of Riesz pyramid coefficients.
     [h, w] = size(grayscale_frame);
     %level = maxSCFpyrHt(zeros(h,w));
     
    [laplacian_pyramid, gaussianPyramid] = genPyr(grayscale_frame,'lap',level);

    %[pyr, pind] = buildNewPyr(grayscale_frame);
    %number_of_levels = size(pind,1); 
    number_of_levels = numel(laplacian_pyramid)-1;
    if level>number_of_levels
         level = number_of_levels;
     end

    % The approximate Riesz transform of each level that is not the
    % low pass residual is computed. For more details on the approximation,
    % see supplemental material.
    kernel_x = [0.0 0.0 0.0;
                0.5 0.0 -0.5;
                0.0 0.0 0.0];
    kernel_y = [0.0 0.5 0.0;
                0.0 0.0 0.0;
                0.0 -0.5 0.0];
    for k = 1:level
        %laplacian_pyramid{k}=pyrBand(pyr,pind,k);
        riesz_x{k} = conv2(laplacian_pyramid{k}, kernel_x,'same');
        riesz_y{k} = conv2(laplacian_pyramid{k}, kernel_y,'same');
    end
end