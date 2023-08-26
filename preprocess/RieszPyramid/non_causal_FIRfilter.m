function [result] = non_causal_FIRfilter(B, phase)
%     len = size(B, 2);
%     middle = len / 2 + 0.5;
%     first_part = B(1, 1) * phase(:, :, middle);
%     second_part = 0;
%     for k = 1:(middle-1)
%         second_part = second_part + B(1, k) * (phase(:, :, middle-k) + phase(:, :, middle+k));
%     end 
%     phase_filtered = first_part + second_part;
%     disp(mean(phase_filtered, "all"))
    len = size(B, 2);
    result = 0;
    for k=1:(len)
        result = result + B(1, k) * phase(:, :, k);
    end
    %disp(mean(result, "all"))
end