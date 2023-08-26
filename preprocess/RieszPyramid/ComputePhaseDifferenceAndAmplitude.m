function [phase_difference_cos, phase_difference_sin, amplitude] = ComputePhaseDifferenceAndAmplitude(current_real, current_x, current_y, ...
    previous_real, previous_x, previous_y)
    % Computes quaternionic phase difference between current frame and previous
    % frame. This is done by dividing the coefficients of the current frame
    % and the previous frame and then taking imaginary part of the quaternionic
    % logarithm. We assume the orientation at a point is roughly constant to
    % simplify the calcuation.

    % q current = current real + i * current x + j * current y
    % q previous = previous real + i * previous x + j * previous y
    % We want to compute the phase difference, which is the phase of
    % q current/q previous
    % This is equal to (Eq. 10 of tech. report)
    % q current * conjugate(q previous)/j jq previousj jˆ2
    % Phase is invariant to scalar multiples, so we want the phase of
    % q current * conjugate(q previous)
    % which we compute now (Eq. 7 of tech. report). Under the constant orientation assumption,
    % we can assume the fourth component of the product is zero.
    q_conj_prod_real = current_real.*previous_real + ...
    current_x.*previous_x + current_y.*previous_y;
    q_conj_prod_x = -current_real.*previous_x + previous_real.*current_x;
    q_conj_prod_y = -current_real.*previous_y + previous_real.*current_y;

    % Now we take the quaternion logarithm of this (Eq. 12 in tech. report)
    % Only the imaginary part corresponds to quaternionic phase.
    q_conj_prod_amplitude = sqrt(q_conj_prod_real.^2 + q_conj_prod_x.^2 + q_conj_prod_y.^2);
    phase_difference = acos(q_conj_prod_real./(q_conj_prod_amplitude+eps));
    cos_orientation = q_conj_prod_x ./ sqrt(q_conj_prod_x.^2+q_conj_prod_y.^2+eps);
    sin_orientation = q_conj_prod_y ./ sqrt(q_conj_prod_x.^2+q_conj_prod_y.^2+eps);

    % This is the quaternionic phase (Eq. 2 in tech. report)
    phase_difference_cos = phase_difference .* cos_orientation;
    phase_difference_sin = phase_difference .* sin_orientation;

    % Under the assumption that changes are small between frames, we can
    % assume that the amplitude of both coefficients is the same. So,
    % to compute the amplitude of one coefficient, we just take the square root
    % of their conjugate product
    amplitude = sqrt(q_conj_prod_amplitude);

end