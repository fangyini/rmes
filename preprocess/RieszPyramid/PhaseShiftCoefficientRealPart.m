function [result] = PhaseShiftCoefficientRealPart(riesz_real, riesz_x, riesz_y, phase_cos, phase_sin)
    phase_mag = sqrt(phase_cos.^2 + phase_sin.^2);
    exp_phase_real = cos(phase_mag);
    exp_phase_x = phase_cos./(phase_mag+eps).*sin(phase_mag);
    exp_phase_y = phase_sin./(phase_mag+eps).*sin(phase_mag);

    result = exp_phase_real.*riesz_real - exp_phase_x.*riesz_x - exp_phase_y.*riesz_y;
end