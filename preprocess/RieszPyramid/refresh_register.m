function new_register = refresh_register(old_register,Nr)
    new_register = zeros(size(old_register));
    for i=1:Nr-1
        new_register(:,:,i) = old_register(:,:,i+1);
    end
end