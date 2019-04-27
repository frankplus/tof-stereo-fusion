function [ output ] = medfilter( input, window_size )
% computes the medfilt2 on each slice (first ad second dimension )of the 3D matrix
output = input;
for i = 1:size(input,3)
    output(:,:,i) = medfilt2(input(:,:,i), window_size*[1 1],'symmetric');
end

end

