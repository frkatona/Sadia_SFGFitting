% Sum of SFG signal % 

% OUTPUT:
% return the signal array (real numberes) as function of frequencies

% INPUT:
% w frequencies, can be array
% n number of peaks
function y = SFG_signal_sum (parameters, frequency)


ki=zeros(size(frequency));
y=zeros(size(frequency));

num_peaks = (length(parameters)-2)/4; 
for i = 1:num_peaks
   index = (i-1)*4 + 2 ;
	ki=ki+SFG_Lorentzian_Gaussian(parameters(index+1),parameters(index+2),frequency,parameters(index+3),parameters(index+4));
end


ki= ki+ parameters(2);  % non-resonent SFG signal

y = abs(ki).^2;

y= y+ parameters(1);  % Backgroud noise from green light scattering
