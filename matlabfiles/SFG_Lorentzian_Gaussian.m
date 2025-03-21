% Suscepbility with a gaussion distributed phase angle centered at zero degree% 

% OUTPUT:
% return the suscepbility array (complex numberes) as function of frequencies for a resonent peak

% INPUT:
% w frequencies, can be array
% wr resonent frequency, Tau, width, A, oscillation strength. All scaler
% sigma, distribution of phase angle, in arc unit;
function ki = SFG_Lorentzian_Gaussian (A,wr,w,Tau,sigma)


if (sigma<=10) % wavenumbers
   ki=SFG_Lorentzian(A,wr,w,Tau);
   return;
end

ki=zeros(size(w));

num_per_sigma=30;  %number of point evaluated for each sigma
max_sigma=3;       %max sigma evaluated

norm=0;
for n= -num_per_sigma*max_sigma:num_per_sigma*max_sigma 
   weight= exp(-1*n*n/(num_per_sigma*num_per_sigma)); % weight according to guassion distribution.
   ki=ki+SFG_Lorentzian(A,wr-n/num_per_sigma*sigma,w,Tau)*weight;
   norm=norm + weight;
end

ki=ki/norm;

