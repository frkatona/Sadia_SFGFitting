% A Lorentzian type function for suscepbility in SFG, no phase angle % 

% OUTPUT:
% return the suscepbility array (complex numberes) as function of frequencies for a resonent peak

% INPUT:
% w frequencies, can be array
% wr resonent frequency, Tau, width, A, oscillation strength. All scaler
function ki = SFG_Lorentzian (A,wr,w,Tau)

ki=A*ones(size(w))./(w-wr+i*Tau);