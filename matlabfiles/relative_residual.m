function [relative_residual,residual]=relative_residual(y_predicted,y)

residual = y_predicted-y;

relative_residual=sqrt(sum(residual.^2)/sum(y.^2));

