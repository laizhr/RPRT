function [b_tplus1_hat, phi_tplus1_hat] = RPRT(data, phi_t_hat, daily_port_total)
%{
This function is the main code for the Reweighted Price Relative Tracking System for Automatic 
Portfolio Optimization (RPRT)[4]system. It exploits a reweighted price relative, which 
automatically assigns separate weights to the price relative predictions according to 
each asset¡¯s performance, and these weights will also be automatically updated.

For any usage of this function, the following papers should be cited as
reference:

[1] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. ¡°A peak price tracking 
based learning system for portfolio selection¡±, 
IEEE Transactions on Neural Networks and Learning Systems, 2017. Accepted.
[2] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang.  ¡°Radial basis functions 
with adaptive input and composite trend representation for portfolio selection¡±, 
IEEE Transactions on Neural Networks and Learning Systems, 2018. Accepted.
[3] Pei-Yi Yang, Zhao-Rong Lai*, Xiaotian Wu, Liangda Fang. ¡°Trend Representation 
Based Log-density Regularization System for Portfolio Optimization¡±,  
Pattern Recognition, vol. 76, pp. 14-24, Apr. 2018.
[4]Zhao-Rong Lai, Pei-Yi Yang, Liangda Fang and Xiaotian Wu.
 ¡°Reweighted Price Relative Tracking System for Automatic Portfolio Optimization¡±. 
IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2018. Accepted.
[5]Zhao-Rong Lai, Pei-Yi Yang, Xiaotian Wu and Liangda Fang. ¡°A kernel-based trend pattern 
tracking system for portfolio optimization¡±, Data Mining and Knowledge
Discovery, 2018. Accepted.

At the same time, it is encouraged to cite the following papers with previous related works:

[6] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, ¡°Efficient
projections onto the \ell_1-ball for learning in high dimensions,¡± in
Proceedings of the International Conference on Machine Learning (ICML 2008), 2008.
[7] B. Li, D. Sahoo, and S. C. H. Hoi. Olps: a toolbox for on-line portfolio selection.
Journal of Machine Learning Research, 17, 2016.

Inputs:
data                      -data with price relative sequences
phi_t_hat				  -price relative prediction at time t
daily_port_total          -daily selected portfolios of RPRT

Output:
b_tplus1_hat              -portfolios selection at time t+1
phi_tplus1_hat            -price relative prediction at time t+1
%}

%% Parameter Setting
epsilon=50;				  % -expected profiting level
win_size = 5; 			  % -window size
theta = 0.8; 			  % -mixing parameter

%% Variables Inital
[T, N] = size(data);
b_t_hat = daily_port_total(T,:)';
x_t = data(T,:);

%% main
gamma_tplus1 = theta*x_t./(theta*x_t+phi_t_hat);  					
phi_tplus1_hat = gamma_tplus1+(1-gamma_tplus1).*(phi_t_hat./x_t); 	
if (T < win_size+1)
    x_tplus1_hat = data(T, :);
else
    x_tplus1_hat = zeros(1, N);
    tmp_x = ones(1, N);
    for i = 1:win_size
        x_tplus1_hat = x_tplus1_hat + 1./tmp_x;
        tmp_x = tmp_x.*data(T-i+1, :);
    end
    x_tplus1_hat = x_tplus1_hat*(1/win_size); 						
end

D = diag(x_tplus1_hat'); 											

phi_tplus1_bar = mean(phi_tplus1_hat);
ell = max([0, epsilon - phi_tplus1_hat*b_t_hat]); 					
denominator = norm(phi_tplus1_hat - phi_tplus1_bar,2)^2; 			
if (~eq(denominator, 0.0))
    lambda = ell / denominator; 									
else  
    lambda = 0;
end
% Update portfolio
b_tplus1 = b_t_hat + lambda*D*(phi_tplus1_hat' - phi_tplus1_bar); 	
% Normalize portfolio
b_tplus1_hat = simplex_projection_selfnorm2(b_tplus1, 1); 			
end
