%%%%%%%%%%%%%%%%%%%%
% CAR RACING
%%%%%%%%%%%%%%%%%%%%
clc;clear all;
addpath('D:\TANG Yunxi\casadi-windows-matlabR2016a-v3.5.5');
%% setup
import casadi.*
N = 200; % number of control intervals
opti = casadi.Opti();

% decision variable
X = opti.variable(2,N+1);
pos = X(1,:);
vel = X(2,:);

U = opti.variable(1,N); % throttle
T = opti.variable();    % final time

objFunc = T;

opti.minimize(T);

f = @(x,u)[x(2);u-x(2)];  % system dynamics
% create gap closing constraint
dt = T / N;
for k=1:N
    % Runge-Kutta 4 integration
    k1 = f(X(:,k),         U(:,k));
    k2 = f(X(:,k)+dt/2*k1, U(:,k));
    k3 = f(X(:,k)+dt/2*k2, U(:,k));
    k4 = f(X(:,k)+dt*k3,   U(:,k));
    x_next = X(:,k) + dt/6*(k1+2*k2+2*k3+k4);
    opti.subject_to(X(:,k+1)==x_next);
end

% set path constraints
limit = @(pos) 1-sin(2*pi*pos)/1.5;
opti.subject_to(vel<=limit(pos));
opti.subject_to(0<=U<=1);

% set boudary condition
opti.subject_to(pos(1)==0);   % start at position 0 ...
opti.subject_to(vel(1)==0);   % ... from stand-still
opti.subject_to(pos(N+1)==1); % finish line at position 1
opti.subject_to(vel(N+1)<=1.0);
opti.subject_to(T>0);        % Time must be positive

% set initial guess
opti.set_initial(vel, 1);
opti.set_initial(T, 2);

opti.solver('ipopt');
sol = opti.solve();
%%
figure(111);hold on;
T_val = sol.value(T);
time_scale_s = linspace(0,T_val,N+1);
time_scale_u = linspace(0,T_val,N);
plot(time_scale_s, sol.value(vel),'r-','LineWidth',2.0);hold on;
plot(time_scale_s, limit(sol.value(pos)),'m-.','LineWidth',2.0);hold on;
plot(time_scale_s, sol.value(pos),'b-','LineWidth',2.0);hold on;
plot(time_scale_u, sol.value(U),'k-','LineWidth',2.0);grid on;
legend('Vel','Pos Limit','Pos','Throttle');
