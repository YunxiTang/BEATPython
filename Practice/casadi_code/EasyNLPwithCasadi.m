addpath('D:\TANG Yunxi\casadi-windows-matlabR2016a-v3.5.5');
opti = casadi.Opti();

x = opti.variable();
y = opti.variable();
r = opti.parameter();

objFunc = (1-x)^2 + (y - x^2)^2;
eqCons = (x^2 + y^2 <= r);
ieqCons = (y >= x);

opti.minimize(objFunc);
opti.subject_to(eqCons);
opti.subject_to(ieqCons);
opti.solver('ipopt');
for r_val = linspace(1,3,25)
    opti.set_value(r, r_val);
    sol = opti.solve();
    figure(111);hold on;
    plot(r_val,sol.value(y),'ko-');
end
figure(222);
plot(sol.value(x),sol.value(y),'ro-');