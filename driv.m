function driv

% choose a solution
x = zeros(3,1);
y = zeros(3,1);
x(1) = 1;
x(2) = 2;
x(3) = 3;
y(1) = 4; % imaginary component
y(2) = 5;
y(3) = 0; % need to enforce this?

% right-hand side for this solution
b = F(x, y, 0, zeros(6,1));

% check initial solution is not a final solution
F(x, y, 1, b);
F2(x, y, 1, b);

% rank is 5
%J = Jac(x, y, 0.1);
%J2 = Jac2(x, y, 0.1);
% J = J(1:5,1:5)
%svd(J)



deltat = 0.0001;
t = 0;

for step = 1:100
  fprintf('======== step %d ========= %f\n', step, t);
  % predictor step using tangent
  J = Jac(x, y, t);
  f = dFdt(x, y, t);
  inc = -J(1:5,1:5) \ f(1:5);
  x = x + deltat*inc(1:3);
  y(1:2) = y(1:2) + deltat*inc(4:5);
  t = t + deltat;
  norm(F(x, y, t, b))
  
  % corrector using Newton iterations UNDONE
  % is this the correct way to solve a nonlinear least squares problem?
  % how to simply to get a square matrix??
  for iter = 1:5
    J = Jac(x, y, t);
    f = F(x, y, t, b);
    inc = -J(1:5,1:5) \ f(1:5);
    x = x + inc(1:3);
    y(1:2) = y(1:2) + inc(4:5);
    norm(F(x, y, t, b))
  end
  
  % if Newton did not converge, then backtrack?
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = F2(x, y, t, b)

A0 = [1 .5 0; 0 1 .5; .5 0 1];
A1 = [1  1 0; 0 1  1;  1 0 1];

C = (1-t)*A0 + t*A1;

z = x + i*y;

f = diag(z)*(C*conj(z)) - (b(1:3)+i*b(4:6));
f = [real(f); imag(f)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = F(x, y, t, b)

f = zeros(6,1);

f(1) = x(1)*x(1) + y(1)*y(1) + .5*(1+t) * ( x(1)*x(2) + y(1)*y(2) ) - b(1);
f(2) = x(2)*x(2) + y(2)*y(2) + .5*(1+t) * ( x(2)*x(3) + y(2)*y(3) ) - b(2);
f(3) = x(3)*x(3) + y(3)*y(3) + .5*(1+t) * ( x(3)*x(1) + y(3)*y(1) ) - b(3);

f(4) = .5*(1+t) * ( -x(1)*y(2) + x(2)*y(1) ) - b(4);
f(5) = .5*(1+t) * ( -x(2)*y(3) + x(3)*y(2) ) - b(5);
f(6) = .5*(1+t) * ( -x(3)*y(1) + x(1)*y(3) ) - b(6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d = dFdt2(x, y, t)

A0 = [1 .5 0; 0 1 .5; .5 0 1];
A1 = [1  1 0; 0 1  1;  1 0 1];
T = (A1-A0);

z = x + i*y;

f = diag(z)*(T*conj(z));
d = [real(f); imag(f)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d = dFdt(x, y, t)

d = zeros(6,1);
d(1) =  x(1)*x(2) + y(1)*y(2);
d(2) =  x(2)*x(3) + y(2)*y(3);
d(3) =  x(3)*x(1) + y(3)*y(1);

d(4) = -x(1)*y(2) + x(2)*y(1);
d(5) = -x(2)*y(3) + x(3)*y(2);
d(6) = -x(3)*y(1) + x(1)*y(3);

d = 0.5*d;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = Jac2(x, y, t);
A0 = [1 .5 0; 0 1 .5; .5 0 1];
A1 = [1  1 0; 0 1  1;  1 0 1];

C = (1-t)*A0 + t*A1;
n = length(C);

J = zeros(2*n,2*n);

for i = 1:n

  % equation for real part
  for j = 1:n
    % partial of equation i wrt x_j
    if i == j
      J(i,j) = C(i,i)*x(i) + C(i,:)*x;
    else
      J(i,j) = C(i,j)*x(i);
    end

    % partial of equation i wrt y_j
    if i == j
      J(i,n+j) = C(i,i)*y(i) + C(i,:)*y;
    else
      J(i,n+j) = C(i,j)*y(i);
    end
  end

  % equation for imaginary part
  for j = 1:n
    % partial of equation i wrt x_j
    if i == j
      J(n+i,j) = C(i,i)*y(i) - C(i,:)*y;
    else
      J(n+i,j) = C(i,j)*y(i);
    end

    % partial of equation i wrt y_j
    if i == j
      J(n+i,n+j) = -C(i,i)*x(i) + C(i,:)*x;
    else
      J(n+i,n+j) = -C(i,j)*x(i);
    end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = Jac(x, y, t)
% Jacobian of f with respect to x, evaluated at x
% but assume y(3)=0

t = .5*(1+t); % kludge

J = zeros(6,6);

J(1,1) =  2*x(1) + t*x(2);
J(1,2) =           t*x(1);
J(1,3) =  0;
J(1,4) =  2*y(1) + t*y(2);
J(1,5) =           t*y(1);
J(1,6) =  0;

J(2,1) =  0;
J(2,2) =  2*x(2) + t*x(3);
J(2,3) =           t*x(2);
J(2,4) =  0;
J(2,5) =  2*y(2) + t*y(3);
J(2,6) =           t*y(2);

J(3,1) =           t*x(3);
J(3,2) =  0;
J(3,3) =  2*x(3) + t*x(1);
J(3,4) =           t*y(3);
J(3,5) =  0;
J(3,6) =  2*y(3) + t*y(1);

J(4,1) = -t*y(2);
J(4,2) =  t*y(1);
J(4,3) =  0;
J(4,4) =  t*x(2);
J(4,5) = -t*x(1);
J(4,6) =  0;

J(5,1) =  0;
J(5,2) = -t*y(3);
J(5,3) =  t*y(2);
J(5,4) =  0;
J(5,5) =  t*x(3);
J(5,6) = -t*x(2);

J(6,1) =  t*y(3);
J(6,2) =  0;
J(6,3) = -t*y(1);
J(6,4) = -t*x(3);
J(6,5) =  0;
J(6,6) =  t*x(1);
