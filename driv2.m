function driv

% matrix A dimensions is n by n
n = 7;

% construct sparse A0 and A1 with random values
%rng(0);
A0 = spdiags(rand(n,2),0:1,n,n);
A0(n,1) = rand;
A1 = spdiags(rand(n,2),0:1,n,n);
A1(n,1) = rand;

%A0 = [1 .5 0; 0 1 .5; .5 0 1];
%A1 = [1  1 0; 0 1  1;  1 0 1];
%n = length(A0);

% choose a solution
x = rand(n,1);
y = rand(n,1);
y(n) = 0;

% right-hand side for this solution
b = F(x, y, A0, A1, zeros(2*n,1), 0);

% check initial solution is not a final solution
F(x, y, A0, A1, b, 1);

% check rank
J3 = Jac(x, y, A0, A1, 0.1);
svd(J3);

deltat = 0.1;
t = 0;

for step = 1:10
  % predictor step using tangent
  J = Jac(x, y, A0, A1, t);
  f = dFdt(x, y, A0, A1);
  inc = -J(:,1:end-1) \ f;
  x = x + deltat*inc(1:n);
  y(1:n-1) = y(1:n-1) + deltat*inc(n+1:end);
  t = t + deltat;
  
  fprintf('======== time step %d: %f =========\n', step, t);
  fprintf('After predict: %f\n', norm(F(x, y, A0, A1, b, t)));

  % corrector using Newton iterations
  for iter = 1:5
    J = Jac(x, y, A0, A1, t);
    f = F(x, y, A0, A1, b, t);
    inc = -J(:,1:end-1) \ f;
    x = x + inc(1:n);
    y(1:n-1) = y(1:n-1) + inc(n+1:end);
    fprintf('Newton step %d: %f\n', iter, norm(F(x, y, A0, A1, b, t)));
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = F(x, y, A0, A1, b, t)
C = (1-t)*A0 + t*A1;
n = length(C);
z = x + i*y;
f = diag(z)*(C*conj(z)) - (b(1:n)+i*b(n+1:end));
f = [real(f); imag(f)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = dFdt(x, y, A0, A1);
T = (A1-A0);
z = x + i*y;
f = diag(z)*(T*conj(z));
f = [real(f); imag(f)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = Jac(x, y, A0, A1, t);
C = (1-t)*A0 + t*A1;
n = length(C);
Cx = C*x;
Cy = C*y;
J = [diag(x)*C+diag(Cx)    diag(y)*C+diag(Cy)
     diag(y)*C-diag(Cy)   -diag(x)*C+diag(Cx)];

