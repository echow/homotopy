function [a r] = imgs(a)
% incomplete modified Gram-Schmidt
% r is a square upper triangular matrix
% after factorization: input_a'*input_a = r'*r
% factorization does not break down unless exact zero pivot encountered

% note that the diagonal of r is not scaled to be positive
% resulting a has orthogonal columns (not normalized)

n = size(a,2);
pat = spones(a'*a); % choose pattern of r factor
r = zeros(n,n);

for i = 1:n
  r(i,i) = norm(a(:,i));
  q(:,i) = a(:,i)/r(i,i);
  for j = i+1:n
    if pat(i,j) ~= 0
      r(i,j) = q(:,i)'*a(:,j);
      a(:,j) = a(:,j) - r(i,j)*q(:,i);
    end
  end
end

