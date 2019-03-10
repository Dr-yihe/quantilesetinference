function u = dirrnd(d,nr)
% generating uniformly distributed unit vectors on the unit sphere
% d: the dimension of the Euclidean space
% nr: the number of directions for each dimension

nr=nr+mod(nr,2);
u=randn(nr/2*d,d); % standard normal margins
u=u./(sqrt(sum(u.^2,2))*ones(1,d)); % standardized to have unit norm
u=[u;-u];

end

