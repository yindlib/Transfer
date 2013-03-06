function [B, L] = pca_basis(X, ld)
% Find a PCA mapping between X and L to initialize LATTL
%   [B, L] = pca_basis(X, d)
%
% Input
%   X:  n x d matrix (n=ex, d=feats)
%   ld: dimension of latent space (number of PCs to retain)
%
% Output
%   B:  d x r basis
%   L:  n x r latent representation
%
% Note the following relationship:
%
% [B, L] = princomp(X);
% L = X * B;
% X = L * B';
%
% In other words, because PCA enforces an orthogonality constraint, it is
% reasonable to convert back and forth between the true representations
% using the transpose of the basis. This may not be true in general of the
% more general sparse-coding bases.
%
% PCA is the solution to the sparse-coding like optimization problem:
%
% minimize_b,a  Sum_i ||x_i - Sum_j a_ij b_j||_2^2
% such that     b_j's are orthogonal
%
% See STL learning paper from Raina, Battle, Lee, Packer, Ng, ICML 2007.

[ B, L ] = princomp(X);
B = B(:, 1:ld);
L = L(:, 1:ld);
