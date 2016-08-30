function cov_mat = createLVCovarianceMat(num_lvs)
% Create covariance matrix for latent variables

% define probability distribution for different covariances; probabilities
% associated with the values of q below
p = [0.00 0.15 0.20 0.20 0.15 0.10 0.10 0.10 0.00 0.00];
q = [0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90];
cp = cumsum(p);
cov_mat = diag(ones(num_lvs,1));
for i = 1:num_lvs
	for j = (i+1):num_lvs
		r = rand();
		if r < cp(1)
			cov_mat(i,j) = q(1);
			cov_mat(j,i) = q(1);
		elseif r < cp(2)
			cov_mat(i,j) = q(2);
			cov_mat(j,i) = q(2);
		elseif r < cp(3)
			cov_mat(i,j) = q(3);
			cov_mat(j,i) = q(3);
		elseif r < cp(4)
			cov_mat(i,j) = q(4);
			cov_mat(j,i) = q(4);
		elseif r < cp(5)
			cov_mat(i,j) = q(5);
			cov_mat(j,i) = q(5);
		elseif r < cp(6)
			cov_mat(i,j) = q(6);
			cov_mat(j,i) = q(6);
		elseif r < cp(7)
			cov_mat(i,j) = q(7);
			cov_mat(j,i) = q(7);
		elseif r < cp(8)
			cov_mat(i,j) = q(8);
			cov_mat(j,i) = q(8);
		elseif r < cp(9)
			cov_mat(i,j) = q(9);
			cov_mat(j,i) = q(9);
		elseif r < cp(10)
			cov_mat(i,j) = q(10);
			cov_mat(j,i) = q(10);
		end
	end
end


