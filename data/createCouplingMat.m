function coupling_mat = createCouplingMat(num_cells, num_lvs)

coupling_mat = zeros(num_cells, num_lvs);
x = (1:num_cells)';

% exponential decay on diagonal
base_cluster = exp(-x/(num_cells/(2*num_lvs)));
insert_indxs = (floor((num_cells/num_lvs))+1):length(base_cluster);
base_cluster(insert_indxs) = zeros(length(insert_indxs),1); % get rid of small/overlap values
coupling_mat(:,1) = base_cluster;
for i = 2:num_lvs
	coupling_mat(:,i) = circshift(circshift(base_cluster,1,2),round(((i-1)/(num_lvs))*num_cells),1);
end

% add random neurons to each cluster above exp decays
for i = 1:num_lvs
	[~,indx] = max(coupling_mat(:,i));
	temp = randn(indx-1,1);
	temp = abs(temp);
	temp = temp - 1;
	temp(temp < 0) = 0;
	temp(temp > 1) = 1;
	coupling_mat(1:(indx-1),i) = temp;
end