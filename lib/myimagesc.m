function myimagesc(data,varargin)
% nice colormap for displaying positive and negative data

red = [...
linspace(0,0,8)';
linspace(0,0,8)';
linspace(0,0,8)';
linspace(0,0.64,4)';
linspace(0.8,1,4)';
linspace(1,1,8)';
linspace(1,1,8)';
linspace(1,0.5,16)'];
	
green = [...
linspace(0,0,8)';
linspace(0,1,22)';
linspace(1,1,2)';
linspace(1,1,2)';
linspace(1,0,22)';
linspace(0,0,8)'];

blue = [...
linspace(0.5,1,16)';
linspace(1,1,8)';
linspace(1,1,8)';
linspace(1,0.8,4)';
linspace(0.64,0,4)';
linspace(0,0,8)';
linspace(0,0,8)';
linspace(0,0,8)'];

cmap = [red green blue];
	
imagesc(data,[-max(abs(data(:))),max(abs(data(:)))],varargin{:})

colormap(cmap)

