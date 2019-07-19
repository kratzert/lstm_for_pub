function colors = grab_plot_colors

% there are seven plotting colors in Matlab's default set
nColors = 7;

% init storage
colors = zeros(nColors,3)./0;

% plot enough lines to grab all the colors
h = plot(randn(nColors));

% grab the colors 
for c = 1:nColors; colors(c,:) = h(c).Color; end

% get rid of the plot
close all;

