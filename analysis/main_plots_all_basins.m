%% *** Initialize Workspace **************************************************************
clear all; close all; clc
cd '/Users/gsnearing/Desktop/lstm_pub';
restoredefaultpath
addpath(genpath(pwd))
colors = grab_plot_colors;

%% --- Load Data -------------------------------------------------------------------------

% column names
[temp,colNames] = tblread('basin_specific/benchmark_sacsma_ensemble_14139800.csv',',');

% model names
modelNames = [{'SAC-SMA'},{'NWM'},{'Global LSTM (no statics)'},{'Global LSTM (with statics)'},{'PUB LSMT'}];
         
data = zeros(531,13,16,5)./0;

% sacsma
files = dir('basin_specific/benchmark_sacsma_ensemble_*.csv');
for f = 1:length(files)
    data(f,1:11,:,1) = tblread(strcat(files(f).folder,'/',files(f).name),','); 
end

% nwm
files = dir('basin_specific/benchmark_nwm_retrospective_*.csv');
for f = 1:length(files)
    data(f,1,:,2) = tblread(strcat(files(f).folder,'/',files(f).name),','); 
    data(f,:,:,2) = repmat(data(f,1,:,2),[1,13,1,1]);
end

% global no static
files = dir('basin_specific/global_lstm_no_static_*.csv');
for f = 1:length(files)
    data(f,:,:,3) = tblread(strcat(files(f).folder,'/',files(f).name),','); 
end

% global static
files = dir('basin_specific/global_lstm_static_*.csv');
for f = 1:length(files)
    data(f,:,:,4) = tblread(strcat(files(f).folder,'/',files(f).name),','); 
end

% pub
files = dir('basin_specific/pub_lstm_*.csv');
for f = 1:length(files)
    data(f,1:12,:,5) = tblread(strcat(files(f).folder,'/',files(f).name),','); 
end

% remove empty columns
data(:,:,1,:) = [];

% dimensions
nModels = length(modelNames);
nBasins = size(data,1);

%% --- Calculate Statistics for Plotting/Analysis ----------------------------------------

% stat names
statNames = [{'Nash Sutcliffe Efficiency'}
             {'Fractional Bias'}
             {'Stdandard Deviation Ratio'}
             {'95th Percentile Difference'}
             {'5th Percentile Difference'}];
%              {'Flow Duraction Curve Difference Ratio'}];
%              {'# Zero-Flow Days Difference'}
%              {'Low-Flow Frequency Difference Ratio'}
%              {'High-Flow Frequency Difference Ratio'}];

% dimensions
nStats = length(statNames);
stats = zeros(nStats,nBasins,10,nModels)./0;

% calc the stats
for m = [1,5]
    stats(1,:,:,m) = squeeze( data(:,2:11,1,m));                                     % nse
    stats(2,:,:,m) = squeeze( data(:,2:11,2,m));                                     % bias
    stats(3,:,:,m) = squeeze( data(:,2:11,3,m));                                     % std rat
    stats(4,:,:,m) = squeeze((data(:,2:11,4,m) - data(:,2:11,5,m)) ./ data(:,2:11,4,m));   % 5%
    stats(5,:,:,m) = squeeze((data(:,2:11,6,m) - data(:,2:11,7,m)) ./ data(:,2:11,6,m));   % 95%
end

% clc
% for s = 1:nStats
%     for m = 1:nModels
%         allBasins = stats(s,:,:,m);
%         ensMedians(m,s) = nanmedian(allBasins(:));
%         ensMeans(m,s) = nanmean(allBasins(:));
%     end
% end
% 
% for s = 1:nStats
%     fprintf('---- %s ------------ \n',statNames{s})
%     fprintf('%f %f\n',[ensMedians(:,s),ensMeans(:,s)].')
% end
% 
% stats = reshape(stats,[5,nBasins*9,5]);

%% --- Plot the Metrics ------------------------------------------------------------------

plotdata = max(-1,min(1,squeeze(stats(1,:,:,5))'));

% init figure
fignum = 1;
figure(fignum); close(fignum); figure(fignum)
set(gcf,'color','w')
set(gcf,'position',[1434         911        1509         576]);

% plot stuff
imagesc(plotdata);
colorbar

% aesthetics
xlabel('Basin #','fontsize',24);
ylabel('Ensemble #','fontsize',24);
title('NSE of each PUB LSTM Ensemble Member','fontsize',26);
set(gca,'fontsize',18);
    
% save figure
figname = sprintf('pub_ensembles_NSE.png');
fig = gcf;
saveas(fig,figname);


% init figure
fignum = 2;
figure(fignum); close(fignum); figure(fignum)
set(gcf,'color','w')
set(gcf,'position',[1434         911        1509         576]);

% plot stuff
% plotdata1 = max(-1,min(1,squeeze(stats(1,:,:,1))'));
plotdata5 = max(-1,min(1,squeeze(stats(1,:,:,5))'));
i = find(any(plotdata5<0));
% bar([mean(plotdata5(:,i));mean(plotdata1(:,i))]'); hold on;
bar(mean(plotdata5)); hold on;
% errorbar(1:length(i),mean(plotdata(:,i)),min(plotdata(:,i))-mean(plotdata(:,i)),max(plotdata(:,i))-mean(plotdata(:,i)),'o');
errorbar(1:nBasins,mean(plotdata),min(plotdata)-mean(plotdata),max(plotdata)-mean(plotdata),'o');
% plot(repmat(1:length(i),[10,1]),plotdata(:,i),'o','color',colors(3,:))
plot(repmat(1:nBasins,[10,1]),plotdata,'o','color',colors(3,:))

% aesthetics
xlabel('Basin #','fontsize',24);
ylabel('Ensemble #','fontsize',24);
title('NSE of each PUB LSTM Ensemble Member','fontsize',26);
set(gca,'fontsize',18);
    
% % save figure
% figname = sprintf('pub_ensembles_NSE.png');
% fig = gcf;
% saveas(fig,figname);



%% *** END SCRIPT ************************************************************************
