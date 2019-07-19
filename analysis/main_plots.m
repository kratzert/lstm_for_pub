%% *** Initialize Workspace **************************************************************
clear all; close all; clc
restoredefaultpath
addpath(genpath(pwd))
colors = grab_plot_colors;

%% --- Load Data -------------------------------------------------------------------------

% grab column names
[~,colNames] = tblread('./stats/benchmark_sacsma_ensemble.csv',',');

% load all data
data(:,:,1) = tblread('./stats/benchmark_sacsma_ensemble.csv',',');
data(:,:,2) = tblread('./stats/benchmark_nwm_retrospective.csv',',');
data(:,:,3) = tblread('./stats/global_lstm_no_static.csv',',');
data(:,:,4) = tblread('./stats/global_lstm_static.csv',',');
data(:,:,5) = tblread('./stats/pub_lstm.csv',',');

% grab catchment ids
basinIDs = data(:,1,1);
data(:,1,:) = [];

% model names
modelNames = [{'SAC-SMA'},{'NWM'},{'Global LSTM (no statics)'},{'Global LSTM (with statics)'},{'PUB LSTM'}];
         
% dimensions
nModels = length(modelNames);
nBasins = size(data,1);

%% --- Calculate Statistics for Plotting/Analysis ----------------------------------------

% stat names
statNames = [{'Nash Sutcliffe Efficiency'}
             {'Fractional Bias'}
             {'Stdandard Deviation Ratio'}
             {'95th Percentile Difference'}];
%              {'5% Quantile Difference Ratio'}
%              {'Flow Duraction Curve Difference Ratio'}];
%              {'# Zero-Flow Days Difference'}
%              {'Low-Flow Frequency Difference Ratio'}
%              {'High-Flow Frequency Difference Ratio'}];

% dimensions
nStats = length(statNames);

% init storage for performance stats
stats = zeros(nStats,nBasins,nModels)./0;

% calc the stats
for m = 1:nModels
    stats(1,:,m) = data(:,1,m);                                     % nse
    stats(2,:,m) = data(:,2,m);                                     % bias
    stats(3,:,m) = data(:,3,m);                                     % std rat
    stats(4,:,m) = (data(:,6,m) - data(:,7,m)) ./ data(:,6,m);      % 95%
%     stats(4,:,m) = (data(:,4,m) - data(:,5,m)) ./ data(:,4,m);      % 5%
%     stats(6,:,m) = (data(:,14,m) - data(:,15,m)) ./ data(:,14,m);   % fdc
end

% remove stats that are div/0
% stats(:,392,:) = [];
% stats(:,169,:) = [];
stats(isinf(stats))=0/0;

% calculate the ensemble stats reported in the paper 
clc
for s = 1:nStats
    for m = 1:nModels
        ensMedians(m,s) = nanmedian(stats(s,:,m));
        ensMeans(m,s) = nanmean(stats(s,:,m));
        ensMin(m,s) = nanmin(stats(s,:,m));
        ensMax(m,s) = nanmax(stats(s,:,m));
    end
end

% print the table for the figure in Latex format
for s = 1:nStats
    fprintf('\\textbf{%s:} & & & & \n',statNames{s})
    for m = 1:nModels
        fprintf('\\hspace{1em}%s: & %3.2f & %3.2f & %3.2f & %3.2f\n',modelNames{m},[ensMedians(m,s),ensMeans(m,s),ensMin(m,s),ensMax(m,s)]);
    end
end

%% --- Plot Boxplots for all performance stats (not used in the paper) -------------------

% set axis limits for plotting
axisLims = [-1, 1;
            -2, 1;
             0, 2.5;
            -1, 1;
           -10, 1;
            -1, 1];
%            -10,10;
%            -25, 1;
%             -5, 1];

optimal = [1, 0, 1, 0, 0, 0, 0];

% init figure
fignum = 1;
figure(fignum); close(fignum); figure(fignum)
set(gcf,'color','w')
set(gcf,'position',[1640         298        2062        1200]);

% separate plot for each of the stats
for s = 1:nStats
    
    subplot(2,2,s)
    plot([0,nModels+0.5],ones(2,1)*optimal(s),'k--','linewidth',3); hold on
    boxplot(squeeze(stats(s,:,:))); hold on;
    
    % labels & legend
    title(statNames{s},'fontsize',20);
    set(gca,'xticklabel',modelNames);
    
    % aesthetics
    grid on;
    set(gca,'ylim',axisLims(s,:))
    
end % loop over stats

%% --- Plot PDFs and CDFs of all performance metrics -------------------------------------

% separate plot for each performance stat (only NSE is used in the paper)
for s = 1:nStats
    
    % init figure
    fignum = s+1;
    figure(fignum); close(fignum); figure(fignum)
    set(gcf,'color','w')
    set(gcf,'position',[1039         591        1683         422]);

    % plot the KDE-interpolated densities of performance metrics    
    subplot(1,3,1:2)
    for m = 1:nModels
        [f,xi] = ksdensity(max(axisLims(s,1),squeeze(stats(s,:,m))),linspace(axisLims(s,1),axisLims(s,2),100)); hold on;
        h = plot(xi,f);
        h.LineWidth = 3;
        h.Color = colors(m,:);
    end
    
    % PDF aesthetics
    legend(modelNames,'location','nw');
    set(gca,'xlim',axisLims(s,:))
    grid on;
    set(gca,'fontsize',18)
    ylabel('f(x)','fontsize',24);
    xlabel(statNames{s},'fontsize',24);
    titstr = sprintf('Densities of %s Values over %d Basins',statNames{s},nBasins);
    title(titstr,'fontsize',26)
    
    % plot the CDFs of performance metrics
    subplot(1,3,3)
    for m = 1:nModels
        h = cdfplot(squeeze(stats(s,:,m))); hold on;
        h.LineWidth = 3;
        h.Color = colors(m,:);
    end

    % CDF aesthetics
    h.Parent.TitleFontSizeMultiplier = 1.5;
    legend(modelNames,'location','nw');
    grid on;
    set(gca,'fontsize',18)
    set(gca,'xlim',axisLims(s,:))

    % save figure
    figname = sprintf('./figures/frequencies_%s.png',statNames{s});
    fig = gcf;
    saveas(fig,figname);
    
end % stat-loop

%% --- Plot comparative scatterplots for all performance metrics -------------------------
% these were not used in the paper

% separate plot for each performance stat (only NSE is used in the paper)
for s = 1:nStats
    
    % init figure
    fignum = 2*s+2;
    figure(fignum); close(fignum); figure(fignum)
    set(gcf,'color','w')
    set(gcf,'position',[2127         537        1080         968]); 

    % comparison between benchmark models and global no static LSTMs
    subplot(2,2,1)
    plot([axisLims(s,1),axisLims(s,2)],[axisLims(s,1),axisLims(s,2)],'k--','linewidth',2); hold on;
    plot(squeeze(stats(s,:,1)),squeeze(stats(s,:,3)),'.','markersize',15); hold on;
    plot(squeeze(stats(s,:,2)),squeeze(stats(s,:,3)),'+','markersize',7); hold on;
    grid on;
    set(gca,'fontsize',18)
    set(gca,'xlim',[-1,1])
    ylabel(modelNames{3},'fontsize',24);
    xlabel('Benchmark Model','fontsize',24);
    legend(modelNames{1},modelNames{2},'location','nw')
    set(gca,'xlim',axisLims(s,:))
    set(gca,'ylim',axisLims(s,:))
    
    % comparison between benchmark model and global wtih statics LSTMs
    subplot(2,2,2)
    plot([axisLims(s,1),axisLims(s,2)],[axisLims(s,1),axisLims(s,2)],'k--','linewidth',2); hold on;
    plot(squeeze(stats(s,:,1)),squeeze(stats(s,:,4)),'.','markersize',15); hold on;
    plot(squeeze(stats(s,:,2)),squeeze(stats(s,:,4)),'+','markersize',7); hold on;
    grid on;
    set(gca,'fontsize',18)
    set(gca,'xlim',[-1,1])
    ylabel(modelNames{4},'fontsize',24);
    xlabel('Benchmark Model','fontsize',24);
    legend(modelNames{1},modelNames{2},'location','nw')
    set(gca,'xlim',axisLims(s,:))
    set(gca,'ylim',axisLims(s,:))
    
    % comparison between benchmark model and PUB LSTMs
    subplot(2,2,3)
    plot([axisLims(s,1),axisLims(s,2)],[axisLims(s,1),axisLims(s,2)],'k--','linewidth',2); hold on;
    plot(squeeze(stats(s,:,1)),squeeze(stats(s,:,5)),'.','markersize',15); hold on;
    plot(squeeze(stats(s,:,2)),squeeze(stats(s,:,5)),'+','markersize',7); hold on;
    grid on;
    set(gca,'fontsize',18)
    set(gca,'xlim',[-1,1])
    ylabel(modelNames{5},'fontsize',24);
    xlabel('Benchmark Model','fontsize',24);
    legend(modelNames{1},modelNames{2},'location','nw')
    set(gca,'xlim',axisLims(s,:))
    set(gca,'ylim',axisLims(s,:))
    
    % comparison between LSTM models with and without static features
    subplot(2,2,4)
    plot([axisLims(s,1),axisLims(s,2)],[axisLims(s,1),axisLims(s,2)],'k--','linewidth',2); hold on;
    plot(squeeze(stats(s,:,4)),squeeze(stats(s,:,3)),'.','markersize',15,'color',colors(3,:)); hold on;
    plot(squeeze(stats(s,:,4)),squeeze(stats(s,:,5)),'+','markersize',7,'color',colors(4,:)); hold on;
    grid on;
    set(gca,'fontsize',18)
    set(gca,'xlim',[-1,1])
    xlabel(modelNames{4},'fontsize',24);
%     ylabel(modelNames{5},'fontsize',24);
    legend(modelNames{3},modelNames{5},'location','nw')
    set(gca,'xlim',axisLims(s,:))
    set(gca,'ylim',axisLims(s,:))
    
    % save figure
    figname = sprintf('./figures/scatters_%s.png',statNames{s});
    fig = gcf;
    saveas(fig,figname);
    
end % stat loop
 
%% -- Scatterplots against Global LSTM ---------------------------------------------------
% This is a single subplot from the figures above - this was used in the paper.

for s = 1:nStats

    % init figure
    fignum = 3*s+3;
    figure(fignum); close(fignum); figure(fignum)
    set(gcf,'color','w')

    plot([axisLims(s,1),axisLims(s,2)],[axisLims(s,1),axisLims(s,2)],'k--','linewidth',2); hold on;
    h(1) = plot(squeeze(stats(s,:,4)),squeeze(stats(s,:,1)),'o','markersize',7,'color',colors(1,:)); hold on;
    h(2) = plot(squeeze(stats(s,:,4)),squeeze(stats(s,:,2)),'+','markersize',7,'color',colors(2,:)); hold on;
    h(3) = plot(squeeze(stats(s,:,4)),squeeze(stats(s,:,3)),'^','markersize',7,'color',colors(3,:)); hold on;
    grid on;
    set(gca,'fontsize',18)
    set(gca,'xlim',[-1,1])
    xlabel(modelNames{4},'fontsize',24);
    legend(h,modelNames{1},modelNames{2},modelNames{3},modelNames{5},'location','nw')
    set(gca,'xlim',axisLims(s,:))
    set(gca,'ylim',axisLims(s,:))
    title(statNames{s},'fontsize',24);
 
    % save figure
    figname = sprintf('./figures/global_lstm_scatters_%s.png',statNames{s});
    fig = gcf;
    saveas(fig,figname);
    
end % stat loop

%% *** END SCRIPT ************************************************************************
