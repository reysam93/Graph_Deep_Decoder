exp_type = 3;

if exp_type == 1
    load('..\results\matfiles\arch.mat')
    fmts2 = {'o--', '^--', 'X--', 'v--'};
    legend_txt = {'(1). [6,6,6,6,1], 198','(2). [4,4,4,4,4,1], 124', ...
                  '(3). [3,3,3,3,1], 63', '(4). [2,2,2,2,2,2,2,1], 58'};
    fig = '(a) ';
    lims = [-inf inf];
elseif exp_type == 2
    load('..\results\matfiles\ups.mat')
    legend_txt = {'(1). GDD-None','(2). DD-Reg', '(3). GDD-NoA', ...
                    '(4). GDD-Bin','(5). GDD-Wei'};
    fmts(3,1) = '+';
    fmts2 = {'o--', '^--', '+--', 'X--', '>--'};
    fig = '(b) ';
    lims = [1e-2 inf];
elseif exp_type == 3
    load('..\results\matfiles\input.mat')
    legend_txt = {'(1). Linear, BL','(2). Linear, DD-Reg', ...
                    '(3). Linear, GDD-Wei', '(4). Median, BL', ...
                    '(5). Median, DD-Reg', '(6). Median GDD-Wei'};
    fmts2 = {'o:', '^:', 'X:','o-.', '^-.', 'X-.'};
    fig = '(c) ';
    lims = [1e-3 inf];
elseif exp_type == 4
    load('..\results\matfiles\denoise_real.mat')
    legend_txt = {'(1). BL','(2). DD-Reg', '(3). GDD-Wei'};
    fmts2 = {'o--', '^--', 'X--'};
    fig = '';
    lims = [0.2 inf];
end

n_exps = size(error,3);
fmts = cellstr(fmts);
median_err = zeros(length(n_p),n_exps);
for k=1:size(error,3)
    median_err(:,k) = median(error(:,:,k),2);
end



% Plot median
figure();
for i=1:n_exps
    semilogy(n_p,median_err(:,i),fmts{i},'LineWidth',1.5,'MarkerSize',8);hold on
end
hold off
legend(legend_txt, 'FontSize', 11, 'Location', 'southeast');set(gca,'FontSize',14);
ylabel('Median Error','FontSize', 20);
xlabel([fig 'Normalized Noise Power'],'FontSize', 20);grid on;axis tight
ylim(lims)
set(gcf, 'PaperPositionMode', 'auto')

% % Plot median + prctile75
% prctile75_err = zeros(length(n_p),n_exps);
% for k=1:size(error,3)
%     prctile75_err(:,k) = prctile(error(:,:,k),75,2);
% end
% figure();
% legend_txt2 = cell(1,2*n_exps);
% for i=1:2:2*n_exps
%     legend_txt2{i} = legend_txt{ceil(i/2)};
%     legend_txt2{i+1} = [legend_txt{ceil(i/2)} ' prctile75'];
% end
% for i=1:n_exps
%     semilogy(n_p,median_err(:,i),fmts{i},'LineWidth',1.5,'MarkerSize',8);hold on;
%     semilogy(n_p,prctile75_err(:,i),fmts2{i},'LineWidth',1.5,'MarkerSize',8);hold on;
% end
% hold off
% legend(legend_txt2, 'FontSize', 11);set(gca,'FontSize',18);
% ylabel('Median Error','FontSize', 20);
% xlabel('Noise Power','FontSize', 20);grid on;axis tight

