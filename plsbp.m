% warning off             
close all;  clear;clc;
%% 人参数据处理
% 便携可见光
% res = xlsread('G:\在研\cui\便携导入data.xlsx');
% % nir
res = xlsread('alldata1.xlsx'); %人参数据处理

[cols, h]=size (res);

% y = res (:, cols); %col行数据
% x = res(:, 1:cols-1);

y = res (cols, :);
x = res(1:cols-1,:);

% temp = randperm(300);
% save('temp300.mat','temp');
load('temp300.mat')

%% 网络设置
tic;
n_train=240;
% 训练集
P_train = res(1:cols-1, temp(1: n_train))';
T_train = res(cols, temp(1: n_train))';
M = size(P_train, 1);
% 测试集
P_test = res(1: cols-1, temp(n_train+1: end))';
T_test = res(cols, temp(n_train+1: end))';
N = size(P_test, 1);

% X=[P_train;P_test];
% Y=[T_train;T_test];
% data=[X,Y];
% save('data300.mat','data');

[Xloadings,Yloadings,Xscores,Yscores,beta,PCTVAR] = plsregress([P_train;P_test], [T_train;T_test], n_train); %10
atemp=cumsum(100*PCTVAR(2,:));
ncomp = find(cumsum(100*PCTVAR(2,:)) >= 99.7, 1); % 99.7 = 25 99.96 = 30
% ncomp = 25;

P_train1 = Xscores(1:M, 1:ncomp)';
P_test1 = Xscores(M+1:end, 1:ncomp)';

[p_train, ps_input] = mapminmax(P_train1, 0, 1);
p_test  = mapminmax('apply', P_test1, ps_input);
t_train = ind2vec(T_train');
t_test  = ind2vec(T_test' );

net = newff(p_train, t_train, 12);%6

net.trainParam.epochs = 100000;   % 最大迭代次数
net.trainParam.goal = 1e-6;     % 目标训练误差
net.trainParam.lr = 0.75;       % 学习率
%设置训练参数
net.trainParam.show=50;
net.trainParam.epochs=10000;   %最大训练步数为1000
net.trainParam.goal=0.001;


net = train(net, p_train, t_train);

t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

T_sim1 = vec2ind(t_sim1);
T_sim2 = vec2ind(t_sim2);

[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

error1 = sum((T_sim1 == T_train')) / M * 100 ;
error2 = sum((T_sim2 == T_test' )) / N * 100 ;

time = toc;
fprintf('computation time: %.4f [sec]\n',time);
%% 图像显示
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {strcat('训练集预测结果对比：', ['准确率=' num2str(error1) '%'])};
title(string)
grid
% set(gca,'YTick',8:1:24);

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {strcat('测试集预测结果对比：', ['准确率=' num2str(error2) '%'])};
title(string)
grid
% set(gca,'YTick',8:1:24);

%% 混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
