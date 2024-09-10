%% PLS


clc
clear all
% 导入数据
load('data300.mat')
% data=data';
S_Data = data(:,[1:600,601]);

%%
n = 600;   % n 是自变量的个数
m = 1;    % m 是因变量的个数

% 读取训练数据
train_num = 240;  %训练样本数
train_Data = S_Data(1:train_num,:);
mu = mean(train_Data);sig = std(train_Data); %求均值和标准差
ab = zscore(train_Data);  %数据标准化
a = ab(1:train_num,1:n);b = ab(1:train_num,n+1:end);

[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] =plsregress(a,b);
xw=a\XS;  %求自变量提出成分系数，每列对应一个成分，这里xw等于stats.W
% yw=b\YS;  %求因变量提出成分的系数
a_0=PCTVAR(1,:);%b_0=PCTVAR(2,:);
a_1=cumsum(a_0);%b_1=cumsum(b_0);
i=1;%赋初始值
%判断提出成分对的个数
while ((a_1(i)<0.9)&(a_0(i)>0.05))%&(b_1(i)<0.9)&(b_0(i)>0.05))
    i=i+1;
end
ncomp=i;
fprintf('%d对成分分别为：\n',ncomp);
for i=1:ncomp
    fprintf('第%d对成分：\n',i);
    fprintf('u%d=',i);
    for k=1:n%此处为变量x的个数
        fprintf('+(%f*x_%d)',xw(k,i),k);
    end
    fprintf('\n');
end

tic;% 仿真计时开始
[xl,yl,xs,ys,beta,pctvar,mse,stats] = plsregress(a,b,ncomp);
contr =cumsum(pctvar,2);
beta2(1,:)=mu(n+1:end)+mu(1:n)./sig(1:n)*beta(2:end,:).*sig(n+1:end); %原始数据回归方程的常数项
beta2(2:n+1,:)=(1./sig(1:n))'*sig(n+1:end).*beta(2:end,:); %计算原始变量x1，...，xn的系数，每一列是一个回归方程
% 训练数据
train_data = [ones(train_num,1),S_Data(1:train_num,1:n)]';
toc; % 仿真计时结束

% 训练数据误差
PLS_train_Output = beta2' * train_data;
train_Output = S_Data(1:train_num,n+1:end)';
train_err = train_Output - PLS_train_Output;
n1 = length(PLS_train_Output);
train_RMSE = sqrt(sum((train_err).^2)/n1);

% 测试数据的预测结果
%读取测试数据
test_Output = S_Data(train_num+1:end,n+1:end);
test_Input = [ones(length(test_Output),1),S_Data(train_num+1:end,1:n)];
PLS_test_Output = [ones(size(S_Data(train_num+1:end,1:n),1),1) S_Data(train_num+1:end,1:n)]*beta;
%测试数据误差
PLS_test_Output = beta2' * test_Input';
test_err = test_Output' - PLS_test_Output;
n2 = length(PLS_test_Output);
test_RMSE = sqrt(sum((test_err).^2)/n2);

% 预测结果可视化
figure(4);  % 绘制图1
subplot(2,1,1);  % 图1包含2行1列个子图形，首先绘制子图1
plot(PLS_test_Output,':og');  % 用绿色的o绘制测试数据的预测输出值
hold on;
plot(test_Output','-*b');  % 用蓝色的*绘制测试数据的期望输出值
legend('预测输出','期望输出');  % 子图1的注释
title('偏最小二乘法S含量预测结果','fontsize',12)  %子图1的标题
ylabel('S含量','fontsize',12);  % y轴
xlabel('样本','fontsize',12);  % x轴
subplot(2,1,2);  % 绘制子图2
plot(abs(test_Output' - PLS_test_Output),'-*');  % 输出测试数据的预测误差
title('偏最小二乘法S含量预测误差','fontsize',12)  %子图2的标题
ylabel('误差','fontsize',12);  % y轴
xlabel('样本','fontsize',12);  % x轴

%%
% %计算各项误差参数
% error=test_Output-PLS_test_Output;                 % 测试值和真实值的误差
% 
% [~,len]=size(PLS_test_Output);            % len获取测试样本个数，数值等于testNum，用于求各指标平均值
% SSE1=sum(error.^2);                   % 误差平方和
% MAE1=sum(abs(error))/len;             % 平均绝对误差
% MSE1=error*error'/len;                % 均方误差
% RMSE1=MSE1^(1/2);                     % 均方根误差
% MAPE1=mean(abs(error./PLS_test_Output));  % 平均百分比误差
% r=corrcoef(PLS_test_Output,test_Output);    % corrcoef计算相关系数矩阵，包括自相关和互相关系数
% R1=r(1,2);    
% 
% % 显示各指标结果
% disp(' ')
% disp('各项误差指标结果：')
% disp(['误差平方和SSE：',num2str(SSE1)])
% disp(['平均绝对误差MAE：',num2str(MAE1)])
% disp(['均方误差MSE：',num2str(MSE1)])
% disp(['均方根误差RMSE：',num2str(RMSE1)])
% disp(['平均百分比误差MAPE：',num2str(MAPE1*100),'%'])
% disp(['预测准确率为：',num2str(100-MAPE1*100),'%'])
% disp(['相关系数R： ',num2str(R1)])

