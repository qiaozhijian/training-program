%% 子函数 SVMcgForRegress.m
%在一定范围内选择比较不错的支持向量机参数
%train_label train
%cmin cmax c参数的最大最小范围 2^cmin~2^cmax 
%gmin gmax g参数的最大最小范围 2^gmin~2^gmax 
% v 交叉验证分块数
% cstep,gstep c,g参数变化步长
% msestep 显示准确率的图的步进大小
function [mse,bestc,bestg] = SVMcgForRegress(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,msestep)
%%nargin 判断变量的个数给予默认值
if nargin < 10  msestep = 0.06;             end
if nargin < 8   cstep = 0.8;gstep = 0.8;    end
if nargin < 7   v = 5;                      end
if nargin < 5   gmax = 8;   gmin = -8;      end
if nargin < 3   cmax = 8;   cmin = -8;      end

% X:c Y:g cg:acc
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
%c参数共有m个选项，g参数共用n个选项
[m,n] = size(X);
%分别记录每个不同的cg组合的结果
cg = zeros(m,n);

eps = 10^(-4);
%给输出参数赋初值
bestc = 0;bestg = 0;mse = Inf;
%是以2为底的测试序列
basenum = 2;
for i = 1:m
    for j = 1:n
        %-v 交叉验证模式（真的简单方便！） -c -g -s采用epsilon-SVR模型
        %-p设置epsilon模型的损失参数默认0.001
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s 3 -p 0.1'];
        %在交叉验证模式下返回的是交叉验证的平均均方根误差mse
        cg(i,j) = svmtrain(train_label, train, cmd);
        %对最优参数进行更新
        if cg(i,j) < mse
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        %？？
        if abs( cg(i,j)-mse )<=eps && bestc > basenum^X(i,j)
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
    end
end
% 画等高线图以及网栅图
[cg,ps] = mapminmax(cg,0,1);
figure;
[C,h] = contour(X,Y,cg,0:msestep:0.5);
clabel(C,h,'FontSize',10,'Color','r');
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
firstline = 'SVR参数选择结果图(等高线图)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg),'CVmse=',num2str(mse)];
title({firstline;secondline},'Fontsize',12);
grid on;

figure;
meshc(X,Y,cg);
axis([cmin,cmax,gmin,gmax,0,1]);
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
zlabel('MSE','FontSize',12);
firstline = 'SVR参数选择结果图(3D视图)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg),' CVmse=',num2str(mse)];
title({firstline;secondline},'Fontsize',12);
end