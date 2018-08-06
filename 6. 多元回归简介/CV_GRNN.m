function [desired_spread,result_perfp] = CV_GRNN(p_train,t_train,k,spreadMin,spreadMax)
%输入的是行为特征值数
[m n]=size(p_train);
if m<

desired_spread=[];
mse_max=inf;
result_perfp=[];   
 
indices = crossvalind('Kfold',length(p_train),k);
%对spread进行遍历试验
j=0;
for spread=spreadMin:0.01:spreadMax;
    perfp=[];
    j=j+1;
    for i = 1:k
        %得到k折CV集
        %此处test为logical数组，可以用于数组索引
        test = (indices == i); train = ~test;
        p_cv_train=p_train(train,:);t_cv_train=t_train(train,:); p_cv_test=p_train(test,:);t_cv_test=t_train(test,:);
        %转置后行为特征，列为样本，便于后面归一化
        p_cv_train=p_cv_train';t_cv_train=t_cv_train';p_cv_test= p_cv_test';t_cv_test= t_cv_test';

        %归一化
        [p_cv_train,trainPS,t_cv_train,trainLabelPS] = uniformF(p_cv_train,t_cv_train,1,0,1,0,1);
        %用训练集特征值归一化结果对测试集特征值进行归一化
        p_cv_test=mapminmax('apply',p_cv_test,trainPS);

        %建立GRNN神经网络
        net=newgrnn(p_cv_train,t_cv_train,spread);
        %输出测试集预测结果
        test_Out=sim(net,p_cv_test);
        %对结果进行反归一化
        test_Out=mapminmax('reverse',test_Out,trainLabelPS);
        %求出预测集的mse
        error=t_cv_test-test_Out;
        %记录交叉验证的mse序列
        perfp=[perfp mse(error)];
    end
    result_perfp(j,:)=perfp;
    perfp=mean(abs(perfp));
    %对最优参数进行更新，默认两个最优取最小
    if perfp<mse_max
       mse_max=perfp;
       desired_spread=spread;
    end
end
end

