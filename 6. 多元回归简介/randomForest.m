function [testY,bestNum] = randomForest(sampleX,sampleY,testX,treeNumMin,treeNumMax,step)

bestMse=inf;
bestNum=0;
n=length(sampleX);

for ii=treeNumMin:step:treeNumMax
    p_train=sampleX(1:n*3/4,:);
    t_train=sampleY(1:n*3/4,:);
    p_test=sampleX(n*3/4+1:end,:);
    t_test=sampleY(n*3/4+1:end,:);
    B= TreeBagger(ii,p_train,t_train,'Method','regression'); 
    for i=1:length(p_test)
        y3(i)=B.predict(p_test(i,:));
    end 

    p_train=sampleX(n/4+1:end,:);
    t_train=sampleY(n/4+1:end,:);
    p_test=sampleX(1:n/4,:);
    t_test=sampleY(1:n/4,:);
    B= TreeBagger(ii,p_train,t_train,'Method','regression'); 
    for i=1:length(p_test)
        y4(i)=B.predict(p_test(i,:));
    end

    mseB=mse(y3-t_test)+mse(y4-t_test);
    if mseB<bestMse
        bestMse=mseB;
        bestNum=ii;
    end

end

B= TreeBagger(bestNum,sampleX,sampleY,'Method','regression');  
for i=1:length(testX)
    testY(i)=B.predict(testX(i,:));  
end
bestNum=[bestNum bestMse];
end

