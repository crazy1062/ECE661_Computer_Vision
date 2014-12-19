%--------------------------------------------------------------------------
% AdaTest
%--------------------------------------------------------------------------
nPos = 178; % number of positive samples
nNeg = 440; % 1/2 number of negative samples
nTot = nPos+nNeg; % total number of samples
nFts = 166000; % number of features
nSet = 5000; % number of features in a group
nGrp = ceil(nFts/nSet); % nunmber of groups
%-------------------------------------------------------------------
% load results from the training stage
%-------------------------------------------------------------------
load('results.mat','-mat','h','a','s','t','accuracy');
%-------------------------------------------------------------------
% concatenate feature indices for all weak classifiers
%-------------------------------------------------------------------
nStages = find(s == 0,1)-1;
FeatInd = zeros(1,sum(s(1:nStages)));
for i = 1:nStages
    k0 = sum(s(1:i-1))+1;
    kf = sum(s(1:i));
    FeatInd(1,k0:kf) = h(3*(i-1)+1,1:s(i));
end
%-------------------------------------------------------------------
% stack test feature vectors for all weak classifiers
%-------------------------------------------------------------------
[sFeatInd,iFeatInd] = sort(FeatInd,2);
f = zeros(sum(s(1:nStages)),nTot);
for i = 1:nGrp
    filename = sprintf('FEAT%02d.mat',i);
    load(filename,'-mat','feat');
    k0 = find(sFeatInd > (i-1)*nSet, 1, 'first');
    kf = find(sFeatInd < i*nSet+1, 1, 'last');
    for j = k0:kf
        f(iFeatInd(j),:) = feat(mod(sFeatInd(j),nSet),:);
    end
end
%-------------------------------------------------------------------
% evaluate the classifier on the test data set
%-------------------------------------------------------------------
positive = ones(1,nTot);
accuracy = zeros(4,nStages);
false_neg = 0;
for i = 1:nStages % for each stage of the cascade
    accuracy(2,i) = sum(positive(1:nPos));
    accuracy(4,i) = sum(positive(nPos+1:nTot));
    for j = 1:nPos % for each positve test sample
        if positive(j) ~= 1
            continue;
        end
        HS = 0;
        for k = 1:s(i) % for each weak classifier
            p = h(3*(i-1)+2,k);
            theta = h(3*(i-1)+3,k);
            if p*f(sum(s(1:i-1))+k,j) < p*theta
                HS = HS+a(i,k);
            end
        end
        if HS >= t(i)
            positive(j) = 1;
        else
            false_neg = false_neg+1;
            positive(j) = 0;
        end
    end
    false_pos = 0;
    for j = nPos+1:nTot % for each negative test sample
        if positive(j) ~= 1
            continue;
        end
        HS = 0;
        for k = 1:s(i) % for each weak classifier
            p = h(3*(i-1)+2,k);
            theta = h(3*(i-1)+3,k);
            if p*f(sum(s(1:i-1))+k,j) < p*theta
                HS = HS+a(i,k);
            end
        end
        if HS >= t(i)
            false_pos = false_pos+1;
            positive(j) = 1;
        else
            positive(j) = 0;
        end
    end
    accuracy(1,i) = false_pos;
    accuracy(3,i) = false_neg;
end
%-------------------------------------------------------------------
% plot the results
%-------------------------------------------------------------------
figure(1)
plot(1:nStages,accuracy(1,:)/nNeg,'-k')
hold on
plot(1:nStages,accuracy(3,:)/nPos,'-b')
hold off
title('Experimental Accuracy')
ylabel('rate')
xlabel('stage')
legend('false positve','false negative','location','northeast');
print -depsc 'accuracy.eps'