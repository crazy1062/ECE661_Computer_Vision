clear all;
% nPos = 710;
% nNeg = 879;
% nTot = nPos+2*nNeg;
nPos = 178;
nNeg = 440;
nTot = nPos+nNeg;
nFts = 166000;
nSet = 5000;
nGrp = ceil(nFts/nSet);
fprintf('number of groups = %d\n', nGrp);
fprintf('groups processed =');
feat = zeros(nSet,nTot);
for i = 1:floor(nFts/nSet)
    %feat = zeros(nSet,nTot);
    k0 = nSet*(i-1)+1;
    kf = nSet*i;
    %load('positive.mat','-mat','f');
    load('positive_test.mat','-mat','f');
    feat(:,1:nPos) = f(k0:kf,:);
    clear f;
    %load('negativeA.mat','-mat','f');
    load('negative_test.mat','-mat','f');
    feat(:,nPos+1:nPos+nNeg) = f(k0:kf,:);
    clear f;
    %load('negativeB.mat','-mat','f');
    %feat(:,nPos+nNeg+1:nTot) = f(k0:kf,:);
    %clear f;
    filename = sprintf('FEAT%02d.mat',i);
    save(filename,'feat','-mat','-v7.3');
    fprintf(' %d', i);
end
clear feat;
feat = zeros(mod(nFts,nSet),nTot);
k0 = nSet*floor(nFts/nSet)+1;
kf = nFts;
%load('positive.mat','-mat','f');
load('positive_test.mat','-mat','f');
feat(:,1:nPos) = f(k0:kf,:);
clear f;
%load('negativeA.mat','-mat','f');
load('negative_test.mat','-mat','f');
feat(:,nPos+1:nPos+nNeg) = f(k0:kf,:);
clear f;
%load('negativeB.mat','-mat','f');
%feat(:,nPos+nNeg+1:nTot) = f(k0:kf,:);
%clear f;
filename = sprintf('FEAT%02d.mat',nGrp);
save(filename,'feat','-mat','-v7.3');
fprintf(' %d\n', nGrp);
fprintf('\n');