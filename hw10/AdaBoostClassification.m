%--------------------------------------------------------------------------
% AdaClass
%--------------------------------------------------------------------------
nPos = 710; % number of positive samples
nNeg = 879; % 1/2 number of negative samples
nTot = nPos+2*nNeg; % total number of samples
nFts = 166000; % number of features
nSet = 5000; % number of features in a group
nGrp = ceil(nFts/nSet); % nunmber of groups
%nGrp = 1;
nWClass = 50; % maximum number of weak classifiers
nStages = 20; % maximum number of stages in the cascade
%-------------------------------------------------------------------
% initialize the cascaded classifier
%-------------------------------------------------------------------
h = zeros(3*nStages,nWClass);
a = zeros(nStages,nWClass);
s = zeros(1,nStages);
t = zeros(1,nStages);
accuracy = zeros(4,nStages);
positive = ones(1,nTot);
feat = zeros(nSet,nTot);
f = zeros(nWClass,nTot);
%-------------------------------------------------------------------
% add stages until 100% accuracy obtained
%-------------------------------------------------------------------
stage_cnt = 0;
false_pos = inf;
false_neg = inf;
while false_pos+false_neg > 0
    %---------------------------------------------------------------
    % increment cascade stage count
    %---------------------------------------------------------------
    stage_cnt = stage_cnt+1;
    %---------------------------------------------------------------
    % initialize the weights
    %---------------------------------------------------------------
    mPos = sum(positive(1:nPos));
    mNeg = sum(positive(nPos+1:nTot));
    w = [ones(1,nPos)/(2*mPos),ones(1,2*nNeg)/(2*mNeg)];
    fprintf('cascade stage %d\n', stage_cnt);
    fprintf('negative samples pruned =');
    %---------------------------------------------------------------
    % add weak classifiers until 50% of negative samples pruned
    %---------------------------------------------------------------
    cnt = 0;
    negCnt = 0;
    while negCnt < mNeg/2
        %-----------------------------------------------------------
        % increment weak classifier count
        %-----------------------------------------------------------
        cnt = cnt+1;
        %-----------------------------------------------------------
        % normalize the weights
        %-----------------------------------------------------------
        w = w/sum(positive.*w);
        %-----------------------------------------------------------
        % search for the best weak classifier
        %-----------------------------------------------------------
        eMin = inf;
        TPos = sum(positive(1:nPos).*w(1:nPos));
        TNeg = sum(positive(nPos+1:nTot).*w(nPos+1:nTot));
        for i = 1:nGrp % for each group of features
            filename = sprintf('FEAT%02d.mat',i);
            load(filename,'-mat','feat');
            [sfeat,ifeat] = sort(feat,2);
            nFeats = size(feat,1);
            fNew = 0;
            for j = 1:nFeats % for each feature in the group
                SPos = 0;
                SNeg = 0;
                for k = 1:nTot % for each training sample
                    if positive(ifeat(j,k)) ~= 1
                        continue;
                    end
                    e1 = SPos+(TNeg-SNeg);
                    e2 = SNeg+(TPos-SPos);
                    if e2 < e1
                        eTmp = e2;
                        pTmp = 1;
                    else
                        eTmp = e1;
                        pTmp = -1;
                    end
                    if eTmp < eMin
                        fNew = 1;
                        eMin = eTmp;
                        h(3*(stage_cnt-1)+1,cnt) = nSet*(i-1)+j;
                        h(3*(stage_cnt-1)+2,cnt) = pTmp;
                        h(3*(stage_cnt-1)+3,cnt) = sfeat(j,k);
                    end
                    if ifeat(j,k) > nPos
                        SNeg = SNeg+w(ifeat(j,k));
                    else
                        SPos = SPos+w(ifeat(j,k));
                    end
                end
            end
            if fNew == 1
                jFeat = h(3*(stage_cnt-1)+1,cnt)-nSet*(i-1);
                f(cnt,:) = feat(jFeat,:);
            end
        end
        %-----------------------------------------------------------
        % update the weights
        %-----------------------------------------------------------
        beta = eMin/(1-eMin);
        for i = 1:nPos
            if positive(i) ~= 1
                continue;
            end
            p = h(3*(stage_cnt-1)+2,cnt);
            theta = h(3*(stage_cnt-1)+3,cnt);
            if p*f(cnt,i) < p*theta
                w(i) = w(i)*beta;
            end
        end
        for i = nPos+1:nTot
            if positive(i) ~= 1
                continue;
            end
            p = h(3*(stage_cnt-1)+2,cnt);
            theta = h(3*(stage_cnt-1)+3,cnt);
            if ~(p*f(cnt,i) < p*theta)
                w(i) = w(i)*beta;
            end
        end
        %-----------------------------------------------------------
        % set threshold for strong classifier
        %-----------------------------------------------------------
        a(stage_cnt,cnt) = log(1/beta);
        HSMin = inf;
        for i = 1:nPos
            if positive(i) ~= 1
                continue;
            end
            HS = 0;
            for j = 1:cnt
                p = h(3*(stage_cnt-1)+2,j);
                theta = h(3*(stage_cnt-1)+3,j);
                if p*f(j,i) < p*theta
                    HS = HS+a(stage_cnt,j);
                end
            end
            if HS < HSMin
                HSMin = HS;
            end
        end
        %-----------------------------------------------------------
        % test strong classifier on negative samples
        %-----------------------------------------------------------
        negCnt = 0;
        for i = nPos+1:nTot
            if positive(i) ~= 1
                continue;
            end
            HS = 0;
            for j = 1:cnt
                p = h(3*(stage_cnt-1)+2,j);
                theta = h(3*(stage_cnt-1)+3,j);
                if p*f(j,i) < p*theta
                    HS = HS+a(stage_cnt,j);
                end
            end
            if HS < HSMin
                negCnt = negCnt+1;
            end
        end
        %-----------------------------------------------------------
        % display progress
        %-----------------------------------------------------------
        fprintf(' %d',negCnt);
    end
    fprintf('\n');
    %---------------------------------------------------------------
    % save the weak class count and the strong class threshold
    %---------------------------------------------------------------
    s(stage_cnt) = cnt;
    t(stage_cnt) = HSMin;
    %---------------------------------------------------------------
    % evaluate on samples classified as positive
    %---------------------------------------------------------------
    false_neg = 0;
    for i = 1:nPos
        if positive(i) ~= 1
            continue;
        end
        HS = 0;
        for j = 1:cnt
            p = h(3*(stage_cnt-1)+2,j);
            theta = h(3*(stage_cnt-1)+3,j);
            if p*f(j,i) < p*theta
                HS = HS+a(stage_cnt,j);
            end
        end
        if HS >= HSMin
            positive(i) = 1;
        else
            false_neg = false_neg+1;
            positive(i) = 0;
        end
    end
    false_pos = 0;
    for i = nPos+1:nTot
        if positive(i) ~= 1
            continue;
        end
        HS = 0;
        for j = 1:cnt
            p = h(3*(stage_cnt-1)+2,j);
            theta = h(3*(stage_cnt-1)+3,j);
            if p*f(j,i) < p*theta
                HS = HS+a(stage_cnt,j);
            end
        end
        if HS >= HSMin
            false_pos = false_pos+1;
            positive(i) = 1;
        else
            positive(i) = 0;
        end
    end
    accuracy(1,cnt) = false_pos;
    accuracy(2,cnt) = mPos;
    accuracy(3,cnt) = false_neg;
    accuracy(4,cnt) = mNeg;
end
%-------------------------------------------------------------------
% save the results to a MAT file
%-------------------------------------------------------------------
save('results.mat','h','a','s','t','accuracy','-mat','-v7.3');