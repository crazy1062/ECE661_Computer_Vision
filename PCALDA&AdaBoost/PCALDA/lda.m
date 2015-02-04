function [vecU, normW] = lda(imgVec, numPerson, numFace)
    [featureSize, sampleSize] = size(imgVec);
    meanWClass = zeros(featureSize, numPerson);

    % mean of within class    
    for i = 1:sampleSize
        imgI = ceil(i/numFace);
        meanWClass(:, imgI) = meanWClass(:, imgI) + imgVec(:, i);
    end
    meanWClass = meanWClass/numFace;
    
    % mean of global vectors
    meanVec = mean(imgVec, 2);

    XB = meanWClass - repmat(meanVec, 1, numPerson);
    XW = zeros(featureSize, sampleSize);
    for i = 1:sampleSize
        XW(:, i) = imgVec(:, i) - meanWClass(:, ceil(i/numFace));
    end
    
    %Yu and Yang's algorithm
    [vecB, valB] = eig(XB'*XB);
    D = diag(valB);
    [temp, sortInd] = sort(D, 1, 'descend');
    vecB = vecB(:, sortInd);
    D = D(sortInd);
    V = XB*vecB;
    
    maxFeatureNum = 30;
    Y = V(:, 1:maxFeatureNum);
    DB = Y'*XB*XB'*Y;
    Z = Y*DB^(-0.5);
    H = Z'*XW;
    [vecU, valU] = eig(H*H');
    W = Z*vecU;
    normW = normVec(W);
end
    
    
    