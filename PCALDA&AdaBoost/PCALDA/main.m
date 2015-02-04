clc;
clear all;





% PCA method

numPerson = 30;
numFace = 21;
numTrain = numPerson*numFace;
numTest = numTrain;

[nImg, img] = loadImg('train', numPerson, numFace);

[nTest, imgTest] = loadImg('test', numPerson, numFace);

[nFace, snum] = pca(nImg);

accuracyPCA = zeros(numTest, snum);

for k = 1:snum
    eigenFaceSub = nFace(:, 1:k);
    projectedImg = eigenFaceSub'*nImg;
    projectedTestImg = eigenFaceSub'*nTest;
    
    err = zeros(numTest, 1);
    
    for testInd = 1:numTest
        testI = ceil(testInd/numFace);
        
        for trainInd = 1:numTrain
            err(trainInd) = (norm(projectedTestImg(:, testInd) - projectedImg(:, trainInd)))^2;
        end
        
        [minDist, recogInd] = min(err);
        
        recogI = ceil(recogInd/21);
        
        if testI == recogI
            accuracyPCA(testInd, k) = 1;
        end
    end    
end

accuracyPCARate = sum(accuracyPCA, 1)/numTest;
figure(1)
hold on
ind = 1:snum;
plot(ind(1:25), accuracyPCARate(1:25), 'r*-');
axis([0 25 0 1]);




% LDA method


[vecU, nW] = lda(img, numPerson, numFace);

maxEigenNum = 30;

accuracyLDA = zeros(numTest, maxEigenNum - 1);
meanVec = mean(img, 2);

for L = 1:maxEigenNum - 1
    vecUSub = vecU(:, 1:L);
    W = nW(:, 1:L);
    
    projectedImg = W'*(img - repmat(meanVec, 1, numTrain));
    projectedTestImg = W'*(imgTest - repmat(meanVec, 1, numTest));

    err = zeros(numTest, 1);
    
    for testInd = 1:numTest
        testI = ceil(testInd/numFace);
        
        for trainInd = 1:numTrain
            err(trainInd) = (norm(projectedTestImg(:, testInd) - projectedImg(:, trainInd)))^2;
        end
        [minDist, recogInd] = min(err);
        
        recogI = ceil(recogInd/numFace);
        if testI == recogI
            accuracyLDA(testInd, L) = 1;
        end
    end
end

accuracyLDARate = sum(accuracyLDA, 1)/numTest;

figure(1)
hold on
ind = 1:maxEigenNum - 1;
plot(ind(1:25), accuracyLDARate(1:25), 'b*-');
axis([0 25 0 1]);
legend('PCA', 'LDA')








