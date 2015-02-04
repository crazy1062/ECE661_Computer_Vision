function [W, snum] = pca(nImgVec)
    
    % computation trick
    C = nImgVec'*nImgVec;
    
    [EVec, D] = eig(C);
    
    % choose streched vectors
    EVal = diag(D);
    [temp, ind] = sort(EVal, 1, 'descend');
    EVal = EVal(ind);
    EVec = EVec(:, ind);
    
    snum = length(find(EVal >= 1));
    
    % projection matrix
    W = nImgVec * EVec;
    W = normVec(W);
    
end