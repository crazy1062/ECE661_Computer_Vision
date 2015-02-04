% function for normalize feature vectors

function N = normVec(X)

    nsize = size(X);
    N = X;
    len = nsize(2);
    X2 = X.^2;
    
    for i = 1:len
       n = sum(X2(:, i));
       n = sqrt(n);
       N(:, i) = X(:, i)/n;
    end
    
end