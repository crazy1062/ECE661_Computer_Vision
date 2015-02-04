% function for load database

function [nImgVec, imgVec] = loadImg(path, numPerson, numFace)

    d = dir([path, '/*.png']);
    len = length(d)
    
    numTrain = numPerson*numFace;
    if(len ~= numTrain)
        disp('wrong number of training sets');
        nImgVec = []; imgVec = [];
        return;
    end
    
    img = imread([path, '/', d(1).name]);
    [row, col, channel] = size(img);
    
    imgSize = row*col;
    imgVec = zeros(imgSize, len);
    nImgVec = zeros(imgSize, len);
    
    for i = 1:len
       img = imread([path, '/', d(i).name]);
       [row, col, channel] = size(img);
       if(channel == 3)
           img = rgb2gray(img);
       end
       
       imgVec(:, i) = double(img(:));
       
    end
    
    meanVec = mean(imgVec, 2);
    imgVec = imgVec - repmat(meanVec, 1, len);
    nImgVec = normVec(imgVec);
    
end