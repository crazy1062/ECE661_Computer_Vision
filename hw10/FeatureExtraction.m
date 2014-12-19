function FeatureExtraction(nImg,inFile,outFile,offset)
%>>> initialize the data matrices <<<%
T = zeros(20,1);
S = zeros(21,41);
f = zeros(166000,nImg);
for k = 1:nImg
    %load training images
    imageName = sprintf('%s%06d.png',inFile,offset+k);
    image = imread(imageName);
    image = rgb2gray(image);
    %compute integral images
    for i = 2:21
        for j = 2:41
            T(1:i-1) = sum(image(1:i-1,1:j-1),2);
            S(i,j) = sum(T(1:i-1),1);
        end
    end
    %Extract the vertical Harr-like features
    cnt = 0;
    for h = 1:20
        for w = 1:20
            for i = 1:21-h
                for j = 1:41-2*w
                    x1 = j;
                    x2 = j;
                    x3 = j+w;
                    x4 = j+w;
                    x5 = j+2*w;
                    x6 = j+2*w;
                    y1 = i;
                    y3 = i;
                    y5 = i;
                    y2 = i+h;
                    y4 = i+h;
                    y6 = i+h;
                    cnt = cnt+1;
                    f(cnt,k) = -S(y1,x1)+S(y2,x2)+2*S(y3,x3)-2*S(y4,x4)-S(y5,x5)+S(y6,x6);
                end
            end
        end
    end
    %extract the horizontal Harr-like festures
    for h = 1:10
        for w = 1:40
            for i = 1:21-2*h
                for j = 1:41-w
                    x1 = j;
                    x3 = j;
                    x5 = j;
                    x2 = j+w;
                    x4 = j+w;
                    x6 = j+w;
                    y1 = i;
                    y2 = i;
                    y3 = i+h;
                    y4 = i+h;
                    y5 = i+2*h;
                    y6 = i+2*h;
                    cnt = cnt+1;
                    f(cnt,k) = -S(y1,x1)+S(y2,x2)+2*S(y3,x3)-2*S(y4,x4)-S(y5,x5)+S(y6,x6);
                end
            end
        end
    end
end
%save features to MAT file
save(outFile,'f','-mat','-v7.3');

