clear all;
clc;
%extract training image feature
file_train = sprintf('train/positive/');
FeatureExtraction(710,file_train,'positive.mat',0);
file_train = sprintf('train/negative/');
FeatureExtraction(879,file_train,'negativeA.mat',0);
FeatureExtraction(879,file_train,'negativeB.mat',879);
%extract testing image feature
file_test = sprintf('test/positive/');
FeatureExtraction(178,file_test,'positive_test.mat',710);
file_test = sprintf('test/negative/');
FeatureExtraction(440,file_test,'negative_test.mat',1758);