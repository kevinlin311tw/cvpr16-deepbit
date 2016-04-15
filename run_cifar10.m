close all;
clear;

addpath(genpath(pwd));
fprintf('cvpr16-deepbit startup\n');

% -- settings start here ---
% set 1 to use gpu, and 0 to use cpu
use_gpu = 1;

% top K returned images
top_k = 1000;
feat_len = 32;

% set result folder
result_folder = './analysis';

% models
% model_file = './examples/deepbit-cifar10-32/DeepBit32_final_iter_1.caffemodel';  %model trained by yourself
model_file = './models/deepbit/DeepBit32_final_iter_1.caffemodel';
% model definition
% model_def_file = './examples/deepbit-cifar10-32/deploy32.prototxt';
model_def_file = './models/deepbit/deploy32.prototxt';

% train-test
test_file_list = '/data/cifar10/test-file-list.txt';
test_label_file = '/data/cifar10/test-label.txt';
train_file_list = '/data/cifar10/train-file-list.txt';
train_label_file = '/data/cifar10/train-label.txt';
% --- settings end here ---

% caffe mode setting
phase = 'test'; % run with phase test (so that dropout isn't applied)


% --- settings end here ---

% outputs
feat_test_file = sprintf('%s/feat-test.mat', result_folder);
feat_train_file = sprintf('%s/feat-train.mat', result_folder);
binary_test_file = sprintf('%s/binary-test.mat', result_folder);
binary_train_file = sprintf('%s/binary-train.mat', result_folder);

% map and precision outputs
map_file = sprintf('%s/map-64.txt', result_folder);
precision_file = sprintf('%s/precision-at-k-64.txt', result_folder);
mean_th = 0;

% feature extraction- training set
if exist(feat_train_file, 'file') ~= 0
    load(feat_train_file);
    mean_bin = mean(feat_train');
    mean_th = mean(mean_bin);
    binary_train = (feat_train>mean_th);
else
    feat_train = feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
    save(feat_train_file, 'feat_train', '-v7.3');
    mean_bin = mean(feat_train');
    mean_th = mean(mean_bin);
    binary_train = (feat_train>mean_th);
    save(binary_train_file,'binary_train','-v7.3');
end


% feature extraction- test set
if exist(feat_test_file, 'file') ~= 0
    load(feat_test_file);
    binary_test = (feat_test>mean_th);    
else
    feat_test = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
    save(feat_test_file, 'feat_test', '-v7.3');
    binary_test = (feat_test>mean_th);
    save(binary_test_file,'binary_test','-v7.3');
end

trn_label = load(train_label_file);
tst_label = load(test_label_file);

[map, precision_at_k] = precision( trn_label, binary_train, tst_label, binary_test, top_k, 1);
fprintf('MAP = %f\n',map);
save(map_file, 'map', '-ascii');
P = [[1:1:top_k]' precision_at_k'];
save(precision_file, 'P', '-ascii');

