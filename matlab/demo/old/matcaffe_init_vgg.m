function  matcaffe_init_vgg(use_gpu, model_name, model_def_file, model_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 2
  % By default use CPU
  use_gpu = 0;
end
if nargin < 3 || isempty(model_def_file)
  % By default use imagenet_deploy
  %model_def_file = '/home/titan/deep/rcnn_packages/caffe-new/models/bvlc_reference_caffenet/deploy_l7.prototxt';
  model_def_file = sprintf('/home/iis/adsc/vgg_liberty/deploy.prototxt');
  %model_def_file = '/home/iis/adsc/yosemite_3/bdl_64_deploy.prototxt';
end
if nargin < 4 || isempty(model_file)
  % By default use caffe reference model
  %model_file = '/home/titan/deep/rcnn_packages/caffe-new/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
  model_file = sprintf('/home/iis/adsc/vgg_liberty/BDL_128_iter_10000.caffemodel');
  %model_file = '/home/iis/adsc/yosemite_3/BDL_64_iter_50000.caffemodel';
end


%if caffe('is_initialized') == 0
  if exist(model_file, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end
  if ~exist(model_def_file,'file')
    % NOTE: you'll have to get network definition
    error('You need the network prototxt definition');
  end
  caffe('init', model_def_file, model_file)
%end
fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

% put into test mode
caffe('set_phase_test');
fprintf('Done with set_phase_test\n');
