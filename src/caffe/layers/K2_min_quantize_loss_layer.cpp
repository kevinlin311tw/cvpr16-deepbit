#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MinQuantizeLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void MinQuantizeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  int * binary = new int[count];

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype loss = 0;

  for ( int i = 0; i < count; i++)
  {
	if(bottom_data[i] > 0.5){
		binary[i] = 1;
	}
	else{
		binary[i] = 0; 
	}
  }
  for (int i = 0; i < count; ++i) {
	loss = loss + (binary[i]-bottom_data[i])*(binary[i]-bottom_data[i]);
        diff_.mutable_cpu_data()[0] = diff_.mutable_cpu_data()[0] + (binary[i]-bottom_data[i])*(-bottom_data[i]);
  } 

  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num() / Dtype(2);
}

template <typename Dtype>
void MinQuantizeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];
      Dtype result = (alpha * diff_.cpu_data()[0] / bottom[i]->count()) / bottom[i]->num();
      for (int j = 0; j < bottom[i]->count(); j++)
      	bottom[i]->mutable_cpu_diff()[j] = result; 
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(MinQuantizeLossLayer);
#endif

INSTANTIATE_CLASS(MinQuantizeLossLayer);
REGISTER_LAYER_CLASS(MinQuantizeLoss);

}  // namespace caffe
