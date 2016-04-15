#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EvenDistLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EvenDistLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  //LOG(ERROR)<< "batch size:"<< bottom[0]->num();//batch size, defined in the network
  //LOG(ERROR)<< "batch channel:"<< bottom[0]->channels();//feature length, 4096
  //LOG(ERROR)<< "batch total nodes:"<< bottom[0]->count();//channels*size
  int batch_size = bottom[0]->num();
  int feat_length = bottom[0]->channels();
  //int count = bottom[0]->count();
  //const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype loss = 0;
  Dtype sum = 0;
  Dtype avg = 0;
  Dtype diff = 0;
  for( int i = 0; i < feat_length; i++){
	sum = 0;
	for(int j = 0; j < batch_size; j++){
		float data = bottom[0]->cpu_data()[j*bottom[0]->channels()+i];
		int binary = 0;
		if(data>0.5){
			binary = 1;
		}else{
			binary = 0;
		}
		sum = sum + binary;
	}
	avg = sum / batch_size;
	loss = loss + ((avg - Dtype(0.5)) * (avg - Dtype(0.5)));
        diff = diff + avg - Dtype(0.5);
  }
  //avg = sum / bottom[0]->count();
  //loss = loss + ((avg - Dtype(0.5)) * (avg - Dtype(0.5)));
  //diff = diff + avg - Dtype(0.5);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num() / Dtype(2);


  diff_.mutable_cpu_data()[0] = diff;

}

template <typename Dtype>
void EvenDistLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(EvenDistLossLayer);
#endif

INSTANTIATE_CLASS(EvenDistLossLayer);
REGISTER_LAYER_CLASS(EvenDistLoss);

}  // namespace caffe
