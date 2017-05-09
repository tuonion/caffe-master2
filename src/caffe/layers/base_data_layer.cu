#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

//  LOG(INFO)<<"data_string:"<<batch->data_.toString();
//  LOG(INFO)<<"label_string:"<<batch->label_.toString();
	// Reshape to loaded data.
	top[0]->ReshapeLike(batch->data_);

//	const Dtype* cpu_label = batch->label_.cpu_data();
//	ostringstream osm;
//	for (int i = 0; i < 8; ++i) {
//		for (int j = 0; j < 15; j++) {
//			osm << cpu_label[i * 15 + j] << " ";
//		}
//		osm << "\n";
//	}
//	LOG_IF(INFO,gb_logDataFlag) << "label_data:\n" << osm.str();


	// Copy the data
	caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
	if (this->output_labels_) {
		// Reshape to loaded labels.
		top[1]->ReshapeLike(batch->label_);
		// Copy the labels.
		caffe_copy(batch->label_.count(), batch->label_.gpu_data(), top[1]->mutable_gpu_data());
	}
	// Ensure the copy is synchronous wrt the host, so that the next batch isn't
	// copied in meanwhile.
	CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
	prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
