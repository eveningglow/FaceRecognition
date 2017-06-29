//#include <caffe/caffe.hpp>
//#include <caffe/data_transformer.hpp>
//
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//
//#include <iostream>
//#include <string>
//#include <vector>
//
//using namespace std;
//using namespace cv;
//using namespace caffe;
//
//typedef double Dtype;
//
//int main(int argc, char * argv[])
//{
//	Caffe::set_mode(Caffe::GPU);
//
//	// Model file(prototxt), weight file(caffemodel) and test image file path
//	string model_path(argv[1]);
//	string weight_path(argv[2]);
//	string img_path(argv[3]);
//
//	Mat img = imread(img_path, CV_LOAD_IMAGE_COLOR);
//
//	// Make Net using model file
//	NetParameter net_param;
//	ReadNetParamsFromTextFileOrDie(model_path, &net_param);
//	std::shared_ptr<Net<Dtype>> vgg_net = make_shared<Net<Dtype>>(net_param);
//
//	// Apply trained weight to the Net
//	vgg_net->CopyTrainedLayersFromBinaryProto(weight_path);
//
//	// Normalize test image using mean image (mean value is given in the matlab code)
//	Mat normalized_img;
//	cv::Scalar mean_val(129.1863, 104.7624, 93.5940);	// These values are given by the MATLAB example.
//	Mat mean_img(224, 224, CV_8UC3, mean_val);
//	cv::subtract(img, mean_img, normalized_img);
//
//	// Convert cv::Mat to Blob and push it to the input of the Net
//	TransformationParameter trans_param;
//	DataTransformer<Dtype> transformer(trans_param, caffe::TEST);
//	Blob<Dtype> * input_blob = vgg_net->input_blobs()[0];
//	transformer.Transform(normalized_img, input_blob);
//
//	// Forward Propagation
//	vgg_net->Forward();
//
//	// Get feature vector
//	boost::shared_ptr<Blob<Dtype>> feature_blob = vgg_net->blob_by_name("fc7");
//	Dtype * feature = feature_blob->mutable_cpu_data();
//	vector<Dtype> feature_vec;
//
//	for (int i = 0; i < 4096; i++)
//		feature_vec.push_back(feature[i]);
//}
