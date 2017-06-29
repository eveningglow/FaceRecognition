#pragma once
#include <caffe/caffe.hpp>
#include <caffe/data_transformer.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <io.h>

using namespace std;
using namespace cv;
using namespace caffe;

typedef double Dtype;

/*
class : VGGDeepFaceNet

This class is providing all methods for face recognition using VGG Deep Face.
*/
class VGGDeepFaceNet
{
private:
	string	model_file;		// prototxt file path
	string	weight_file;	// caffemodel file path

	Mat		mean_img;		// mean img
	int		img_height;		// image height for VGG Deep Face
	int		img_width;		// image width for VGG Deep Face

	std::shared_ptr<Net<Dtype>> vgg_net;	// VGG Deep Face Network

public:

	// Constructor
	VGGDeepFaceNet(string _model, string _weight) : img_height(224), img_width(224)
	{
		model_file.assign(_model);
		weight_file.assign(_weight);

		cv::Scalar mean_scalar(129.1863, 104.7624, 93.5940);
		mean_img = Mat(224, 224, CV_8UC3, mean_scalar);
	}

	// Initialize vgg_net
	void	initNet();

	// Make blob using cv::Mat and push it to the input of the vgg_net
	void	wrapInputBlobWithMat(Mat _img);

	// Do forward propagation
	void	forwardPropagation();

	// Get feature vector(output of fc7 layer)
	void	getFeatureVec(vector<Dtype>& _feature_vec);

	// Get classification result with probability
	int		getClassificationResult(vector<Dtype>& _prob_vec);

	// Get Euclidean distance of two feature vectors
	double	getEuclideanDistance(vector<Dtype> _feature_vec1, vector<Dtype> _feature_vec2);

	// Get similarity of two feature vectors by 'cosine similarity'
	double	getSimilarities(vector<Dtype> _feature_vec1, vector<Dtype> _feature_vec2);

	// If a image is given, find the most similar person in the DB.
	//void	findPersonInDB(string _DB_path, Mat _stranger, vector<Mat>& top_five_similar);
	void	findPersonInDB(string _DB_path, Mat _stranger, vector<pair< Mat, double>>& top_five_vec);
};