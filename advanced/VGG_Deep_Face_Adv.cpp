#include "VGGDeepFace.h"

/*

Method : initNetwork
Read network architecture from model_file and apply weights from weight_file

*/
void VGGDeepFaceNet::initNet()
{
	NetParameter net_param;
	ReadNetParamsFromTextFileOrDie(this->model_file, &net_param);
	this->vgg_net = make_shared<Net<Dtype>>(net_param);
	this->vgg_net->CopyTrainedLayersFromBinaryProto(weight_file);	
}

/*

Method : wrapInputBlobWithMat
Make blob using cv::Mat and push it to the input of the vgg_net

Parameter :
1. _img [in] : Input image

*/
void VGGDeepFaceNet::wrapInputBlobWithMat(Mat _img)
{
	Mat resized_img;
	Mat normalized_img;

	cv::resize(_img, resized_img, cv::Size(this->img_height, this->img_width));
	cv::subtract(resized_img, this->mean_img, normalized_img);

	TransformationParameter trans_param;
	DataTransformer<Dtype> transformer(trans_param, caffe::TEST);
	transformer.Transform(normalized_img, this->vgg_net->input_blobs()[0]);
}

/*

Method : forwardPropagation
Do forward propagation

*/
void VGGDeepFaceNet::forwardPropagation()
{
	this->vgg_net->Forward();
}

/*

Method : getFeatureVec
Get feature from the fc7 layer.

Parameter :
1. _feature_vector [out] : output feature vector

*/
void VGGDeepFaceNet::getFeatureVec(vector<Dtype>& _feature_vec)
{
	boost::shared_ptr<Blob<Dtype>> feature_blob = this->vgg_net->blob_by_name("fc7");
	Dtype * feature_val = feature_blob->mutable_cpu_data();
	
	int feature_num = feature_blob->count();
	for (int i = 0; i < feature_num; i++)
		_feature_vec.push_back(feature_val[i]);
}

/*

Method : getClassificationResult
VGG Deep Face has 2622 classes (see the model prototxt file).
Caluculate the probabilities and get the highest one.

Parameter :
1. _prob_vec [out] : probabilities for each class.

Return :
The class number which has the highest probability

*/
int VGGDeepFaceNet::getClassificationResult(vector<Dtype>& _prob_vec)
{
	vector<Blob<Dtype> *> output_blob_vec = this->vgg_net->output_blobs();
	Blob<Dtype> * output_blob = output_blob_vec[0];
	Dtype * prob = output_blob->mutable_cpu_data();
	
	for (int i = 0; i < output_blob->count(); i++)
		_prob_vec.push_back(prob[i]);

	auto maximum_iter = std::max_element(_prob_vec.begin(), _prob_vec.end(), [](Dtype i, Dtype j)->bool {return (i < j); });	
	return maximum_iter - _prob_vec.begin();
}

/*

Method : getSimilarities
Get the similarity of two feature vectors using 'cosine similarity'.

Parameter :
1. _feature_vec1 [in] : first feature vector
2. _feature_vec2 [out] : second feature vector

Return :
The similarity

*/
double VGGDeepFaceNet::getSimilarities(vector<Dtype> _feature_vec1, vector<Dtype> _feature_vec2)
{
	double dot_product = 0;

	double len1 = 0;
	double len2 = 0;
	double multiply_len = 0;

	for (int i = 0; i < _feature_vec1.size(); i++)
		dot_product += _feature_vec1[i] * _feature_vec2[i];

	std::for_each(_feature_vec1.begin(), _feature_vec1.end(), [&len1](Dtype i) {len1 += (i*i); });
	std::for_each(_feature_vec2.begin(), _feature_vec2.end(), [&len2](Dtype i) {len2 += (i*i); });

	len1 = sqrt(len1);
	len2 = sqrt(len2);

	multiply_len = len1 * len2;

	double cos_val = dot_product / multiply_len;
	return cos_val;
}

/*

Method : getEuclideanDistance
Get the euclidean distance between two feature vectors.

Parameter :
1. _feature_vec1 [in] : first feature vector
2. _feature_vec2 [out] : second feature vector

Return :
The similarity

*/
double VGGDeepFaceNet::getEuclideanDistance(vector<Dtype> _feature_vec1, vector<Dtype> _feature_vec2)
{
	double distance = 0;

	for (int i = 0; i < _feature_vec1.size(); i++)
		distance += pow(_feature_vec1[i] - _feature_vec2[i], 2);
	
	return sqrt(distance);
}

/*

Method : findPersonInDB
Find the person with the most similar looking of the given face image.

Parameter :
1. _DB_path [in] : DB image path
2. _stranger [in] : Face image of the person. We want to find this guy in the DB.
3. top_five_vec [out] : Vector of pairs of image and similarity 
*/
void VGGDeepFaceNet::findPersonInDB(string _DB_path, Mat _stranger, vector<pair< Mat, double>>& top_five_vec)
{
	// vector of Euclidean distances between a stranger and all in the DB.
	//vector <pair<string, double>> result_vec;
	vector<tuple<string, double, double>> result_vec;

	// Get feature vector of stranger
	vector<Dtype> feat_vec_stranger;

	this->wrapInputBlobWithMat(_stranger);
	this->forwardPropagation();
	this->getFeatureVec(feat_vec_stranger);

	// Handle and structure for searching all directories in _DB_path
	intptr_t handle_img;
	_finddata_t fd_img;

	// Ready for searching all directories in _DB_path
	handle_img = _findfirst((_DB_path + "/*.*").c_str(), &fd_img);
	_findnext(handle_img, &fd_img);

	// Search all directories in _DB_path
	int cnt = 0;
	while (_findnext(handle_img, &fd_img) == 0)
	{
		string img_name(fd_img.name);
		string img_path(_DB_path + "/" + img_name);

		Mat db_img = imread(img_path, CV_LOAD_IMAGE_COLOR);

		// Get feature vec of db
		vector<Dtype> feat_vec_db;
		this->wrapInputBlobWithMat(db_img);
		this->forwardPropagation();
		this->getFeatureVec(feat_vec_db);
		
		// Get the Euclidean distance and save it.
		double distance = this->getEuclideanDistance(feat_vec_stranger, feat_vec_db);
		double similarity = this->getSimilarities(feat_vec_stranger, feat_vec_db);
		
		std::tuple<string, double, double> path_dist_sim =
			std::make_tuple(img_path, distance, similarity);

		result_vec.push_back(path_dist_sim);

		cout << "Checking DB : " << img_name << endl;
	}

	std::sort(result_vec.begin(), result_vec.end(), [](tuple<string, double, double> t1, tuple<string, double, double> t2) -> bool {
		return (std::get<1>(t1) < std::get<1>(t2));
	});


	for (int i = 0; i < 5; i++)
	{
		Mat img = imread(std::get<0>(result_vec[i]), CV_LOAD_IMAGE_COLOR);
		double sim = std::get<2>(result_vec[i]);
		std::pair<Mat, double> top_fives = make_pair(img, sim);
		top_five_vec.push_back(top_fives);
	}
}