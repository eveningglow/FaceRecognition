#include "VGGDeepFace.h"

Mat makeResultImage(Mat _stranger, vector<pair<Mat, double>>& _mat_sim_vec)
{
	// Positions and sizes
	int interval = 40;
	int img_width = 224;
	int img_height = 224;
	int full_width = (img_width * 5) + (interval * 2);
	int full_height = (img_height * 2) + (interval * 3);

	cv::resize(_stranger, _stranger, cv::Size(img_width, img_height));

	// Full image
	Mat full_img(full_height, full_width, CV_8UC3, cv::Scalar(0, 0, 0));

	// Set stranger image to full image
	int stranger_x = (full_width - img_width) / 2;
	int stranger_y = interval;
	Mat stranger_pos(full_img, cv::Rect(stranger_x, stranger_y, img_width, img_height));

	cv::rectangle(_stranger, cv::Rect(0, 0, 224, 224), cv::Scalar(255, 255, 255), 10);
	cv::putText(full_img, "Target", cv::Point(stranger_x, stranger_y - 10),
		cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
	_stranger.copyTo(stranger_pos);

	int line_x = 0;
	int line_y = img_height + (interval * (3 / 2));

	// Set top-5-images to full image
	for (int i = 0; i < 5; i++)
	{
		cv::resize(_mat_sim_vec[i].first, _mat_sim_vec[i].first, cv::Size(img_width, img_height));
		int top_five_x = interval + (img_width * i);
		int top_five_y = img_height + (2 * interval);
		Mat top_five_pos(full_img, cv::Rect(top_five_x, top_five_y, img_width, img_height));

		//cv::rectangle(_top_five_similar[i], cv::Rect(0, 0, 224, 224), cv::Scalar(255, 255, 255), 10);
		cv::rectangle(_mat_sim_vec[i].first, cv::Rect(0, 0, 224, 224), cv::Scalar(255, 255, 255), 10);

		// Below is how to add similarities to the result, but it is cosine similarities whic is not that accurate.
		// So I decided not to add these to result.

		//string sim;
		//std::ostringstream oss(sim);
		//oss.setf(std::ios::fixed, std::ios::floatfield);
		//oss.precision(2);
		//oss << _mat_sim_vec[i].second;
		//sim = oss.str();

		cv::String text("Top " + to_string(i + 1) /*+ " (" + sim + " %)"*/);
		cv::putText(full_img, text, cv::Point(top_five_x, top_five_y - 10),
			cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

		_mat_sim_vec[i].first.copyTo(top_five_pos);
	}

	return full_img;
}

int main(int argc, char * argv[])
{
	if (argc != 5)
	{
		cout << "Command should be like... " << endl;
		cout << "VGGDeepFace.exe \"MODEL_PATH\" \"WEIGHT_PATH\" \"TEST_IMG_PATH\" \"DB_PATH\"" << endl;

		return 0;
	}

	Caffe::set_mode(Caffe::GPU);

	string model_path(argv[1]);
	string weight_path(argv[2]);
	string test_img_path(argv[3]);
	string db_path(argv[4]);

	vector<Dtype> feature_vec1;
	vector<Dtype> feature_vec2;

	vector<Dtype> prob_vec;

	Mat test_img = imread(test_img_path, CV_LOAD_IMAGE_COLOR);

	VGGDeepFaceNet vgg(model_path, weight_path);
	vgg.initNet();

	vector<pair<Mat, double>> top_five_similar;
	vgg.findPersonInDB(db_path, test_img, top_five_similar);

	Mat result_img = makeResultImage(test_img, top_five_similar);
	imshow("Result", result_img);
	waitKey(0);
}