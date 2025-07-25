# include <stdio.h>
# include <opencv2/opencv.hpp>

int main()
{
	printf("Hello\n");

	cv::Mat image = cv::imread(RESOURCE_DIR"zidane.jpg");
	cv::imshow("Display", image);

	cv::waitKey(0);
	return 0;
}
