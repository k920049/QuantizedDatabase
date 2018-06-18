#include <iostream>
#include <string>

#include <data_read.h>
#include <opencv2/xfeatures2d.hpp>


const std::string data_name("../data/converted/train0.xml");
const std::string file_name("load_data");

int main(int argc, const char *argv[])
{
    int flags = 10;
    cv::DataReader reader(flags);
    cv::Mat tmp, cur;

    reader.read(data_name, tmp);

    return 0;
}
