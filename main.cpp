#include <iostream>
#include <string>

#include <data_read.h>


const std::string data_name("../data/val/val_data");
const std::string file_name("load_data");

int main(int argc, const char *argv[])
{
    cv::Pickle *obj = new cv::Pickle(file_name);

    cv::Mat data;
    obj->read_data(data_name, data);

    delete obj;

    return 0;
}