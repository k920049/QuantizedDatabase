//
// Created by JeaSung Park on 2018. 6. 18..
//

#ifndef QUANTIZATION_DATABASE_FEATURE_EXTRACT_H
#define QUANTIZATION_DATABASE_FEATURE_EXTRACT_H
// C++ HEADERS
#include <iostream>
#include <vector>
// OPENCV HEADERS
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace cv
{
    class FeatureExtract
    {
    public:

        FeatureExtract(Mat &_img,
                       int _width,
                       int _height,
                       int _channel,
                       int _features) : m_img(_img), m_width(_width), m_height(_height), m_channel(_channel), n_features(_features)
        {
            // initialize how many images are there
            m_size = m_img.size[0];
            m_buffer.resize(m_size);
            // build a model that generates SIFT features
            model = xfeatures2d::SIFT::create(n_features, 8);
        }

        ~FeatureExtract()
        {
            model.release();
        }

        bool extract(std::vector<std::vector<KeyPoint>> &key, Mat &descriptor)
        {
            if (!model)
            {
                std::cerr << "Error: The model hasn't been created" << std::endl;
                return false;
            }

            try
            {
                model->detect(m_img, key, descriptor);

            }
            catch (Exception e)
            {
                std::cerr << e.msg << std::endl;
                return false;
            }

            return true;
        }

    private:
        int m_width, m_height, m_channel, n_features, m_size;

        Mat &m_img;
        std::vector<cv::Mat> m_buffer;
        Ptr<xfeatures2d::SIFT> model;
    };
}

#endif //QUANTIZATION_DATABASE_FEATURE_EXTRACT_H
