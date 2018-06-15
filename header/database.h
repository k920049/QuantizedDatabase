//
// Created by JeaSung Park on 2018. 6. 11..
//

#ifndef QUANTIZATION_DATABASE_DATABASE_H
#define QUANTIZATION_DATABASE_DATABASE_H
// C HEADERS
#include <stdio.h>
#include <stdlib.h>
// C++11 HEADERS
#include <iostream>
#include <algorithm>
// STL HEADERS
#include <vector>
#include <map>
// LINUX SYSTEM HEADERS
#include <sys/fcntl.h>
// OPENCV HEADERS
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>

namespace cv
{
class CV_EXPORTS_W BOWQuantized : public BOWTrainer
{
public:

    CV_WRAP BOWQuantized(int clusterCount, const TermCriteria &termcrit = TermCriteria(), int attempts = 3, int flags = KMEANS_PP_CENTERS);
    virtual ~BOWQuantized();

    CV_WRAP virtual Mat cluster() const CV_OVERRIDE;
    CV_WRAP virtual Mat cluster(const Mat &descriptors) const CV_OVERRIDE;

protected:

    int clusterCount;
    TermCriteria termcrit;
    int attempts;
    int flags;
};
}

#endif //QUANTIZATION_DATABASE_DATABASE_H
