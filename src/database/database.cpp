//
// Created by JeaSung Park on 2018. 6. 13..
//

#include "../../header/database.h"

namespace cv
{
    BOWQuantized::BOWQuantized(int _clusterCount,
                               const cv::TermCriteria &_termcrit,
                               int _attempts,
                               int _flags) : clusterCount(_clusterCount), termcrit(_termcrit), attempts(_attempts), flags(_flags)
    {

    }

    BOWQuantized::~BOWQuantized()
    {

    }

    Mat BOWQuantized::cluster() const
    {
        CV_ASSERT(!descriptors.empty());

        Mat mergedDescriptors(descriptorsCount(), descriptors[0].cols, descriptors[0].type());

        for (size_t i = 0, start = 0; i < descriptors.size(); ++i)
        {
            Mat submut = mergedDescriptors.rowRange((int)start, (int)(start + descriptors[i].rows));
            descriptors[i].copyTo(submut);
            start += descriptors[i].rows;
        }

        return cluster(mergedDescriptors);
    }

    Mat BOWQuantized::cluster(const cv::Mat &_descriptors) const
    {
        Mat labels, vocabulary;
        kmeans(_descriptors, clusterCount, labels, termcrit, attempts, flags, vocabulary);
        return vocabulary;
    }


}
