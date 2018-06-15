//
// Created by JeaSung Park on 2018. 6. 13..
//

#ifndef QUANTIZATION_DATABASE_KNN_H
#define QUANTIZATION_DATABASE_KNN_H
// C++ HEADERS
#include <cfloat>
#include <map>
#include <iostream>
// OPENCV HEADERS
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
// Namespace declaration
namespace cv
{
    namespace ml
    {
        class CV_EXPORTS_W QuantizedKNearest : public StatModel
        {
        public:
            CV_WRAP virtual int getDefaultK() const = 0;
            CV_WRAP virtual void setDefaultK(int val) = 0;

            CV_WRAP virtual bool getIsClassifier() const = 0;
            CV_WRAP virtual void setIsClassifier(bool val) = 0;

            CV_WRAP virtual int getEmax() const = 0;
            CV_WRAP virtual void setEmax(int val) = 0;

            CV_WRAP virtual int getAlgorithmType() const = 0;
            CV_WRAP virtual void setAlgorithmType(int val) = 0;

            CV_WRAP virtual float findNearest(InputArray samples,
                                              int k,
                                              OutputArray results,
                                              OutputArray neighborResponses=noArray(), OutputArray dist=noArray()) const = 0;
            enum Types
            {
                BRUTE_FORCE = 1, QUANTIZED = 2
            };

            CV_WRAP static Ptr<QuantizedKNearest> create();
        };

        const String NAME_BRUTE_FORCE = "opencv_ml_knn";
        const String NAME_QUANTIZED = "opencv_ml_knn_quantized";

        class Impl
        {
        public:

            Impl()
            {
                defaultK = 10;
                isclassifier = true;
                Emax = INT_MAX;
            }

            virtual ~Impl()
            {

            }

            virtual String getModelName() const = 0;
            virtual int getType() const = 0;
            virtual float findNearest(InputArray _samples,
                                      int k,
                                      OutputArray _result,
                                      OutputArray _neighborResponses,
                                      OutputArray _dists) const = 0;

            bool train(const Ptr<TrainData> &data, int flags)
            {
                Mat new_samples = data->getTrainSamples(ROW_SAMPLE);
                Mat new_responses;
                data->getTrainResponses().convertTo(new_responses, CV_32F);
                bool update = (flags & ml::QuantizedKNearest::UPDATE_MODEL) != 0 && !samples.empty();

                CV_ASSERT(new_samples.type() == CV_32F);

                if (!update)
                {
                    clear();
                }
                else
                {
                    CV_ASSERT(new_samples.cols == samples.cols &&
                              new_responses.cols == responses.cols);
                }

                samples.push_back(new_samples);
                responses.push_back(new_responses);

                doTrain(samples);

                return true;
            }

            virtual void doTrain(InputArray points)
            {
                (void)points;
            }

            void clear()
            {
                samples.release();
                responses.release();
            }

            void read( const FileNode& fn )
            {
                clear();
                isclassifier = (int)fn["is_classifier"] != 0;
                defaultK = (int)fn["default_k"];

                fn["samples"] >> samples;
                fn["responses"] >> responses;
            }

            void write( FileStorage& fs ) const
            {
                fs << "is_classifier" << (int)isclassifier;
                fs << "default_k" << defaultK;

                fs << "samples" << samples;
                fs << "responses" << responses;
            }

        public:
            int defaultK;
            bool isclassifier;
            int Emax;

            Mat samples;
            Mat responses;
        };
    }
}

#endif //QUANTIZATION_DATABASE_KNN_H
