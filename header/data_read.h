//
// Created by JeaSung Park on 2018. 6. 15..
//

#ifndef QUANTIZATION_DATABASE_DATA_READ_H
#define QUANTIZATION_DATABASE_DATA_READ_H
// C HEADERS
#include <stdlib.h>
#include <fcntl.h>
// C++ HEADERS
#include <iostream>
#include <string>
// OPENCV HEADERS
#include <opencv2/opencv.hpp>

namespace cv
{
    class DataReader
    {
    public:

        explicit DataReader(int _flags) : flags(_flags)
        {

        }

        void read(std::string filename, Mat &res)
        {
            try
            {
                float length = 0;

                FileStorage fs(filename, FileStorage::Mode::FORMAT_XML | FileStorage::Mode::READ);

                fs["size"] >> length;
                for (int i = 0; i < (int)length; ++i) {
                    std::string key("image");
                    std::string index = std::to_string(i);
                    Mat image;

                    key = key + index;
                    fs[key.c_str()] >> image;
                    std::cout << image.channels() << std::endl;
                    res.push_back(image);
                }

                fs.release();
            }
            catch (Exception e)
            {
                std::cerr << e.msg << std::endl;
            }

        }

    private:
        int flags;
    };
}


/*
namespace cv
{
    class Pickle
    {
    public:
        explicit Pickle(std::string _filename)
        {
            this->filename = _filename;
            if (!init_pickle())
            {
                exit(1);
            }
        }

        ~Pickle()
        {

        }

        bool init_pickle()
        {
            setenv("PYTHONPATH",
                   "/Users/jeasungpark/CLionProjects/Quantization Database/python",
                   1);
            Py_Initialize();

            pModule = PyImport_ImportModule(this->filename.c_str());

            if (pModule != NULL)
            {
                pFunc = PyObject_GetAttrString(pModule, "unpickle");

                if (pFunc && PyCallable_Check(pFunc))
                {
                    return true;
                }
                else
                {
                    Py_XDECREF(pFunc);
                    if (PyErr_Occurred())
                    {
                        PyErr_Print();
                    }
                    fprintf(stderr, "Cannot find function \"%s\"\n", "unpickle");
                    return false;
                }
            }
            Py_XDECREF(pModule);
            return false;
        }

        int read_data(const std::string _filename, Mat &batch, int width, int height, int n_channel)
        {
            Size image_size(width, height);

            pArgs = PyTuple_New(1);
            pName = PyUnicode_FromString(_filename.c_str());

            if (!pName)
            {
                Py_DECREF(pArgs);
                fprintf(stderr, "Cannot convert argument");
                return -1;
            }

            PyTuple_SetItem(pArgs, 0, pName);
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pName);
            Py_DECREF(pArgs);

            if (pValue != nullptr && PyList_CheckExact(pValue))
            {

            }
            else
            {
                Py_DECREF(pValue);
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr, "Call failed\n");
                return -1;
            }
        }


    private:
        PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
        std::string filename;
    };
}
 */

#endif //QUANTIZATION_DATABASE_DATA_READ_H
