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
// PYTHON BINDING
#include <Python.h>
#include <numpy/arrayobject.h>

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

        int read_data(const std::string _filename, Mat &batch)
        {
            pArgs = PyTuple_New(1);
            pName = PyUnicode_FromString(_filename.c_str());

            if (!pName)
            {
                Py_DECREF(pArgs);
                Py_DECREF(pModule);

                fprintf(stderr, "Cannot convert argument");

                return -1;
            }

            PyTuple_SetItem(pArgs, 0, pName);
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pName);
            Py_DECREF(pArgs);

            if (PyList_Check(pValue))
            {
                size_t elements = PyList_GET_SIZE(pValue);
                int buffer[3][32][32];
                Size dims(32, 32);

                for (size_t i = 0; i < elements; i++)
                {
                    PyObject *cur = PyList_GET_ITEM(pValue, i);
                    size_t index = 0;

                    if (PyList_Check(cur))
                    {
                        Mat image;
                        for (int channel = 0; channel < 3; ++channel) {
                            for (int x = 0; x < 32; ++x) {
                                for (int y = 0; y < 32; ++y) {
                                    PyObject *pixel = PyList_GET_ITEM(cur, index);
                                    buffer[channel][x][y] = (int)PyFloat_AS_DOUBLE(pixel);
                                    index = index + 1;
                                }
                            }
                            Mat each_channel(dims, CV_32F, buffer[channel]);
                            image.push_back(each_channel);
                        }
                        batch.push_back(image);
                        Py_XDECREF(cur);
                    } else {
                        Py_XDECREF(cur);
                        return -1;
                    }
                }
                return 1;
            }
            else
            {
                if (pValue->ob_refcnt > 0)
                {
                    Py_DECREF(pValue);
                }
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

#endif //QUANTIZATION_DATABASE_DATA_READ_H
