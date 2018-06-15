//
// Created by JeaSung Park on 2018. 6. 13..
//

#include "../../header/knn.h"
#include <opencv2/ml.hpp>

namespace cv
{
    namespace ml
    {
        class BruteForceImpl : public Impl
        {
        public:
            String getModelName() const CV_OVERRIDE
            {
                return NAME_BRUTE_FORCE;
            }

            int getType() const CV_OVERRIDE
            {
                return ml::QuantizedKNearest::BRUTE_FORCE;
            }

            void findNearestCore(const Mat &_samples,
                                 int k0,
                                 const Range &range,
                                 Mat *results,
                                 Mat *neighbor_responses,
                                 Mat *dist,
                                 float * presult) const
            {
                int testidx, baseidx, i, j, d = samples.cols, nsamples = samples.rows;
                int testcount = range.end - range.start;
                int k = std::min(k0, nsamples);

                AutoBuffer<float> buf((size_t)testcount * k * 2);
                float *dbuf = buf;
                float *rbuf = dbuf + testcount * k;

                const float *rptr = responses.ptr<float>();

                for (testidx = 0; testidx < testcount; testidx++)
                {
                    for (i = 0; i < k; i++)
                    {
                        dbuf[testidx * k + i] = FLT_MAX;
                        rbuf[testidx * k + i] = 0.f;
                    }
                }

                for (baseidx = 0; baseidx < nsamples; baseidx++)
                {
                    for (testidx = 0; testidx < testcount; testidx++)
                    {
                        const float *v = samples.ptr<float>(baseidx);
                        const float *u = _samples.ptr<float>(testidx + range.start);

                        float s = 0;
                        for (i = 0; i <= d - 4; i += 4)
                        {
                            float t0 = u[i] - v[i];
                            float t1 = u[i + 1] - v[i + 1];
                            float t2 = u[i + 2] - v[i + 2];
                            float t3 = u[i + 3] - v[i + 3];
                            s += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
                        }

                        for ( ; i < d; i++)
                        {
                            float t0 = u[i] - v[i];
                            s += t0 * t0;
                        }

                        Cv32suf si;
                        si.f = s;
                        Cv32suf *dd = (Cv32suf*)(&dbuf[testidx * k]);
                        float *nr = &rbuf[testidx * k];

                        for (i = k; i > 0; i--)
                        {
                            if (si.i >= dd[i - 1].i)
                            {
                                break;
                            }
                        }

                        if (i >= k)
                        {
                            continue;
                        }

                        for (j = k - 2; j >= i; j--)
                        {
                            dd[j + 1].i = dd[j].i;
                            nr[j + 1] = nr[j];
                        }

                        dd[i].i = si.i;
                        nr[i] = rptr[baseidx];
                    }
                }

                float result = 0.f;
                float inv_scale = 1.f / k;

                for (testidx = 0; testidx < testcount; testidx++)
                {
                    if( neighbor_responses )
                    {
                        float* nr = neighbor_responses->ptr<float>(testidx + range.start);
                        for( j = 0; j < k; j++ )
                            nr[j] = rbuf[testidx*k + j];
                        for( ; j < k0; j++ )
                            nr[j] = 0.f;
                    }

                    if( dists )
                    {
                        float* dptr = dists->ptr<float>(testidx + range.start);
                        for( j = 0; j < k; j++ )
                            dptr[j] = dbuf[testidx*k + j];
                        for( ; j < k0; j++ )
                            dptr[j] = 0.f;
                    }

                    if (results || testidx + range.start == 0)
                    {
                        if (!isclassifier || k == 1)
                        {
                            float s = 0.f;
                            for (j = 0; j < k; j++)
                            {
                                s += rbuf[testidx * k + j];
                            }
                            result = (float)(s * inv_scale);
                        }
                        else
                        {
                            float *rp = rbuf + testidx * k;
                            std::sort(rp, rp + k);

                            result = rp[0];
                            int prev_start = 0;
                            int best_count = 0;

                            for (j = 1; j <= k; j++)
                            {
                                if (j == k || rp[j] != rp[j - 1])
                                {
                                    int count = j - prev_start;
                                    if (best_count < count)
                                    {
                                        best_count = count;
                                        result = rp[j - 1];
                                    }
                                    prev_start = j;
                                }
                            }
                        }

                        if (results)
                        {
                            results->at<float>(testidx + range.start) = result;
                        }
                        if (presult && testidx + range.start == 0)
                        {
                            *presult = result;
                        }
                    }
                }
            }

            struct findKNearestInvoker : public ParallelLoopBody
            {

                findKNearestInvoker(const BruteForceImpl *_p,
                                    int _k,
                                    const Mat &__samples,
                                    Mat *__results,
                                    Mat *__neighbor_response,
                                    Mat *__dists,
                                    float *_presult)
                {
                    p = _p;
                    k = _k;
                    _samples = &__samples;
                    _results = __results;
                    _neighbor_responses = __neighbor_response;
                    _dists = __dists;
                    presult = _presult;
                }

                void operator ()(const Range &range) const
                {
                    int delta = std::min(range.end - range.start, 256);

                    for (int start = range.start; start < range.end; start += delta)
                    {
                        p->findNearestCore(*_samples,
                                           k,
                                           Range(start, std::min(start + delta, range.end)),
                                           _results,
                                           _neighbor_responses,
                                           _dists,
                                           presult);
                    }
                }

                const BruteForceImpl *p;
                int k;
                const Mat *_samples;
                Mat *_results;
                Mat *_neighbor_responses;
                Mat *_dists;
                float *presult;
            };

            float findNearest( InputArray _samples,
                               int k,
                               OutputArray _results,
                               OutputArray _neighborResponses,
                               OutputArray _dists ) const
            {
                float result = 0.f;
                CV_ASSERT(0 < k);

                Mat test_samples = _samples.getMat();
                CV_ASSERT(test_samples.type() == CV_32F && test_samples.cols == samples.cols);
                int testcount = test_samples.rows;

                if (testcount == 0)
                {
                    _results.release();
                    _neighborResponses.release();
                    _dists.release();
                    return 0.f;
                }

                Mat res, nr, d, *pres = 0, *pnr = 0, *pd = 0;
                if( _results.needed() )
                {
                    _results.create(testcount, 1, CV_32F);
                    pres = &(res = _results.getMat());
                }
                if( _neighborResponses.needed() )
                {
                    _neighborResponses.create(testcount, k, CV_32F);
                    pnr = &(nr = _neighborResponses.getMat());
                }
                if( _dists.needed() )
                {
                    _dists.create(testcount, k, CV_32F);
                    pd = &(d = _dists.getMat());
                }

                findKNearestInvoker invoker(this, k, test_samples, pres, pnr, pd, &result);
                parallel_for_(Range(0, testcount), invoker);
                //invoker(Range(0, testcount));
                return result;
            }
        };

        class QuantizedImpl : public Impl
        {
        public:

            String getModelName() const
            {
                return NAME_QUANTIZED;
            }

            int getType() const
            {
                return ml::QuantizedKNearest::QUANTIZED;
            }

            void doTrain(InputArray points)
            {

            }

            float findNearest(InputArray _samples,
                              int k,
                              OutputArray _results,
                              OutputArray _neighborResponses,
                              OutputArray _dists ) const
            {

            }

        };

        class QuantizedKNearestImpl : public QuantizedKNearest
        {
            inline int getDefaultK() const
            {
                return impl->defaultK;
            }
            inline void setDefaultK(int val)
            {
                impl->defaultK = val;
            }
            inline bool getIsClassifier() const
            {
                return impl->isclassifier;
            }
            inline void setIsClassifier(bool val)
            {
                impl->isclassifier = val;
            }
            inline int getEmax() const
            {
                return impl->Emax;
            }
            inline void setEmax(int val)
            {
                impl->Emax = val;
            }

        public:
            int getAlgorithmType() const
            {
                return impl->getType();
            }

            void setAlgorithmType(int val)
            {
                if (val != BRUTE_FORCE && val != KDTREE)
                {
                    val = BRUTE_FORCE;
                }

                int k = getDefaultK();
                int e = getEmax();
                bool c = getIsClassifier();

                initImpl(val);

                setDefaultK(k);
                setEmax(e);
                setIsClassifier(c);
            }

        public:
            QuantizedKNearestImpl()
            {
                initImpl(BRUTE_FORCE);
            }

            ~QuantizedKNearestImpl()
            {

            }

            bool isClassifier() const
            {
                return impl->isclassifier;
            }

            bool isTrained() const
            {
                return !impl->samples.empty();
            }

            int getVarCount() const
            {
                return impl->samples.cols;
            }

            void write(FileStorage &fs) const
            {
                writeFormat(fs);
                impl->write(fs);
            }

            void read(const FileNode &fn)
            {
                int algorithmType = BRUTE_FORCE;
                if (fn.name() == NAME_QUANTIZED)
                {
                    algorithmType = QUANTIZED;
                }
                initImpl(algorithmType);
                impl->read(fn);
            }

            float findNearest(InputArray samples,
                              int k,
                              OutputArray results,
                              OutputArray neighborResponses=noArray(),
                              OutputArray dist=noArray()) const
            {
                return impl->findNearest(samples, k, results, neighborResponses, dist);
            }

            float predict (InputArray inputs, OutputArray outputs, int temp) const
            {
                return impl->findNearest(inputs, impl->defaultK, outputs, noArray(), noArray());
            }

            bool train(const Ptr<TrainData> &data, int flags)
            {
                return impl->train(data, flags);
            }

            String getDefaultName() const
            {
                return impl->getModelName();
            }

        protected:
            void initImpl(int algorithmType)
            {
                if (algorithmType != QUANTIZED)
                {
                    impl = makePtr<BruteForceImpl>();
                }
                else
                {
                    impl = makePtr<QuantizedImpl>();
                }
            }
            Ptr<Impl> impl;
        };

        Ptr<QuantizedKNearest> QuantizedKNearest::create()
        {
            return makePtr<QuantizedKNearestImpl>();
        }

    }
}

