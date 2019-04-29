#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <pymem.h>

#include <immintrin.h>
#include <type_traits>
#include <thread>
#include <future>
#include <vector>
#include <iostream>
#include <complex>
#include <typeinfo>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

static unsigned int nthreads = 1;

#define NUM_CORES 4

#define EXTERNC extern "C"
//T,Uの組み合わせの確認マクロ
#define type_check() \
 static_assert(((is_same<double,U>::value) && (is_same<__m256d,T>::value)) || \
    ((is_same<float,U>::value) && (is_same<__m256,T>::value)) ||\
    ((is_same<complex<float>,U>::value) && (is_same<__m256,T>::value)) ||\
    ((is_same<complex<double>,U>::value) && (is_same<__m256d,T>::value))\
    ,"type mismatch")

template<bool aligned>
static inline auto m256_load(const double* d1){
    if(aligned){
        return _mm256_load_pd(d1);
    }else{
        return _mm256_loadu_pd(d1);
    }
}
template<bool aligned>
static inline auto m256_load(const complex<double>* d1){
    if(aligned){
        return _mm256_load_pd((const double*)d1);
    }else{
        return _mm256_loadu_pd((const double*)d1);
    }
}
template<bool aligned>
static inline auto m256_load(const float* d1){
    if(aligned){
        return _mm256_load_ps(d1);
    }else{
        return _mm256_loadu_ps(d1);
    }
}
template<bool aligned>
static inline auto m256_load(const complex<float>* d1){
    if(aligned){
        return _mm256_load_ps((const float*)d1);
    }else{
        return _mm256_loadu_ps((const float*)d1);
    }
}

static inline __m256d m256_fmadd(__m256d d1, __m256d d2, __m256d d3){
    return _mm256_fmadd_pd(d1,d2,d3);
}

static inline __m256 m256_fmadd(__m256 d1, __m256 d2, __m256 d3){
    return _mm256_fmadd_ps(d1,d2,d3);
}

static inline __m256d m256_hsub(__m256d d1, __m256d d2){
    return _mm256_hsub_pd(d1,d2);
}
static inline __m256 m256_hsub(__m256 d1, __m256 d2){
    return _mm256_hsub_ps(d1,d2);
}
static inline __m256d m256_hadd(__m256d d1, __m256d d2){
    return _mm256_hadd_pd(d1,d2);
}
static inline __m256 m256_hadd(__m256 d1, __m256 d2){
    return _mm256_hadd_ps(d1,d2);
}

static inline __m256d m256_blend(__m256d d1, __m256d d2){
    return _mm256_blend_pd(d1,d2,0xa);
}

template <typename T>
static inline T m256_setzerp(){
    if(is_same<__m256d,T>::value){
        return (T)_mm256_setzero_pd();
    }else{
        return (T)_mm256_setzero_ps();
    }
}

//simd無しの内積を求める関数
template <typename U>
static inline U calc_inner_nosimd(const U* v1, const U* v2, ssize_t size){
    U result = 0;
    //cout << "calc_inner_nosimd " << size <<"\n";
    for(ssize_t i = 0; i < size; i++){
        //cout << i << " " << v1[i] << "*" << v2[i] << "=" << r << "\n";
        result += v1[i] * v2[i];
    }
    return result;
}
//
// d1.real * d2.real - d1.img * d2.img + (d1.real * d2.img + d2.real * d1.img)j
// d1 [0] [1] [2] [3]
// d2 [0] [1] [2] [3]
// ans[0] = d1[0] * d2[0] - d1[1] * d2[1]
// ans[1] = d1[0] * d2[1] + d1[1] * d2[0]
// ans[2] = d1[2] * d2[2] - d1[3] * d2[3]
// ans[3] = d1[2] * d2[3] + d1[3] * d2[2]

//実数部の積和計算
static inline auto muladd(const __m256d d1, const __m256d d2, __m256d sum){
    return m256_fmadd(d1,d2,sum);//sum=d1*d2+sum
}
static inline auto muladd(const __m256 d1, const __m256 d2, __m256 sum){
    return m256_fmadd(d1,d2,sum);//sum=d1*d2+sum
}

//虚数部の積和計算
//配列の順序を入れ替えて積和計算を行う。
//dummyは型を決定するための引数。
//doubleとfloatは虚数部がないので、そのままの値を返す。最適化で消えて無くなる。
static inline auto muladd_imag(const __m256d d1, const __m256d d2, __m256d sum, const complex<double>* dummy){
    auto d = _mm256_permute_pd(d1, 0x5);
    return m256_fmadd(d,d2,sum);
}
static inline auto muladd_imag(const __m256 d1, const __m256 d2, __m256 sum, const complex<float>* dummy){
    auto d = _mm256_permute_ps(d1, 0xb1);
    return m256_fmadd(d,d2,sum);
}
static inline auto muladd_imag(const __m256d d1, const __m256d d2, __m256d sum, const double* dummy){
    return sum;
}
static inline auto muladd_imag(const __m256 d1, const __m256 d2, __m256 sum, const float* dummy){
    return sum;
}

//YMM regから答え取り出す。型によって結果の入り方が違うので、別関数になる。
static inline auto getResult(const double sum, const __m256d sum_real, const __m256d sum_imag){
    auto real = _mm256_hadd_pd(sum_real, sum_real);
    const double* results = (const double*)&real;
    auto result = sum + results[0] + results[3];
    return result;
}

static inline auto getResult(const complex<double> sum, const __m256d sum_real, const __m256d sum_imag){
    auto real = m256_hsub(sum_real, sum_real);
    auto imag = m256_hadd(sum_imag, sum_imag);
    auto result_complex = m256_blend(real, imag);
    const complex<double>* results = (const complex<double>*)&result_complex;
    return sum + results[0] + results[1];
}

static inline auto getResult(const float sum, const __m256 sum_real, const __m256 sum_imag){
    auto real = _mm256_hadd_ps(sum_real,sum_real);
    const float* results = (const float*)&real;
    auto result = sum + results[0] + results[1] 
        + results[4] + results[5];
    return result;
}
static inline auto getResult(const complex<float> sum, const __m256 sum_real, const __m256 sum_imag){
    auto result_real = m256_hsub(sum_real, sum_real);
    auto result_imag = m256_hadd(sum_imag, sum_imag);
    const float* real = (const float*)&result_real;
    const float* imag = (const float*)&result_imag;
    return sum +  complex<float>(real[0] + real[1] + real[4] + real[5],
                    imag[0] + imag[1] + imag[4] + imag[5]);
}

//get inner product with simd instruction
template <typename T,typename U, bool aligned, bool iscomplex>
static inline U calc_inner(const U* v1, const U* v2, ssize_t size){
    type_check();
    //cout << "calc_inner size " << size << "\n";
    //number of data per SIMD register
    constexpr int step = sizeof(T) / sizeof(U);
    static_assert(step > 0,"illeagal type");
    //cout << "step " << typeid(U).name() << step << "\n";
    if( size < step ){
        return calc_inner_nosimd<U>(v1,v2,size);//arrays are too short to simd
    }
    auto remain = size % step;
    size -= remain;
    auto sum_real = m256_setzerp<T>();
    T sum_imag;
    if( iscomplex ){
        sum_imag = m256_setzerp<T>();//虚数部。double/floatの計算の際にはこの変数は使われない為、最適化で消える。
    }
    for(ssize_t i = 0; i < size; i+= step){
        //cout << " i " << i << "\n";
        auto d1 = m256_load<aligned>(v1);
        auto d2 = m256_load<aligned>(v2);
        sum_real = muladd(d1,d2,sum_real);
        if( iscomplex ){
            sum_imag = muladd_imag(d1,d2,sum_imag, v1);
        }
        v1 += step;
        v2 += step;
    }
    U result = calc_inner_nosimd<U>(v1,v2,remain);//あまりの計算を行う。
    result = getResult(result, sum_real, sum_imag);
    return result;
}

//NUM_THREADSの数のスレッドに計算を割り当てて計算する関数。
//future/asyncを使用
template <typename T, typename U, unsigned NUM_THREADS, bool ALIGNMENT, bool iscomplex>
static inline U calc_inner_multithread(const U* d1, const U* d2, ssize_t size1) {
    //cout << "calc_inner_multithread size " << size1 << "\n";
    auto numthreads = nthreads;
    if( size1 < numthreads * sizeof(T) * 16){//配列が短い場合にはスレッドを起こさない。この数は環境によって要調整
        return calc_inner<T,U,ALIGNMENT,iscomplex>(d1, d2, size1);
    }
    auto remain = size1 % numthreads;//スレッドに割り振れないあまり。
    auto size = size1 / numthreads;//スレッド１個あたりの計算数
    vector<future<U>> futures;
    if( size > 0 ){
        for(unsigned i = 0; i < numthreads-1; i++){//自分を除いた数のスレッドを実行
            ssize_t start = i * size;
            auto d1start = d1 + start;
            auto d2start = d2 + start;
            futures.push_back(async(launch::async,[d1start,d2start,size](){//別スレッドでcalc_innerを実行
                return calc_inner<T,U,ALIGNMENT,iscomplex>(d1start, d2start, size);
            }));
        }
    }
    ssize_t start = (numthreads-1) * size;
    U result = calc_inner<T,U,ALIGNMENT,iscomplex>(d1 + start, d2 + start, size + remain);//自スレッド分の計算
    for(auto&& f : futures){//他のスレッドの計算結果を集計
        result += f.get();
    }
    return result;
}
template <bool ALIGNMENT>
static PyObject* inner_aligned(const PyArrayObject* array1, const PyArrayObject* array2, 
    decltype(array1->dimensions[0]) size){
    //cout << (array1->descr->type) << "\n";
    switch(array1->descr->type ){
        case 'd':{
            const double* data1 = (const double*)array1->data;
            const double* data2 = (const double*)array2->data;
            auto result = calc_inner_multithread<__m256d,double,NUM_CORES,ALIGNMENT,false>(data1, data2, size);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("d", result);
        }
        case 'D':{
            const complex<double>* data1 = (const complex<double>*)array1->data;
            const complex<double>* data2 = (const complex<double>*)array2->data;
            auto result = calc_inner_multithread<__m256d,complex<double>,NUM_CORES,ALIGNMENT,true>(data1, data2, size);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("D", &result);
        }
        case 'f':{
            const float* data1 = (const float*)array1->data;
            const float* data2 = (const float*)array2->data;
            auto result = calc_inner_multithread<__m256,float,NUM_CORES,ALIGNMENT,false>(data1, data2, size);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("f", result);
        }
        case 'F':{
            const complex<float>* data1 = (const complex<float>*)array1->data;
            const complex<float>* data2 = (const complex<float>*)array2->data;
            complex<double> result = calc_inner_multithread<__m256,complex<float>,NUM_CORES,ALIGNMENT,true>(data1, data2, size);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("D", &result);//"F"ではエラーとなる。
        }
        default:{
            PyErr_SetString(PyExc_TypeError, "type miss match");
            return nullptr;
        }
    }

}

static PyObject* inner(PyObject *self, PyObject *args)
{
    PyArrayObject *array1,*array2;
    if (!PyArg_ParseTuple(args, "OO", &array1, &array2 )){
        PyErr_SetString(PyExc_ValueError, "not array");
        return NULL;
    }
    auto size1 = array1->dimensions[0];
    auto size2 = array2->dimensions[0];
    if( array1->nd != 1 || array2->nd != 1 || size1 != size2 ){
        PyErr_SetString(PyExc_IndexError, "shapes not aligned");
        return NULL;
    }
    if( array1->descr->type != array2->descr->type ){
        PyErr_SetString(PyExc_IndexError, "type miss match");
        return NULL;
    }
    if( ((reinterpret_cast<unsigned long long>(array1->data) | reinterpret_cast<unsigned long long>(array2->data)) 
        & (32-1)) == 0 ){//32 bytes aligned
        return inner_aligned<true>(array1, array2, size1);
    }else{//miss aligned
        return inner_aligned<false>(array1, array2, size1);
    }
}

template<typename T, typename U, bool ALIGNED, bool iscomplex>
static inline void dot21_line(U* data, const U* data1, ssize_t skip, const U* data2, ssize_t size){
    for( ssize_t i = 0; i < size; i ++){
        data[i] += calc_inner<T,U,ALIGNED,iscomplex>(data1, data2, skip);
        data1 += skip;
    }
}
template<typename T, typename U, bool ALIGNED, unsigned NTHREADS,bool iscomplex>
static PyObject* dot21_multithread(const PyArrayObject *array1, const PyArrayObject *array2, PyArrayObject* result){
    auto size = array1->dimensions[0];
    auto skip = array1->dimensions[1];

    U* data = (U*)result->data;
    auto data1 = (const U*)array1->data;
    auto data2 = (const U*)array2->data;
    auto remain = size % nthreads;
    auto chunksize = size / nthreads;
    //printf("chunksize %llx remain %llx nthreads %llx\n",chunksize, remain, nthreads);
    vector<future<void>> futures;
    if( chunksize > 0 ){
        for(unsigned i = 0; i < nthreads-1; i++){//自分を除いた数のスレッドを実行
            ssize_t start = i * chunksize;
            auto d1start = data1 + start * skip;
            auto dstart = data + start;
            //printf("%d start %lld \n",i, start);
            futures.push_back(async(launch::async,[dstart,d1start,data2,chunksize,skip](){//別スレッドで1要素を計算
                dot21_line<T,U,ALIGNED,iscomplex>(dstart, d1start, skip, data2, chunksize);
            }));
        }
    }
    ssize_t start = (nthreads-1) * chunksize;
    dot21_line<T,U,ALIGNED,iscomplex>(data + start, data1 + start* skip, skip, data2, chunksize + remain);//自スレッド分の計算
    for(auto&& f : futures){//他のスレッドの計算結終了を待つ
        f.wait();
    }
    return (PyObject*)result;
}
static PyObject* dot(PyObject *self, PyObject *args)
{
    PyArrayObject *array1,*array2;
    if (!PyArg_ParseTuple(args, "OO", &array1, &array2 )){
        PyErr_SetString(PyExc_ValueError, "not array");
        return NULL;
    }
    if( array1->descr->type != array2->descr->type ){
        PyErr_SetString(PyExc_IndexError, "type miss match");
        return NULL;
    }
    auto ndim1 = array1->nd;
    auto ndim2 = array2->nd;
    auto dim1 = array1->dimensions;
    //printArray<double>(array1, 1);
    //printArray<double>(array2, 2);
    //bool aligned = (((unsigned int)((unsigned long long)array1->data) | (unsigned int)((unsigned long long)array2->data)
    //        & (32-1)) == 0);
    
    if(ndim1 == 1 && ndim2 == 1){
        auto size = dim1[0];
        return inner_aligned<true>(array1, array2, size);
    }else if( ndim1 == 2 && (ndim2 == 1) ){
        if( array1->dimensions[1] != array2->dimensions[0] ){
            PyErr_SetString(PyExc_IndexError, "shapes not aligned");
            return NULL;
        }
        auto size = array1->dimensions[0];
        switch(array1->descr->type){
            case 'D':{
                auto result = PyArray_ZEROS(1, &size, NPY_COMPLEX128, 0);
                if( result == NULL ){
                    PyErr_SetString(PyExc_OverflowError, "");        
                    return NULL;
                }
                return dot21_multithread<__m256d,complex<double>,true,NUM_CORES,true>(array1,array2,(PyArrayObject*)result);
            }
            case 'd':{
                auto result = PyArray_ZEROS(1, &size, NPY_FLOAT64, 0);
                if( result == NULL ){
                    PyErr_SetString(PyExc_OverflowError, "");        
                    return NULL;
                }
                return dot21_multithread<__m256d,double,true,NUM_CORES,false>(array1,array2,(PyArrayObject*)result);
            }
            case 'F':{
                auto result = PyArray_ZEROS(1, &size, NPY_COMPLEX64, 0);
                if( result == NULL ){
                    PyErr_SetString(PyExc_OverflowError, "");        
                    return NULL;
                }
                return dot21_multithread<__m256,complex<float>,true,NUM_CORES,true>(array1,array2,(PyArrayObject*)result);
            }
            case 'f':{
                auto result = PyArray_ZEROS(1, &size, NPY_FLOAT32, 0);
                if( result == NULL ){
                    PyErr_SetString(PyExc_OverflowError, "");        
                    return NULL;
                }
                return dot21_multithread<__m256,float,true,NUM_CORES,false>(array1,array2,(PyArrayObject*)result);
            }
        }
    }
    return NULL;
}

//array->dataを表示
static PyObject* printAddress(PyObject *self, PyObject *args)
{
    PyArrayObject *array1;
    if (!PyArg_ParseTuple(args, "O", &array1 )){
        PyErr_SetString(PyExc_ValueError, "not array");
        return NULL;
    }
    printf("array->data:%llx\n",array1->data);
    return Py_None;
}

static PyMethodDef methods[] = {
    {"inner", inner, METH_VARARGS},
    {"dot", dot, METH_VARARGS},
    {"printAddress", printAddress, METH_VARARGS},
    {NULL, NULL}
};

PyDoc_STRVAR(api_doc, "Python3 API sample.\n");

static struct PyModuleDef cmodule = {
   PyModuleDef_HEAD_INIT,
   "mylib",   /* name of module */
   api_doc, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   methods
};


EXTERNC void* PyInit_mylib(void)
{
    if( PyArray_API==NULL ){
        import_array();
    }
    nthreads = thread::hardware_concurrency();
    //printf("nthreads %d\n", nthreads);
    return PyModule_Create(&cmodule);
}
