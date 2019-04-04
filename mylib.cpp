#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <immintrin.h>
#include <type_traits>
#include <thread>
#include <future>
#include <vector>
#include <iostream>
#include <complex>
#include <typeinfo>

using namespace std;

#define EXTERNC extern "C"
//T,Uの組み合わせの確認マクロ
#define type_check() \
 static_assert(((is_same<double,U>::value) && (is_same<__m256d,T>::value)) || \
    ((is_same<float,U>::value) && (is_same<__m256,T>::value)) ||\
    ((is_same<complex<float>,U>::value) && (is_same<__m256,T>::value)) ||\
    ((is_same<complex<double>,U>::value) && (is_same<__m256d,T>::value))\
    ,"type mismatch")


static inline auto m256_load(const double* d1){
        return _mm256_load_pd(d1);
}
static inline auto m256_load(const complex<double>* d1){
        return _mm256_load_pd((const double*)d1);
}
static inline auto m256_load(const float* d1){
        return _mm256_load_ps(d1);
}
static inline auto m256_load(const complex<float>* d1){
        return _mm256_load_ps((const float*)d1);
}
/*
static inline void m256_store(double* d1, __m256d reg){
    _mm256_store_pd(d1, reg);
}
static inline void m256_store(complex<double>* d1, __m256d reg){
    _mm256_store_pd((double*)d1, reg);
}

static inline void m256_store(float* d1, __m256 reg){
    _mm256_store_ps(d1, reg);
}
static inline void m256_store(complex<float>* d1, __m256 reg){
    _mm256_store_ps((float*)d1, reg);
}
*/
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
static inline __m256d m256_shuffle(__m256d d1, __m256d d2){
    return _mm256_shuffle_pd(d1,d2,0);
}
static inline __m256 m256_shuffle(__m256 d1, __m256 d2){
    return _mm256_shuffle_ps(d1,d2,0);
}

static inline __m256d m256_blend(__m256d d1, __m256d d2){
    return _mm256_blend_pd(d1,d2,0xa);
}
static inline __m256 m256_blend(__m256 d1, __m256 d2){
    return _mm256_blend_ps(d1,d2,0xaa);
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
static U calc_inner_nosimd(const U* v1, const U* v2, ssize_t size){
    U result = 0;
    //cout << "calc_inner_nosimd " << size <<"\n";
    for(ssize_t i = 0; i < size; i++){
        auto r = v1[i] * v2[i];
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
static __m256d muladd(const __m256d d1, const __m256d d2, __m256d sum){
    return m256_fmadd(d1,d2,sum);//sum=d1*d2+sum
}
static __m256 muladd(const __m256 d1, const __m256 d2, __m256 sum){
    return m256_fmadd(d1,d2,sum);//sum=d1*d2+sum
}

//虚数部の積和計算
//配列の順序を入れ替えて積和計算を行う。
static __m256d muladd_imag(const __m256d d1, const __m256d d2, __m256d sum){
    auto d = _mm256_permute_pd(d1, 0x5);
    return m256_fmadd(d,d2,sum);
}
static __m256 muladd_imag(const __m256 d1, const __m256 d2, __m256 sum){
    auto d = _mm256_permute_ps(d1, 0xb1);
    return m256_fmadd(d,d2,sum);
}
static auto muladd_imag(const double* d1, const __m256d d2, __m256d sum){
    return _mm256_setzero_pd();
}
static auto muladd_imag(const float* d1, const __m256d d2, __m256d sum){
    return _mm256_setzero_ps();
}
/*
static auto addresult(const complex<double> &result, const complex<double> &a, const complex<double> &b){
    return complex<double>(result.real() + a.real() - a.imag(), result.imag() + b.real() + b.imag());
}

static auto addresult(const complex<float> &result, const complex<float> &a, const complex<float> &b){
    return complex<float>(result.real() + a.real() - a.imag(), result.imag() + b.real() + b.imag());
}

static auto addresult(const double result, const double a, const double b){
    return result + a;
}
static auto addresult(const float result, const float a, const float b){
    return result + a;
}
*/

static inline auto getResult(const double sum, const __m256d sum_real, const __m256d sum_imag){
    const double* results = (const double*)&sum_real;
    auto result = sum + results[0] + results[1] + results[2] + results[3];
    return result;
}

static inline auto getResult(const complex<double> sum, const __m256d sum_real, const __m256d sum_imag){
    auto result = sum;
    auto real = m256_hsub(sum_real, sum_real);
    auto imag = m256_hadd(sum_imag, sum_imag);
    auto result_complex = m256_blend(real, imag);
    const complex<double>* results = (const complex<double>*)&result_complex;
    return sum + results[0] + results[1];
}

static inline auto getResult(const float sum, const __m256 sum_real, const __m256 sum_imag){
    const float* results = (const float*)&sum_real;
    auto result = sum + results[0] + results[1] + results[2] + results[3] 
        + results[4] + results[5] + results[6] + results[7];
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


//simd命令を使って、内積を求める。
template <typename T,typename U>
static U calc_inner(const U* v1, const U* v2, ssize_t size){
    type_check();
    //cout << "calc_inner size " << size << "\n";
    constexpr int step = (is_same<float,U>::value) ? 8 : 
        (is_same<complex<float>,U>::value) ? 4 : 
        (is_same<double,U>::value) ? 4 : 
        (is_same<complex<double>,U>::value) ? 2 : -1;
    static_assert(step > 0,"illeagal type");
    //cout << "step " << typeid(U).name() << step << "\n";
    //simdレジスタ１個に入るデータの数
    if( size < step ){
        return calc_inner_nosimd<U>(v1,v2,size);//配列が短すぎる場合には普通に計算
    }
    auto remain = size % step;//simdに入りきらないあまり分。後で別に計算する。
    size -= remain;
    auto sum_real = m256_setzerp<T>();
    T sum_imag;
    if(is_same<complex<double>,U>::value || is_same<complex<float>,U>::value){
        sum_imag = sum_real;
    }
    for(ssize_t i = 0; i < size; i+= step){
        //cout << " i " << i << "\n";
        auto d1 = m256_load(v1);
        auto d2 = m256_load(v2);
        sum_real = muladd(d1,d2,sum_real);
        if(is_same<complex<double>,U>::value || is_same<complex<float>,U>::value){
            sum_imag = muladd_imag(d1,d2,sum_imag);
        }
        v1 += step;
        v2 += step;
    }
    U result = calc_inner_nosimd<U>(v1,v2,remain);//４個または８個のあまりの計算を行う。

#if 1
    result = getResult(result, sum_real, sum_imag);
    return result;
#elif 1
    if(is_same<complex<double>,U>::value || is_same<complex<float>,U>::value){
        sum_real = m256_hsub(sum_real, sum_real);
        sum_imag = m256_hadd(sum_imag, sum_imag);
        auto result_complex = m256_blend(sum_real, sum_imag);
        const U* results = (const U*)&result_complex;
        for(int i=0; i < step; i++){//計算結果の集計
            result += results[i];
        }
    }else{
        const U* results = (const U*)&sum_real;
        for(int i=0; i < step; i++){//計算結果の集計
            result += results[i];
        }
    }
    return result;
#else
    U results[step];
    U results_img[step];
    m256_store(results, sum_real);//計算結果を配列に書き出す。
    if(is_same<complex<double>,U>::value || is_same<complex<float>,U>::value){
        m256_store(results_img, sum_imag);//計算結果を配列に書き出す。
        for(int i=0; i < step; i++){//計算結果の集計
            result = addresult(result, results[i], results_img[i]);
        }
    }else{
        for(int i=0; i < step; i++){//計算結果の集計
            result = addresult(result, results[i],0);
        }
    }
    return result;
#endif
}

//NUM_THREADSの数のスレッドに計算を割り当てて計算する関数。
//future/asyncを使用
template <typename T, typename U, unsigned NUM_THREADS>
static U calc_inner_multithread(const U* d1, const U* d2, ssize_t size1) {
    //cout << "calc_inner_multithread size " << size1 << "\n";
    if( size1 < 32 ){//配列が短い場合にはスレッドを起こさない。この数は環境によって要調整
        return calc_inner<T,U>(d1, d2, size1);
    }
    auto remain = size1 % NUM_THREADS;//スレッドに割り振れないあまり。
    auto size = size1 / NUM_THREADS;//スレッド１個あたりの計算数
    vector<future<U>> futures;
    if( size > 0 ){
        for(unsigned i = 0; i < NUM_THREADS-1; i++){//自分を除いた数のスレッドを実行
            ssize_t start = i * size;
            auto d1start = d1 + start;
            auto d2start = d2 + start;
            futures.push_back(async(launch::async,[d1start,d2start,size](){//別スレッドでcalc_innerを実行
                return calc_inner<T,U>(d1start, d2start, size);
            }));
        }
    }
    ssize_t start = (NUM_THREADS-1) * size;
    U result = calc_inner<T,U>(d1 + start, d2 + start, size + remain);//自スレッド分の計算
    for(auto&& f : futures){//他のスレッドの計算結果を集計
        result += f.get();
    }
    return result;
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
    
    //cout << (array1->descr->type) << "\n";
    switch(array1->descr->type ){
        case 'd':{
            const double* data1 = (const double*)array1->data;
            const double* data2 = (const double*)array2->data;
            auto result = calc_inner_multithread<__m256d,double,4>(data1, data2, size1);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("d", result);
        }
        case 'D':{
            const complex<double>* data1 = (const complex<double>*)array1->data;
            const complex<double>* data2 = (const complex<double>*)array2->data;
            auto result = calc_inner_multithread<__m256d,complex<double>,4>(data1, data2, size1);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("D", &result);
        }
        case 'f':{
            const float* data1 = (const float*)array1->data;
            const float* data2 = (const float*)array2->data;
            auto result = calc_inner_multithread<__m256,float,4>(data1, data2, size1);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("f", result);
        }
        case 'F':{
            const complex<float>* data1 = (const complex<float>*)array1->data;
            const complex<float>* data2 = (const complex<float>*)array2->data;
            complex<double> result = calc_inner_multithread<__m256,complex<float>,4>(data1, data2, size1);
            //cout << "result=" << result << "\n";
            return Py_BuildValue("D", &result);
        }
        default:{
            PyErr_SetString(PyExc_TypeError, "type miss match");
            return NULL;
        }
    }
}

static PyMethodDef methods[] = {
    {"inner", inner, METH_VARARGS},
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
    return PyModule_Create(&cmodule);
}
