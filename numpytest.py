import numpy
import mylib
import time
import timeit



#１回の計算にかかる時間を測る。
def laptime(f,vec1,vec2):
    t = time.time()
    result = f(vec1, vec2)
    t = time.time() - t
    return t

def test_inner(v1, v2):
    time_numpy_inner = 0
    time_mylib_inner = 0

    #warming up
    result_numpy_inner = numpy.inner(v1,v2)
    result_mylib_inner = mylib.inner(v1,v2)
    #できるだけ公平になるように、ループ内で交互に呼ぶ
    for i in range(16):
        time_mylib_inner += laptime(mylib.inner,v1,v2)
        time_numpy_inner += laptime(numpy.inner,v1,v2)
    for i in range(16):
        time_numpy_inner += laptime(numpy.inner,v1,v2)
        time_mylib_inner += laptime(mylib.inner,v1,v2)
    #結果の表示
    print("numpy.inner %5f(sec)" % time_numpy_inner,      "result=",result_numpy_inner)
    print("mylib.inner %5f(sec)" % time_mylib_inner,      "result=",result_mylib_inner)
    print("time ratio %5f" % (time_numpy_inner/time_mylib_inner))

def test(arraysize):
    print("------------------------------")
    print("array size = ", arraysize)
    print("double array inner")
    vector1 = numpy.random.rand(arraysize)
    vector2 = numpy.random.rand(arraysize)
    test_inner(vector1,vector2)

    print("float array inner",)
    vector1 = vector1.astype('float32')
    vector2 = vector2.astype('float32')
    test_inner(vector1,vector2)

    print("complex double array inner")
    vector1 = numpy.random.rand(arraysize) + numpy.random.rand(arraysize) * 1j
    vector2 = numpy.random.rand(arraysize) + numpy.random.rand(arraysize) * 1j
    test_inner(vector1,vector2)

    print("complex float array inner")
    vector1 = vector1.astype('complex64')
    vector2 = vector2.astype('complex64')
    test_inner(vector1,vector2)

#配列の長さ。１次元の配列です。
test_count = 256 * 256 *256
for i in range(1,9):
    size = i * test_count
    test(size)

"""
#１回の計算にかかる時間を測る。
def laptime1(f,vec1):
    t = time.time()
    result = f(vec1)
    t = time.time() - t
    return t

def print_time_sum(v1):
    time_numpy = 0
    time_mylib = 0

    #warming up
    result_numpy = numpy.sum(v1)
    result_mylib = mylib.sum(v1)
    #できるだけ公平になるように、ループ内で交互に呼ぶ
    for i in range(16):
        time_mylib += laptime1(mylib.sum,v1)
        time_numpy += laptime1(numpy.sum,v1)
    #結果の表示
    print("numpy.sum %5f(sec)" % time_numpy,      "result=",result_numpy)
    print("mylib.sum %5f(sec)" % time_mylib,      "result=",result_mylib)
    print("time ratio %5f" % (time_numpy/time_mylib))

print("double array sum size=",maxsize)
vector1 = numpy.random.rand(maxsize)
print_time_sum(vector1)

print("float array sum size=",maxsize)
vector1 = vector1.astype('float32')
print_time_sum(vector1)
"""
