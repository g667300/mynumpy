import numpy
import mylib
import time

size = 16*256*256*256;
v1 = numpy.random.rand(size)+numpy.random.rand(size)*1j
v2 = numpy.random.rand(size)+numpy.random.rand(size)*1j
print(v1)
print(v2)


print(v1.dtype, v2.dtype)
print("numpy.inner")
print(numpy.inner(v1,v2))
print("mylib.inner")
print(mylib.inner(v1,v2))

"""
result = 0
for i in range(size):
    result += v1[i] * v2[i]

print(result)
"""

v1 = v1.astype('complex64')
v2 = v2.astype('complex64')
print(v1.dtype, v2.dtype)
print("numpy.inner")
result_numpy = numpy.inner(v1,v2)
print(result_numpy)
print("mylib.inner")
result_mylib = mylib.inner(v1,v2)
print(result_mylib)

diff = result_numpy - result_mylib
print("diff", diff)
