import struct
import numpy as np
from cffi import FFI

__all__ = ['decompress_lz4', 'compress_lz4', 'BSHUF_H5FILTER', 'BSHUF_H5_COMPRESS_LZ4']

BSHUF_H5FILTER = 32008
BSHUF_H5_COMPRESS_LZ4 = 2

ffi = FFI()
ffi.cdef('''
    int64_t bshuf_decompress_lz4(const void* in, void* out, const size_t size, const size_t elem_size, size_t block_size);
    int64_t bshuf_compress_lz4(const void* in, void* out, const size_t size, const size_t elem_size, size_t block_size);
    size_t bshuf_default_block_size(const size_t elem_size);
    size_t bshuf_compress_lz4_bound(const size_t size, const size_t elem_size, size_t block_size);
    void bshuf_write_uint64_BE(void* buf, uint64_t num);
    void bshuf_write_uint32_BE(void* buf, uint32_t num);
    ''')
lib = ffi.dlopen('libbshuf.so')


def decompress_lz4(data, shape, dtype):
    output = np.empty(shape, dtype=dtype)
    nbytes_uncomp, block_size = struct.unpack('>QI', data[:12])
    block_size = block_size // output.itemsize
    num_elements = nbytes_uncomp // output.itemsize
    lib.bshuf_decompress_lz4(ffi.from_buffer(data[12:]), ffi.from_buffer(output), num_elements, output.itemsize, block_size)
    return output 

def compress_lz4(data):
    element_size = data.dtype.itemsize
    block_size = lib.bshuf_default_block_size(element_size)
    num_elements = data.size
    max_out_size = lib.bshuf_compress_lz4_bound(num_elements, element_size, block_size) + 12
    output = np.empty(max_out_size, dtype=np.uint8)
    # HDF5 header http://www.hdfgroup.org/services/filters/HDF5_LZ4.pdf
    lib.bshuf_write_uint64_BE(ffi.from_buffer(output[0:]), data.nbytes)
    lib.bshuf_write_uint32_BE(ffi.from_buffer(output[8:]), block_size * element_size)
    count = lib.bshuf_compress_lz4(ffi.from_buffer(data), ffi.from_buffer(output[12:]), num_elements, element_size, block_size)
    nbytes = count + 12
    return output[:nbytes]
