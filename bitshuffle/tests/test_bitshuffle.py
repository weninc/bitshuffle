import os
import h5py
import bitshuffle
import numpy as np

def test1():
    data = np.ones(1000, dtype=np.uint32)
    compressed = bitshuffle.compress_lz4(data)
    print(data.nbytes / compressed.nbytes)
    decompressed = bitshuffle.decompress_lz4(compressed, data.shape, data.dtype)
    assert(np.array_equal(data, decompressed))
    
def test_hdf5(tmp_path):
    data = np.ones((100, 100), dtype=np.uint32)
    with h5py.File(os.path.join(tmp_path, 'test.h5'), 'w') as fh:
        dset = fh.create_dataset('data', (5, 100, 100), 
                                 maxshape=(None, 100, 100),
                                 chunks=(1, 100, 100),
                                 compression=bitshuffle.BSHUF_H5FILTER,
                                 compression_opts=(0, bitshuffle.BSHUF_H5_COMPRESS_LZ4),
                                 dtype=np.uint32)
        
        compressed = bitshuffle.compress_lz4(data)
        for i in range(5):
            dset.id.write_direct_chunk((i, 0, 0), compressed.tobytes())
            
    with h5py.File(os.path.join(tmp_path, 'test.h5'), 'r') as fh:
        for img in fh['data']:
            assert(np.array_equal(data, img))
