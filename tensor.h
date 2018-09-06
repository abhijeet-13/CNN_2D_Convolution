#ifndef tensor_h
#define tensor_h

#include <vector>
#include <cstdarg>

// Generic tensor object of arbitrarily many dimensions
template<class T>
class tensor {
    std::vector<size_t> shape_sizes;        // shape sizes
    std::vector<size_t> cum_sizes;          // cumulative sizes
    size_t _size, _dims;                    // size of the underlying array and interpreted dimensions of the tensor
    
public:
    // available publically to allow direct modifications
    T* data;
    
    
    // specify number of dimensions, followed by length of each dimension
    tensor(size_t dims, ...) {
        
        // update dimensions
        _dims = dims;
        
        // obtain shape information
        _size = 1;
        va_list arglist;
        va_start(arglist, dims);
        for (int i = 0; i < dims; i++) {
            shape_sizes.emplace_back(va_arg(arglist, int));
            _size *= shape_sizes.back();
        }
        va_end(arglist);
        
        // obtain cumulative sizes to help dereference
        cum_sizes.emplace_back(1);
        for (size_t i = shape_sizes.size() - 1; i > 0; i--) {
            cum_sizes.emplace_back(cum_sizes.back() * shape_sizes[i]);
        }
        std::reverse(cum_sizes.begin(), cum_sizes.end());
        
        // allocate space for data
        data = new T[_size];
    }
    
    
    // delete underlying data on object deletion
    ~tensor() {
        delete data;
    }
    
    
    // obtain length of the underlying 1-D array
    size_t size() const { return _size; }
    
    
    // get the value at a particular index of the tensor
    T get (size_t idx0, ...) const {
        size_t idx = idx0 * cum_sizes[0];
        va_list argalist;
        va_start(argalist, idx0);
        for (int i = 1; i < _dims; i++) {
            size_t val = va_arg(argalist, int);
            idx += (val * cum_sizes[i]);
        }
        va_end(argalist);
        return data[idx];
    }
    
    
    // supports random initialization from existing data as well as sequential based on indices
    typedef enum init_type {
        RANDOM,
        SEQUENTIAL
    } init_type;
    
    
    // initialize the underlying tensor
    void initialize(init_type method, T value = 0, T* in_data = NULL) {
        size_t idx = 0;
        while (idx < _size) {
            data[idx++] = (method == SEQUENTIAL) ? value++ : in_data[idx];
        }
    }
    
};

#endif /* tensor_h */
