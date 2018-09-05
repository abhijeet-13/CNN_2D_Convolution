#ifndef tensor_h
#define tensor_h

template<class T>
class tensor {
    std::vector<size_t> shape_sizes;        // shape sizes
    std::vector<size_t> cum_sizes;          // cumulative sizes
    size_t _size;
    int _dims;
    
    
public:
    
    T * data;
    
    // Constructor
    tensor(int dims, ...) {
        
        _dims = dims;
        _size = 1;
        
        // obtain shape information
        va_list arglist;
        va_start(arglist, dims);
        for (int i = 0; i < dims; i++) {
            shape_sizes.emplace_back(va_arg(arglist, size_t));
            _size *= shape_sizes.back();
        }
        va_end(arglist);
        
        cum_sizes.emplace_back(1);
        for (size_t i = shape_sizes.size() - 1; i > 0; i--) {
            cum_sizes.emplace_back(cum_sizes.back() * shape_sizes[i]);
        }
        std::reverse(cum_sizes.begin(), cum_sizes.end());
        
        // allocate space for data
        data = new T[_size];
    }
    
    // Destructor
    ~tensor() {
        delete data;
    }
    
    // Getters
    size_t size() { return _size; }
    
    T get(int idx0, ...) {
        size_t idx = idx0 * cum_sizes[0];
        va_list argalist;
        va_start(argalist, idx0);
        for (int i = 1; i < _dims; i++) {
            size_t val = va_arg(argalist, size_t);
            idx += (val * cum_sizes[i]);
        }
        va_end(argalist);
        return data[idx];
    }
    
    void init_seq() {
        for (size_t val = 1; val <= _size; val++) {
            data[val - 1] = (T) val;
        }
    }
    
};

#endif /* tensor_h */
