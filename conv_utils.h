#include <vector>
#include <stdarg.h>
#include "tensor.h"


// operation type suggesting three possible operations
enum op_type{
    PADZERO,
    UPSAMPLE,
    DOWNSAMPLE
};


// class encompassing operation details
class operation{
public:
    op_type op;
    int left, top, right, bottom;
    int scaleX, scaleY;
    
    operation(op_type _op, std::vector<int> vals){
        op = _op;
        if(op == PADZERO){
            left = vals[0];
            top = vals[1];
            right = vals[2];
            bottom = vals[3];
        } else{
            scaleX = vals[0];
            scaleY = vals[1];
        }
    }
};



// abstract class to enforce matrix specific indexing operation
template<class T>
class mat_interpreted_tensor : public tensor<T>{
public:
    mat_interpreted_tensor(int a, int b, int c, int d) : tensor<T>(a, b, c, d) {}
    mat_interpreted_tensor(int a, int b, int c, int d, int e) : tensor<T>(a, b, c, d, e) {}
    
    // define abstract method, which matrix interpreted tensors would need to implement
    virtual T mat_value_at(int r, int c) = 0;
    int mat_rows, mat_cols;
};



// for maintaining operation lineage
typedef std::pair<int, int> rows_cols;
typedef std::pair<rows_cols, operation> resultsize_op_pair;








// utility class to allow operations on an image tensor
template <class T>
class image_tensor: public mat_interpreted_tensor<T>{

    int rows, cols, channels;
    std::vector<resultsize_op_pair> img_ops;
    
    // Give result from resulting array at offset - 1
    T at_util(int channel, int row, int col, size_t offset){
        // if no operation applied, index into original array
        if(offset == 0){
            return this->get(channel, row, col);
        }
        
        int rrows = img_ops[offset - 1].first.first;
        int rcols = img_ops[offset - 1].first.second;
        
        if(img_ops[offset - 1].second.op == PADZERO){
            
            int left = img_ops[offset - 1].second.left;
            int top = img_ops[offset - 1].second.top;
            int right = img_ops[offset - 1].second.right;
            int bottom = img_ops[offset - 1].second.bottom;
            
            if(row < top || col < left || row >= rrows - bottom || col >= rcols - right) return 0;
            return at_util(channel, row - top, col - left, offset - 1);
        }
        else if(img_ops[offset - 1].second.op == UPSAMPLE){
            int scaleX = img_ops[offset - 1].second.scaleX;
            int scaleY = img_ops[offset - 1].second.scaleY;
            
            if(row % scaleY != 0 || col % scaleX != 0) return 0;
            return at_util(channel, row / scaleY, col / scaleX, offset - 1);
        }
        else if(img_ops[offset - 1].second.op == DOWNSAMPLE){
            int scaleX = img_ops[offset - 1].second.scaleX;
            int scaleY = img_ops[offset - 1].second.scaleY;
            
            return at_util(channel, row * scaleY, col * scaleX, offset - 1);
        }
        
        return 0;
    }
    
public:
    
    // an image tensor is always going to be 3-dimensional
    image_tensor(int _rows, int _cols, int _channels = 1) :
        mat_interpreted_tensor<T>(3, _channels, _rows, _cols),
        rows(_rows),
        cols(_cols),
        channels(_channels) {}
    
    
    // for indexing to values in filter tensor's current state, regardless of any operations performed on it
    T at(int channel, int row, int col){
        return at_util(channel, row, col, img_ops.size());
    }
    
    
    // Getters
    int get_rows(){
        return (img_ops.size() == 0) ? rows : (img_ops.back().first.first);
    }
    
    int get_cols(){
        return (img_ops.size() == 0) ? cols : (img_ops.back().first.second);
    }
    
    int get_channels(){
        return channels;
    }
    
    
    // pad the image on all sides
    void pad_image(int left, int top, int right, int bottom){
        
        int curr_rows = get_rows();
        int curr_cols = get_cols();
        img_ops.emplace_back(rows_cols({curr_rows + top + bottom, curr_cols + left + right}), operation(PADZERO, {left, top, right, bottom}));
    }
    
    
    // upsample the image rows and columns
    void upsample_image(int scaleX, int scaleY){
        int curr_rows = get_rows();
        int curr_cols = get_cols();
        img_ops.emplace_back(rows_cols({curr_rows * scaleY, curr_cols * scaleX}), operation(UPSAMPLE, {scaleX, scaleY}));
    }
    
    
    // downsample the image rows and columns
    void downsample_image(int scaleX, int scaleY){
        int curr_rows = get_rows();
        int curr_cols = get_cols();
        img_ops.emplace_back(rows_cols({curr_rows / scaleY, curr_cols / scaleX}), operation(DOWNSAMPLE, {scaleX, scaleY}));
    }
    
    
    // undo any of the operations in its lineage
    void undo_operation(){
        if(img_ops.size() > 0) img_ops.pop_back();
    }
    
    
    // for obtaining values in the derived 2D matrix for convolution
    
    int fr, fc, fi, fo, outr, outc;
    void mat_init(int _fr, int _fc, int _fi, int _fo){
        fr = _fr;
        fc = _fc;
        fi = _fi;
        fo = _fo;
        
        outr = get_rows() - fr + 1;
        outc = get_cols() - fc + 1;
        
        std::cout << "outc is " << outc << " and " << outr << "\n";
        
        this->mat_rows = fr * fc * channels;
        this->mat_cols = outr * outc;
    }
    
    T mat_value_at(int r, int c){
        // extract channel info from r
        int channel = r / (fr * fc);
        r = r % (fr * fc);

        // find the origin of the input block
        
        
        int origin_i = c / outc;
        int origin_j = c % outc;
        // std::cout << c << " / " << outc << " = " << origin_i << " ";
        
        // find offset in the input block
        int offset_i = r / fc;
        int offset_j = r % fc;
        
        // std::cout << "(" << channel << ", " << origin_i + offset_i << ", " << origin_j + offset_j << ")";
        return at(channel, origin_i + offset_i, origin_j + offset_j);
    }
    
};









// Utility class to allow operations on a filter tensor
template <class T>
class filter_tensor: public mat_interpreted_tensor<T>{
    
    int irows, icols, ichannels, ochannels;
    std::vector<resultsize_op_pair> img_ops;
    
    // Give result from resulting array at offset - 1
    T at_util(int ochannel, int ichannel, int row, int col, size_t offset){
        // if no operation applied, index into original array
        if(offset == 0){
            return this->get(ochannel, ichannel, row, col);
        }
        
        //int rrows = img_ops[offset - 1].first.first;
        //int rcols = img_ops[offset - 1].first.second;
        if(img_ops[offset - 1].second.op == UPSAMPLE){
            int scaleX = img_ops[offset - 1].second.scaleX;
            int scaleY = img_ops[offset - 1].second.scaleY;
            
            if(row % scaleY != 0 || col % scaleX != 0) return 0;
            return at_util(ochannel, ichannel, row / scaleY, col / scaleX, offset - 1);
        }
        else if(img_ops[offset - 1].second.op == DOWNSAMPLE){
            int scaleX = img_ops[offset - 1].second.scaleX;
            int scaleY = img_ops[offset - 1].second.scaleY;

            return at_util(ochannel, ichannel, row * scaleY, col * scaleX, offset - 1);
        }
        
        return 0;
    }
    
public:
    
    // a filter tensor is always going to be 4-dimensional
    filter_tensor(int _ochannels, int _ichannels, int _irows, int _icols) :
        mat_interpreted_tensor<T>(4, _ochannels, _ichannels, _irows, _icols),
        ochannels(_ochannels),
        ichannels(_ichannels),
        irows(_irows),
        icols(_icols) {}
    
    
    // for indexing to values in filter tensor's current state, regardless of any operations performed on it
    T at(int ochannel, int ichannel, int row, int col){
        return at_util(ochannel, ichannel, row, col, img_ops.size());
    }
    
    
    // upsample the image rows and columns
    void upsample_filter(int scaleX, int scaleY){
        int curr_rows = get_irows();
        int curr_cols = get_icols();
        img_ops.emplace_back(rows_cols({curr_rows * scaleY, curr_cols * scaleX}), operation(UPSAMPLE, {scaleX, scaleY}));
    }
    
    
    // downsample the image rows and columns
    void downsample_filter(int scaleX, int scaleY){
        int curr_rows = get_irows();
        int curr_cols = get_icols();
        img_ops.emplace_back(rows_cols({curr_rows / scaleY, curr_cols / scaleX}), operation(DOWNSAMPLE, {scaleX, scaleY}));
    }
    
    
    // Getters
    int get_irows(){
        return (img_ops.size() == 0) ? irows : (img_ops.back().first.first);
    }
    
    int get_icols(){
        return (img_ops.size() == 0) ? icols : (img_ops.back().first.second);
    }
    
    int get_ichannels(){
        return ichannels;
    }
    
    int get_ochannels(){
        return ochannels;
    }
    
    
    
    // for obtaining values in the derived 2D matrix for convolution
    void mat_init(){
        this->mat_rows = ochannels;
        this->mat_cols = ichannels * irows * icols;
    }
    
    
    T mat_value_at(int r, int c){
        // TODO: Assert to init first
        int rc = irows * icols;
        return at(r, c / rc, (c % rc) / icols, (c % rc) % icols);
    }
};








// A matrix class which reshapes input tensor to 2 dimensions
template<class T>
class matrix2D{
    mat_interpreted_tensor<T>* _from_tensor;
    
    
public:
    T at(int r, int c) {
        return _from_tensor->mat_value_at(r, c);
    }
    
    int get_rows(){
        return _from_tensor->mat_rows;
    }
    
    int get_cols(){
        return _from_tensor->mat_cols;
    }
    
    
    // TODO: cols should be uniquely determined from rows and the underlying tensor
    matrix2D(mat_interpreted_tensor<T> &from_tensor) : _from_tensor(&from_tensor) {}
};


namespace cu{
    // type of data
    enum init_type{
        RANDOM,
        SEQUENTIAL
    };
    
    template<class T>
    void initialize_tensor(tensor<T> &t, init_type method){
        if(method == SEQUENTIAL){
            T value = 0;
            size_t idx = 0;
            while(idx < t.size()){
                t.data[idx++] = value++;
            }
        }
    }
    
};




