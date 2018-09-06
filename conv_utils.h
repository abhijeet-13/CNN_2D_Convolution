#ifndef conv_utils_h
#define conv_utils_h

#include <vector>
#include "tensor.h"
#include <unordered_set>

namespace cu {
    
    // struct encompassing operation details
    struct operation{
        
        // define operation types suggesting three possible operations
        typedef enum op_type {
            PADZERO,
            UPSAMPLE,
            DOWNSAMPLE
        } op_type;
        
        // type of operation and its details
        op_type op;
        int left, top, right, bottom;
        int scaleX, scaleY;
        
        // pass the operation parameters as an array-style list
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
    
    
    
    // used in structures for maintaining operation lineage
    typedef std::pair<int, int> rows_cols;
    
    typedef std::pair<rows_cols, operation> resultsize_op_pair;
    
    
    
    
    // abstract class to enforce matrix specific indexing operation
    template<class T>
    class mat_interpretable_tensor : public tensor<T>{
    public:
        
        // constructor useful for image tensors
        mat_interpretable_tensor(int a, int b, int c, int d) : tensor<T>(a, b, c, d) {}
        
        // constructor useful for filter tensors
        mat_interpretable_tensor(int a, int b, int c, int d, int e) : tensor<T>(a, b, c, d, e) {}
        
        // must maintain row and column size of interpreted matrix
        int mat_rows, mat_cols;
        
        // must provide a value of interpreted matrix
        virtual T mat_value_at(int r, int c) const = 0;
        
    };
    
    
    
    
    // utility class to allow operations on an image tensor
    template <class T>
    class image_tensor: public mat_interpretable_tensor<T>{
        
        // dimension data for original underlying tensor
        int rows, cols, channels;
        
        // a stack of operations to maintain lineage
        std::vector<resultsize_op_pair> img_ops;
        
        // Give result from resulting array at offset - 1
        T at_util(int channel, int row, int col, size_t offset) const {
            // if no operation applied, index into original array
            if(offset == 0){
                return this->get(channel, row, col);
            }
            
            // image size after the current operation was applied
            int rrows = img_ops[offset - 1].first.first;
            int rcols = img_ops[offset - 1].first.second;
            
            // if the last operation was padding
            if(img_ops[offset - 1].second.op == operation::PADZERO){
                
                // obtain padding information
                int left = img_ops[offset - 1].second.left;
                int top = img_ops[offset - 1].second.top;
                int right = img_ops[offset - 1].second.right;
                int bottom = img_ops[offset - 1].second.bottom;
                
                // return default value if padded index, otherwise return appropriate value on the image before the operation
                if(row < top || col < left || row >= rrows - bottom || col >= rcols - right) return 0;
                return at_util(channel, row - top, col - left, offset - 1);
            }
            
            // if the last operation was upsample
            else if(img_ops[offset - 1].second.op == operation::UPSAMPLE){
                
                // obtain scale information
                int scaleX = img_ops[offset - 1].second.scaleX;
                int scaleY = img_ops[offset - 1].second.scaleY;
                
                // return values at only appropriate indices
                if(row % scaleY != 0 || col % scaleX != 0) return 0;
                return at_util(channel, row / scaleY, col / scaleX, offset - 1);
            }
            
            // if the last operation was downsample
            else if(img_ops[offset - 1].second.op == operation::DOWNSAMPLE){
                
                // obtain scale information
                int scaleX = img_ops[offset - 1].second.scaleX;
                int scaleY = img_ops[offset - 1].second.scaleY;
                
                // skip indices based on sampling factor
                return at_util(channel, row * scaleY, col * scaleX, offset - 1);
            }
            
            // default value: ideally does not reach here
            return 0;
        }
        
        
    public:
        
        // an image tensor is always going to be 3-dimensional
        image_tensor(int _rows, int _cols, int _channels = 1) :
        mat_interpretable_tensor<T>(3, _channels, _rows, _cols),
        rows(_rows),
        cols(_cols),
        channels(_channels) {}
        
        
        // for indexing to values in filter tensor's current state, regardless of any operations performed on it
        T at(int channel, int row, int col) const {
            return at_util(channel, row, col, img_ops.size());
        }
        
        // current number of rows
        int get_rows() const {
            return (img_ops.size() == 0) ? rows : (img_ops.back().first.first);
        }
        
        // current number of cols
        int get_cols() const {
            return (img_ops.size() == 0) ? cols : (img_ops.back().first.second);
        }
        
        // current number of channels
        int get_channels() const {
            return channels;
        }
        
        
        // pad the image on all sides
        void pad_image(int left, int top, int right, int bottom){
            int curr_rows = get_rows();
            int curr_cols = get_cols();
            img_ops.emplace_back(rows_cols({curr_rows + top + bottom, curr_cols + left + right}), operation(operation::PADZERO, {left, top, right, bottom}));
        }
        
        
        // upsample the image rows and columns
        void upsample_image(int scaleX, int scaleY){
            int curr_rows = get_rows();
            int curr_cols = get_cols();
            img_ops.emplace_back(rows_cols({curr_rows * scaleY, curr_cols * scaleX}), operation(operation::UPSAMPLE, {scaleX, scaleY}));
        }
        
        
        // downsample the image rows and columns
        void downsample_image(int scaleX, int scaleY){
            int curr_rows = get_rows();
            int curr_cols = get_cols();
            img_ops.emplace_back(rows_cols({curr_rows / scaleY, curr_cols / scaleX}), operation(operation::DOWNSAMPLE, {scaleX, scaleY}));
        }
        
        
        // undo any of the operations in its lineage
        void undo_operation(){
            if(img_ops.size() > 0) img_ops.pop_back();
        }
        
        
        // return Topelitz matrix interpreted value of input image (along with filter)
        T mat_value_at(int r, int c) const {
            // extract channel info from r
            int channel = r / (fr * fc);
            r = r % (fr * fc);
            
            // find the origin of the input block
            int origin_i = c / outc;
            int origin_j = c % outc;
            
            // find offset in the input block
            int offset_i = r / fc;
            int offset_j = r % fc;
            
            return at(channel, origin_i + offset_i, origin_j + offset_j);
        }
        
        int fr, fc, fi, fo, outr, outc;
        
        
        // displays the image tensor
        void display(const char* header = "Image tensor", std::string sep = "\t") const {
            std::cout << header << ":\n";
            for (int c = 0; c < get_channels(); c++) {
                std::cout << "\nChannel " << c + 1 << ":\n";
                for (int i = 0; i < get_rows(); i++) {
                    for (int j = 0; j < get_cols(); j++) {
                        std::cout << (int) this->at(c, i, j) << sep;
                    }
                    std::cout << "\n";
                }
            }
            std::cout << "-->\n";
        }
    };
    
    
    
    
    // Utility class to allow operations on a filter tensor
    template <class T>
    class filter_tensor: public mat_interpretable_tensor<T>{
        
        // dimensions of original underlying tensor
        int irows, icols, ichannels, ochannels;
        
        // to track lineage
        std::vector<resultsize_op_pair> img_ops;
        
        // Give result from resulting array at offset - 1
        T at_util(int ochannel, int ichannel, int row, int col, size_t offset) const {
            // if no operation applied, index into original array
            if(offset == 0){
                return this->get(ochannel, ichannel, row, col);
            }
            
            // if last operation was upsampling
            if(img_ops[offset - 1].second.op == operation::UPSAMPLE){
                int scaleX = img_ops[offset - 1].second.scaleX;
                int scaleY = img_ops[offset - 1].second.scaleY;
                
                if(row % scaleY != 0 || col % scaleX != 0) return 0;
                return at_util(ochannel, ichannel, row / scaleY, col / scaleX, offset - 1);
            }
            
            // if last operation was downsampling
            else if(img_ops[offset - 1].second.op == operation::DOWNSAMPLE){
                int scaleX = img_ops[offset - 1].second.scaleX;
                int scaleY = img_ops[offset - 1].second.scaleY;
                
                return at_util(ochannel, ichannel, row * scaleY, col * scaleX, offset - 1);
            }
            
            // default value
            return 0;
        }
        
        
    public:
        
        // a filter tensor is always going to be 4-dimensional
        filter_tensor(int _ochannels, int _ichannels, int _irows, int _icols) :
        mat_interpretable_tensor<T>(4, _ochannels, _ichannels, _irows, _icols),
        ochannels(_ochannels),
        ichannels(_ichannels),
        irows(_irows),
        icols(_icols) {}
        
        
        // for indexing to values in filter tensor's current state, regardless of any operations performed on it
        T at(int ochannel, int ichannel, int row, int col) const {
            return at_util(ochannel, ichannel, row, col, img_ops.size());
        }
        
        // current number of rows
        int get_irows() const {
            return (img_ops.size() == 0) ? irows : (img_ops.back().first.first);
        }
        
        // current number of columns
        int get_icols() const {
            return (img_ops.size() == 0) ? icols : (img_ops.back().first.second);
        }
        
        // current number of channels from input image
        int get_ichannels() const {
            return ichannels;
        }
        
        // current number of output channels
        int get_ochannels() const {
            return ochannels;
        }
        
        
        // upsample the image rows and columns
        void upsample_filter(int scaleX, int scaleY){
            int curr_rows = get_irows();
            int curr_cols = get_icols();
            img_ops.emplace_back(rows_cols({curr_rows * scaleY, curr_cols * scaleX}), operation(operation::UPSAMPLE, {scaleX, scaleY}));
        }
        
        
        // downsample the image rows and columns
        void downsample_filter(int scaleX, int scaleY){
            int curr_rows = get_irows();
            int curr_cols = get_icols();
            img_ops.emplace_back(rows_cols({curr_rows / scaleY, curr_cols / scaleX}), operation(operation::DOWNSAMPLE, {scaleX, scaleY}));
        }
        
        
        // undo any of the operations in its lineage
        void undo_operation() {
            if (img_ops.size() > 0) img_ops.pop_back();
        }
        
        
        // return matrix interpreted value of filter
        T mat_value_at(int r, int c) const {
            int ir = get_irows(), ic = get_icols();
            int rc = ir * ic;
            return at(r, c / rc, (c % rc) / ic, (c % rc) % ic);
        }
        
        
        // display filter
        void display(const char* header = "Filter tensor") const {
            std::cout << header << ":\n";
            for (int o = 0; o < get_ochannels(); o++) {
                std::cout << "----------\n";
                for (int i = 0; i < get_ichannels(); i++) {
                    std::cout << "Channel " << i + 1 << ":\n";
                    for (int r = 0; r < get_irows(); r++) {
                        for (int c = 0; c < get_icols(); c++) {
                            std::cout << this->at(o, i, r, c) << "\t";
                        }
                        std::cout << "\n";
                    }
                }
                std::cout << "----------\n";
            }
            std::cout << "-->\n";
        }
    };
    
    
    
    
    // Interpreted matrix from tensors
    template<class T>
    class matrix2D{
        
        // underlying tensor from which matrix is interpreted
        mat_interpretable_tensor<T>* _from_tensor;
        
    public:
        
        // reference to the position within matrix data
        T& at(int r, int c) {
            return _from_tensor->data[_from_tensor->mat_cols *  r + c];
        }
        
        // return matrix interpreted value
        T mat_at(int r, int c) const {
            return _from_tensor->mat_value_at(r, c);
        }
        
        // number of rows of matrix
        int get_rows() const {
            return _from_tensor->mat_rows;
        }
        
        // number of columns of matrix
        int get_cols() const {
            return _from_tensor->mat_cols;
        }
        
        
        // for initializing filter matrix
        matrix2D(filter_tensor<T> &conv_filter) : _from_tensor(&conv_filter) {
            _from_tensor->mat_rows = conv_filter.get_ochannels();
            _from_tensor->mat_cols = conv_filter.get_ichannels() * conv_filter.get_irows() * conv_filter.get_icols();
        }
        
        // for initializing Toeplitz matrix from input image and input filter
        matrix2D(image_tensor<T> &conv_input_image, const filter_tensor<T> &conv_filter) : _from_tensor(&conv_input_image) {
            conv_input_image.fr = conv_filter.get_irows();
            conv_input_image.fc = conv_filter.get_icols();
            conv_input_image.fi = conv_filter.get_ichannels();
            conv_input_image.fo = conv_filter.get_ochannels();
            conv_input_image.outr = conv_input_image.get_rows() - conv_input_image.fr + 1;
            conv_input_image.outc = conv_input_image.get_cols() - conv_input_image.fc + 1;
            
            _from_tensor->mat_rows = conv_input_image.fr * conv_input_image.fc * conv_input_image.get_channels();
            _from_tensor->mat_cols = conv_input_image.outr * conv_input_image.outc;
        }
        
        // for initializing output image matrix
        matrix2D(image_tensor<T> &conv_output_image) : _from_tensor(&conv_output_image) {
            _from_tensor->mat_rows = conv_output_image.get_channels();
            _from_tensor->mat_cols = conv_output_image.get_rows() * conv_output_image.get_cols();
        }
        
        
        // display mat interpreted matrix
        void display(const char * header = "Interpreted matrix") const {
            std::cout << header << ":\n";
            for (int i = 0; i < get_rows(); i++) {
                for (int j = 0; j < get_cols(); j++) {
                    std::cout << this->mat_at(i, j) << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "-->\n";
        }
        
    };
    
    
    
    // utility to multiply two interpreted matrices
    template<class T>
    void mult_matrix2D(const matrix2D<T> &m1, const matrix2D<T> &m2, matrix2D<T> out) {
        if (out.get_rows() != m1.get_rows() || out.get_cols() != m2.get_cols()) {
            std::cout << "[Matrix multiplication error] output dimensions do not match.\n";
            return;
        }
        
        if (m1.get_cols() != m2.get_rows()) {
            std::cout << "[Matrix multiplication error] input matrices dimensions are incompatible.\n";
            return;
        }
        
        for (int i = 0; i < m1.get_rows(); i++) {
            for (int j = 0; j < m2.get_cols(); j++) {
                out.at(i, j) = 0;
                for (int k = 0; k < m1.get_cols(); k++) {
                    out.at(i, j) += m1.mat_at(i, k) * m2.mat_at(k, j);
                }
            }
        }
    }
    
    
};

#endif /* conv_utils_h */


