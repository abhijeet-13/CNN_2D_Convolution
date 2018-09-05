#include <iostream>
#include "conv_utils.h"
#include "tensor.h"

// input image shape
#define Ni 3
#define Lr 480
#define Lc 640

// input image up-sampling factors
#define Ur 2
#define Uc 2

// input image padding
#define Pl 10           // left
#define Pt 10           // top
#define Pr 10           // right
#define Pb 10           // bottom

// filter shape
#define Fr 17
#define Fc 17

// filter up sampling
#define Dr 2
#define Dc 2

// output image channels
#define No 3

// output down-sampling factors
#define Sr 2
#define Sc 2


int main() {
    
    filter_tensor<int> ffilter(3, 2, 3, 3);
    cu::initialize_tensor(ffilter, cu::init_type::SEQUENTIAL);
    ffilter.upsample_filter(2, 2);
    ffilter.upsample_filter(2, 2);
    ffilter.downsample_filter(4, 4);
    
    std::cout << "Filter below:\n";
    for(int o = 0; o < ffilter.get_ochannels(); o++){
        for(int c = 0; c < ffilter.get_ichannels(); c++){
            for(int i = 0; i < ffilter.get_irows(); i++){
                for(int j = 0; j < ffilter.get_icols(); j++){
                    std::cout << ffilter.at(o, c, i, j) << ", ";
                }
                std::cout << "|\n";
            }
            std::cout << "#\n";
        }
        std::cout << "============\n";
    }
    
    return 0;
    
    
    image_tensor<int> iimage(5, 5, 2);
    cu::initialize_tensor(iimage, cu::init_type::SEQUENTIAL);
    iimage.pad_image(1, 1, 1, 1);
    iimage.upsample_image(2, 2);
    iimage.downsample_image(2, 2);
    iimage.undo_operation();
    iimage.undo_operation();
    iimage.undo_operation();
    
    std::cout << "Image below:\n";
    for(int c = 0; c < iimage.get_channels(); c++){
        for(int i = 0; i < iimage.get_rows(); i++){
            for(int j = 0; j < iimage.get_cols(); j++){
                std::cout << iimage.at(c, i, j) << "-";
            }
            std::cout << ">>\n";
        }
        std::cout << "#\n";
    }
    
    return 0;
    
    
    

    
    matrix2D<int> ffilter_mat(ffilter);
    ffilter.mat_init();
    
    std::cout << "Filter matrix:\n";
    for(int i = 0; i < ffilter_mat.get_rows(); i++){
        for(int j = 0; j < ffilter_mat.get_cols(); j++){
            std::cout << ffilter_mat.at(i, j) << ", ";
        }
        std::cout << "\n";
    }
    
    matrix2D<int> iimage_mat(iimage);
    iimage.mat_init(ffilter.get_irows(), ffilter.get_icols(), ffilter.get_ichannels(), ffilter.get_ochannels());
    
    std::cout << "Topelitz matrix:\n";
    for(int i = 0; i < iimage_mat.get_rows(); i++){
        for(int j = 0; j < iimage_mat.get_cols(); j++){
            std::cout << iimage_mat.at(i, j) << ", ";
        }
        std::cout << "\n";
    }
    
    
    
    
    // View the tensor data
    /*
     
     tensor<int> input_image(3, 3, 10, 20);
     
     // Initialize tensor with sequential data
     cu::initialize_tensor(input_image, cu::init_type::SEQUENTIAL);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 20; j++) {
                std::cout << (int) input_image.get(c, i, j) << "-";
            }
            std::cout << std::endl;
        }
        std::cout << ">>\n>>\n";
    }
    */
    
    
    
    return 0;
}
