#include <iostream>
#include "conv_utils.h"
#include "tensor.h"
#include <unordered_map>
#include <fstream>

// input image shape
#define Ni 2
#define Lr 5
#define Lc 5

// input image up-sampling factors
#define Ur 1
#define Uc 1

// input image padding
#define Pl 0           // left
#define Pt 0           // top
#define Pr 0           // right
#define Pb 0           // bottom

// filter shape
#define Fr 3
#define Fc 3

// filter up sampling
#define Dr 1
#define Dc 1

// output image channels
#define No 3

// output down-sampling factors
#define Sr 1
#define Sc 1


int main() {
    
    // generate input feature map with sequential data
    cu::image_tensor<int> iimage(Lr, Lc, Ni);
    iimage.initialize(tensor<int>::init_type::SEQUENTIAL, 0);
    
    // generate filter coefficients with sequential data
    cu::filter_tensor<int> ffilter(No, Ni, Fr, Fc);
    ffilter.initialize(tensor<int>::init_type::SEQUENTIAL, 0);
    
    
    
    // upsample the input feature map and filter coefficients
    iimage.upsample_image(Ur, Uc);
    ffilter.upsample_filter(Dr, Dc);
    
    // demonstrate undo operation
    iimage.downsample_image(Ur, Uc);
    iimage.undo_operation();
    
    // zero pad the input image
    iimage.pad_image(Pl, Pt, Pr, Pb);
    
    
    
    // create input matrices
    cu::matrix2D<int> ffilter_mat(ffilter);
    cu::matrix2D<int> iimage_mat(iimage, ffilter);
    
    // create empty output tensor and matrix
    cu::image_tensor<int> oimage(iimage.get_rows() - ffilter.get_irows() + 1, iimage.get_cols() - ffilter.get_icols() + 1, No);
    cu::matrix2D<int> oimage_mat(oimage);
    
    
    
    // multiply the 2D matrices
    cu::mult_matrix2D(ffilter_mat, iimage_mat, oimage_mat);
    
    
    
    // downsample the output
    oimage.downsample_image(Sr, Sc);
    
    
    
    // visualize the results
    iimage.display("\n\nInput feature map");
    ffilter.display("\n\nFilter coefficients");
    oimage.display("\n\nOutput feature map");

    
    // Colors to print on console
    std::unordered_map<unsigned char, int> colors;
    for(uint i = 0; i < 256; i++) colors[i] = 30;
    colors[0] = 30;
    colors[1] = 31;
    colors[2] = 37;
    colors[3] = 37;
    colors[4] = 30;
    colors[7] = 34;
    colors[8] = 33;

    
    // Read image data from file "mario.jpg"
    std::ifstream mario_text("/Users/Abhijeet/Desktop/Fall 2018/CNN/Homeworks/Homework1/code/src/mario.txt");
    int img_data[62 * 47];
    for(int i = 0; i < 62; i++){
        for(int j = 0; j < 47; j++){
            mario_text >> img_data[i * 47 + j];
        }
    }
    
    // input image
    cu::image_tensor<int> mario(62, 47, 1);
    mario.initialize(tensor<int>::init_type::RANDOM, 0, img_data);
    
    // Display the input image
    std::cout << "\nImage before convolution:\n";
    for(int i = 0; i < mario.get_rows(); i++){
        for(int j = 0; j < mario.get_cols(); j++){
            std::cout << "\033[1;" << colors[mario.at(0, i, j)] << "m#\033[0m";
            std::cout << "\033[1;" << colors[mario.at(0, i, j)] << "m#\033[0m";
        }
        std::cout << "\n";
    }
    
    
    // filter
    cu::filter_tensor<int> ff(1, 1, 3, 3);
    ff.data[0] = ff.data[2] = ff.data[6] = ff.data[8] = 0;
    ff.data[1] = ff.data[3] = ff.data[5] = ff.data[7] = 1;
    ff.data[4] = -4;
    
    // pad the image to obtain same size output
    mario.pad_image(1, 1, 1, 1);
    
    // Convolve the image to obtain output
    cu::image_tensor<int> luigi(mario.get_rows() - ff.get_irows() + 1, mario.get_cols() - ff.get_icols() + 1, 1);
    cu::matrix2D<int> ff_mat(ff);
    cu::matrix2D<int> mario_mat(mario, ff);
    cu::matrix2D<int> luigi_mat(luigi);
    cu::mult_matrix2D(ff_mat, mario_mat, luigi_mat);
    
    
    // Display output image
    std::cout << "\nImage after convolution:\n";
    for(int i = 0; i < luigi.get_rows(); i++){
        for(int j = 0; j < luigi.get_cols(); j++){
            std::cout << "\033[1;" << colors[luigi.at(0, i, j)] << "m#\033[0m";
            std::cout << "\033[1;" << colors[luigi.at(0, i, j)] << "m#\033[0m";
            //std::cout << luigi.at(0, i, j) << "\t";
        }
        std::cout << "\n";
    }
    
    
    return 0;
}
