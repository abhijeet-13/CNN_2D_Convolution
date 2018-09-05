#include <iostream>
#include "conv_utils.h"
#include "tensor.h"

int main() {
    
    tensor<int> input_image(3, 3, 10, 20);
    input_image.init_seq();
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 20; j++) {
                std::cout << (int) input_image.get(c, i, j) << "-";
            }
            std::cout << std::endl;
        }
        std::cout << ">>\n>>\n";
    }
    
    getchar();
    return 0;
}
