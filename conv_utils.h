#include <vector>
#include <stdarg.h>

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

// type of data
enum content_data_type{
    RANDOM,
    SEQUENTIAL
};

