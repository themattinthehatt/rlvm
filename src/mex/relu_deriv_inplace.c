#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mwSize *dims;
    mwSize i, j, dim_x, dim_y;
    double *mat_in;
    
    /* Read input */
    mat_in  = mxGetPr(prhs[0]);
    dims    = mxGetDimensions(prhs[0]);
    dim_x   = dims[0];
    dim_y   = dims[1];
    
    /* Create output */
    plhs[0] = prhs[0];
    
    /* The computations */
    for (i = 0; i < dim_x; ++i) {
        for (j = 0; j < dim_y; ++j) {
            if (mat_in[i * dim_y + j] < 0) {
                mat_in[i * dim_y + j] = 0;
            } else {
                mat_in[i * dim_y + j] = 1;
            }
        }
    }
    
    return;
}