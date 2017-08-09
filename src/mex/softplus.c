#include "mex.h"
#include "matrix.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mwSize *dims;
    mwSize i, j, dim_x, dim_y;
    double *mat_in, *mat_out;
    
    /* Read input */
    mat_in  = mxGetPr(prhs[0]);
    dims    = mxGetDimensions(prhs[0]);
    dim_x   = dims[0];
    dim_y   = dims[1];
    
    /* Create output */
    plhs[0] = mxCreateDoubleMatrix(dim_x, dim_y, mxREAL);
    mat_out = mxGetPr(plhs[0]);
    
    /* The computations */
    for (i = 0; i < dim_x; ++i) {
        for (j = 0; j < dim_y; ++j) {
            double result = mat_in[i * dim_y + j];
            mat_out[i * dim_y + j] = log(1+exp(result));
        }
    }
    
    return;
}