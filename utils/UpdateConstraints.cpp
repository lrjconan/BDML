#include <iostream>
#include <matrix.h>
#include "mex.h"

using namespace std;

void mexFunction(
		 int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]
		 )
{
/**********************************************************************************************************/ 
	/*
	Update Constarints Function

	Input:
		prhs[0] : w			(m x 1) vector
		prhs[1] : C			(m x 1) cell array in which each element is a d x d sparse matrix
		prhs[2] : d			scalar
	
	Output:
		plhs[0] : B			(d x d) matrix
	
	Author:
		Renjie Liao
		2014/05/14
	*/
/**********************************************************************************************************/    
    /* Check for proper number of input and output arguments */
    if (nrhs != 3) {
        mexErrMsgTxt("Three input arguments required.");
    }
    if(nlhs > 1){
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument  */
    if (!(mxIsCell(prhs[1]))){
        mexErrMsgTxt("2nd input argument must be of type cell.");
    }  

/**********************************************************************************************************/ 
    /* Declare variable */
    mwSize m, d;
    mwIndex *irs, *jcs;
	mxArray *idxC;
    int i, j, x, y, idx_ir_current, idx_ir_begin, idx_ir_end;
    double *prs, *B, *w;

    /* Get the size and pointers to input data */
    m		= mxGetM(prhs[0]);
        
	w		= mxGetPr(prhs[0]);    
	d		= mxGetScalar(prhs[2]);
	    
    plhs[0] = mxCreateDoubleMatrix(d, d, mxREAL);    
	B		= mxGetPr(plhs[0]);

	for (i = 0; i < m; ++i) 
    {		
		idxC	= mxGetCell(prhs[1], i);
		prs		= mxGetPr(idxC);
		irs		= mxGetIr(idxC);
		jcs		= mxGetJc(idxC);
				
		for (j = 0; j < d; j++) { 
			idx_ir_begin = jcs[j]; 
			idx_ir_end = jcs[j+1];

			if (idx_ir_begin == idx_ir_end)
				continue;
			else {		
				for (idx_ir_current = idx_ir_begin; idx_ir_current < idx_ir_end; idx_ir_current++) {
					x = j;
					y = irs[idx_ir_current];

					B[x*d + y] += w[i]*prs[idx_ir_current];
				}
			}
		}			
	}
}
