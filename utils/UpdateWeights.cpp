#include <iostream>
#include <matrix.h>
#include <math.h>
#include "mex.h"

using namespace std;

#define EPS 2.220446049250313e-16
#define MIN_SINGLE 1.1754944e-38

void mexFunction(
		 int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]
		 )
{
/**********************************************************************************************************/ 
	/*
	Update Weights Function

	Input:
		prhs[0] : wI		(m x 1) vector
		prhs[1] : X			(d x d) matrix 
		prhs[2] : A			(m x 1) cell array in which each element is a d x d sparse matrix
		prhs[3] : b			(m x 1) vector 
		prhs[4] : rho		scalar 
		prhs[5] : epsilon	scalar
        prhs[6] : ell       scalar
	
	Output:
		plhs[0] : w			(m x 1) vector
		plhs[1] : cv		(m x 1) matrix
	
	Author:
		Renjie Liao
		2014/05/14
	*/
/**********************************************************************************************************/    
    /* Check for proper number of input and output arguments */
    if (nrhs != 7) {
        mexErrMsgTxt("Seven input arguments required.");
    }
    if(nlhs > 2){
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument  */
    if (!(mxIsCell(prhs[2]))){
        mexErrMsgTxt("3rd input argument must be of type cell.");
    }    

/**********************************************************************************************************/ 
    /* Declare variable */
    mwSize m, d;
    mwIndex *irs, *jcs;
	mxArray *idxA;
    int i, j, x, y, idx_ir_current, idx_ir_begin, idx_ir_end;
    double *prs, *wI, *X, *b, *w, *cv;
    double sumW(.0), const1, const2, rho, eps, ell;

    /* Get the size and pointers to input data */
    m		= mxGetM(prhs[0]);
	d		= mxGetM(prhs[1]);
    
	wI		= mxGetPr(prhs[0]);
    X		= mxGetPr(prhs[1]);
	b		= mxGetPr(prhs[3]);
	rho		= mxGetScalar(prhs[4]);
	eps		= mxGetScalar(prhs[5]);
    ell		= mxGetScalar(prhs[6]);
    
    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m, 1, mxREAL);
	w		= mxGetPr(plhs[0]);
	cv		= mxGetPr(plhs[1]);
	const1	= eps/rho;
    const2	= eps/ell;

	for (i = 0; i < m; ++i) 
    {
		idxA	= mxGetCell(prhs[2], i);
		prs		= mxGetPr(idxA);
		irs		= mxGetIr(idxA);
		jcs		= mxGetJc(idxA);
				
		// read sparse matrix A{i}
		for (j = 0; j < d; j++) { 
			idx_ir_begin    = jcs[j]; 
			idx_ir_end      = jcs[j+1];

			if (idx_ir_begin == idx_ir_end)
				continue;
			else {
				for (idx_ir_current = idx_ir_begin; idx_ir_current < idx_ir_end; idx_ir_current++) 
                {
					x       = j;
					y       = irs[idx_ir_current];
                    cv[i]   += X[x*d + y]*prs[idx_ir_current];
				}
			}
		}

        // Update formula I (simple and shares the same bound with II)
		cv[i]   -= b[i];
        // w[i]    = max(wI[i] * (1.0 - cv[i]*constC), EPS);    // a numerical trick which stabilizes the result        
        
        /*
        if (cv[i] > rho)
            mexErrMsgTxt("Positive rho is too small!");
        else if (cv[i] < -ell)
            mexErrMsgTxt("Negative ell is too small!");
        */
              
        if (cv[i] > 0)
            w[i] = max(wI[i] * (1.0 - cv[i]*const1), MIN_SINGLE); 
        else if (cv[i] < 0)
            w[i] = max(wI[i] * (1.0 - cv[i]*const2), MIN_SINGLE); 
        
        // Update formula II
        /* 
        cv[i]	-= b[i];
		cv[i]   /= rho; 
        if (cv[i] >= 0)
            w[i] = wI[i] * pow((1.0 - eps), cv[i]);
        else
            w[i] = wI[i] * pow((1.0 + eps), -cv[i]);        
        */
        
		sumW += w[i];
	}
	
	for (int i = 0; i < m; ++i)
		w[i] /= sumW;
}
