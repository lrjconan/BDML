#include <iostream>
#include <matrix.h>
#include <math.h>
#include "mex.h"

using namespace std;

#define EPS 2.220446049250313e-16
#define MIN_SINGLE 1.1754944e-38
#define ABS(x) (((x) >= 0) ? (x) : -(x))

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
		prhs[4] : epsilon	scalar
	
	Output:
		plhs[0] : w			(m x 1) vector
		plhs[1] : rho		(m x 1) scalar
	
	Author:
		Renjie Liao
		2014/05/14
	*/
/**********************************************************************************************************/    
    /* Check for proper number of input and output arguments */
    if (nrhs != 5) {
        mexErrMsgTxt("Six input arguments required.");
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
    mwSize m, n, d;
    mwIndex *irs, *jcs;
	mxArray *idxA;
    int i, j, k, x, y, current_row_index, starting_row_index, stopping_row_index;
    double *prs, *wI, *X, *b, *w, *rho, *cv;
    double sumW(.0), epsilon;

    /* Get the size and pointers to input data */
    m		= mxGetM(prhs[0]);
    n		= mxGetN(prhs[0]);
	d		= mxGetM(prhs[1]);
    
	wI		= mxGetPr(prhs[0]);
    X		= mxGetPr(prhs[1]);
	b		= mxGetPr(prhs[3]);	
	epsilon = mxGetScalar(prhs[4]);
    
    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    plhs[1] = mxCreateDoubleScalar(0.0);
	cv		= new double [m];
	w		= mxGetPr(plhs[0]);
	rho		= mxGetPr(plhs[1]);

	for (i = 0; i < m; ++i) {
		k		= 0;
		
		idxA	= mxGetCell(prhs[2], i);
		prs		= mxGetPr(idxA);
		irs		= mxGetIr(idxA);
		jcs		= mxGetJc(idxA);
				
		// read sparse matrix A{i}
		for (j = 0; j < d; j++) { 
			starting_row_index = jcs[j]; 
			stopping_row_index = jcs[j+1];

			if (starting_row_index == stopping_row_index)
				continue;
			else {
				for (current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
					x = j;
					y = irs[current_row_index];
					cv[i] += X[x*d + y]*prs[k++];		
				}
			}
		}

		cv[i] -= b[i];

		if (ABS(cv[i]) > *rho)
			*rho = ABS(cv[i]); 			
	}

	*rho += EPS;   
    
	for (int i = 0; i < m; ++i) {
		w[i]	= max(wI[i] * (1.0 - cv[i]*epsilon/(*rho)), MIN_SINGLE);
		sumW	+= w[i];
	}
	
	for (int i = 0; i < m; ++i)
		w[i] /= sumW;
}
