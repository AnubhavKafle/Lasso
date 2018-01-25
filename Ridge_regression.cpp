/*
 * =====================================================================================
 *
 *       Filename:  Ridge_regression.cpp
 *
 *    Description:  Python interface for ridge regression in multivariate linear regression with l2 regularization
 *
 *        Version:  1.0
 *        Created:  25.01.2018 13:20:27
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Anubhav Kaphle email: anubhavkaphle@gmail.com
 *   Organization:  Max Planck Institute for Biophysical Chemistry
 *
 * =====================================================================================
 */

#define DLLEXPORT extern "C"

#include <armadillo>
#include <iostream>

using namespace std;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  LLR_calc
 *  Description:  calculate log likelihood ratio calculation between two models from ridge
 * =====================================================================================
 */

    double 
LLR_calc ( double * expr, arma::vec & beta, arma::rowvec & response, int nsample, int ngene, double best_intercept )
{
    double LLR = 0.0;
    float sum = 0.0;
    float deflection = 0.0;

    for ( int s=0; s < nsample; s++)
    {
        sum = sum+response[s];
    }
    float mean = sum/nsample;
    for ( int s=0; s < nsample; s++)
    {
        deflection = deflection + pow((response[s] - mean),2);
    }
    float variance = deflection/nsample;
    for ( int i =0; i < nsample; i++)
    {
        float sample_effect = 0.0;
        for (int j=0; j<ngene; j++)
        {
                sample_effect = sample_effect + beta[j] * expr[(j * nsample + i)];  

        }       
        //float residual  = (pow(response[i],2) - pow(response[i] - sample_effect,2))/variance;
    double residual = pow(response[i] - sample_effect - best_intercept,2);
    LLR = LLR + residual;  

    }
    LLR = nsample - LLR/variance;
    return LLR;                        // This value is -2 * log(likelihood ratio)
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  mean_square_error_prediction
 *  Description:  Find the mean square error between fitted model and true data
 * =====================================================================================
 */
    double
mean_square_error_prediction (arma::mat & row_major_data_3, arma::rowvec & response, int nsample, arma::vec &betas, double & intercept)
{
    arma::vec prediction = (row_major_data_3 * betas) + intercept;
    arma::vec delta = response.t() - prediction;
    double mse = sum(square(delta)) / nsample;

    return mse;

}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  ridge_linear
 *  Description:  Peform a ridge regresion using OLS method given a lambda.
 *                It uses the Linear regression implementation of mlpack.
 * =====================================================================================
 */
    
arma::vec
ridge_linear ( arma::mat & row_major_data_2, int npred, int nsample, arma::rowvec &response, double lambda)
{
 

 arma::mat p = row_major_data_2;
 arma::rowvec r = response;

 p.insert_rows(nsample,npred);

 p.submat(nsample,0,npred+nsample-1,npred-1) = sqrt(lambda) * arma::eye<arma::mat>(npred,npred);  //inserting a submatrix of size P X P with values sqrt(lamda) * Identity matrix I

 arma::mat Q, R;

 arma::qr(Q,R,p);  // QR decomposition. orthogonal and right triangular matrix decomposition 

 r.insert_cols(nsample,npred,0);  // insert p number of 0 to the response vector

 arma::vec betas = solve(R, arma::trans(r * Q));   // solved using R * betas = (r * Q)^T, r is a row-vec

 return betas;

}

/* -----  end of function lasso_lars  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cross_validation_2fold
 *  Description:  Perform cross validation to find the best lambda
 *                To do: save betas ??
 * =====================================================================================
 */
        void
cross_validation_2fold ( arma::mat & row_major_data_1, int npred, arma::rowvec & response, int nsample, double* lambdas, int nlambda, double* cv_mse, double & best_lambda, arma::vec &best_betas, double & best_intercept )
{

    double best_mse;

    int ntrain    = nsample * 0.7 ;
    int nvalidate = nsample - ntrain;
    arma::mat x_train       = arma::mat(ntrain, npred);
    arma::mat x_validate    = arma::mat(nvalidate, npred);
    arma::rowvec y_train    = arma::rowvec(ntrain);
    arma::rowvec y_validate = arma::rowvec(nvalidate);

    for(unsigned int i = 0; i < nsample; i++)
    {
        if (i < ntrain)
        {
            x_train.row(i) = row_major_data_1.row(i);
            y_train[i] = response[i];
        }
        else
        {
            x_validate.row(i - ntrain) = row_major_data_1.row(i);
            y_validate[i - ntrain] = response[i];
        }
    }

    for (unsigned int i = 0; i < nlambda; i++) 
    {
        arma::rowvec xmean = mean(x_train,0);
        arma::mat x_train_scaled = x_train.each_row() - xmean;
        arma::rowvec y_train_scaled = y_train - mean(y_train);
        arma::vec betas = ridge_linear (x_train_scaled, npred, ntrain, y_train_scaled, lambdas[i]);
        double intercept = (mean(y_train) - (xmean * betas)).eval()(0,0);
        double mse = mean_square_error_prediction (x_validate, y_validate, nvalidate, betas, intercept);
        cv_mse[i] = mse;

        if ( i == 0 ) {

            best_mse = mse;
            best_lambda = lambdas[i];
            best_betas = betas;
            best_intercept = intercept;
        }
        
        if ( mse < best_mse ) {
            best_mse = mse;
            best_lambda = lambdas[i];
            best_betas = betas;
            best_intercept = intercept;
        }
    }


    return ;
}        /* -----  end of function cross_validation_2fold  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cvfold2
 *  Description:  
 * =====================================================================================
 */

DLLEXPORT
        bool
cvfold2 (  double* target, double* predictor, int npred, int nsample, double* lambdas, int nlambda, double* cv_mse, double* best_lambda, double* model_betas )
{
    arma::mat row_major_data = arma::mat(predictor, nsample, npred);
    arma::rowvec response = arma::rowvec(target, nsample);
    arma::vec best_betas = arma::vec(model_betas, npred);
    double my_best_lambda;
    double my_bst_intercept;
    cross_validation_2fold ( row_major_data, npred, response, nsample, lambdas, nlambda, cv_mse, my_best_lambda, best_betas, my_bst_intercept);
    best_lambda[0] = my_best_lambda;
    for (unsigned int i = 0; i < npred; i++) {
        model_betas[i] = best_betas[i];
    }
    double LLR = LLR_calc(predictor, best_betas, response, nsample, npred, my_bst_intercept);
    //double LLR = easy_LLR(row_major_data, best_betas, response, nsample);
    cout<<LLR<<endl;
    return true;
}        /* -----  end of function cvfold2  ----- */

