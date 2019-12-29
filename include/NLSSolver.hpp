/*
solve y=exp(ax^2+bx+c)
by gauss-newton, LM and Dog-Leg algorithm
*/


#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <algorithm>

namespace nlssolver{

class NLSSolver{

public:

    NLSSolver(double a,double b,double c)
     : a_(a),b_(b),c_(c) {}

    //Solve system using Gauss-Newton algorithm
    bool solveByGN();
    //Solve system using LenvenbergMarquardt algorithm
    bool solveByLM();
    //Solve system using Dog-Leg algorithm
    bool solveByDogLeg();

    //add observation
    void addObservation(double x,double y){
        observations_.push_back(Eigen::Vector2d(x,y));
    }

    //set initial states
    void setInitialStates(double a,double b,double c){
        a_ = a;
        b_ = b;
        c_ = c;
    }

    //Set maximum numbers of iteration 
    void setMaximumIterations(int num){
        maximumIterations_ = num;
    }
    //Set estimation precision
    void setEstimationPrecision(double epsilon){
        epsilon_ = epsilon;
    }


private:
    
    //Compute jacobian matrix
    inline void computeJacobianAndError(){
        jacobian_.resize(observations_.size(),3);
        error_.resize(observations_.size());
        for(size_t i = 0; i < observations_.size(); ++i){
            double xi = observations_[i](0);
            double yi = observations_[i](1);
            Eigen::Matrix<double,1,3> jacobian_i;
            double exp_y = exp(a_ * xi * xi + b_ * xi + c_);
            jacobian_i(0,0) = exp_y * xi * xi;
            jacobian_i(0,1) = exp_y * xi;
            jacobian_i(0,2) = exp_y;
            jacobian_.row(i) = jacobian_i;
            error_(i) = exp_y - yi;
        }
    }

    //Compute hessian matrix and b 
    inline void computeHessianAndg(){
        hessian_ = jacobian_.transpose() * jacobian_;
        g_ = -jacobian_.transpose() * error_;
    }

    //Solve linear system
    inline void solveLinearSystem(){
        delta_x_ = hessian_.inverse() * g_;
    }

    //Update states
    inline void updateStates(){
        a_ += delta_x_(0);
        b_ += delta_x_(1);
        c_ += delta_x_(2);
    }


private:

    //The states that need to be estimated
    double a_;
    double b_;
    double c_;

    int maximumIterations_;
    double epsilon_;

    //Jacobian matrix
    Eigen::MatrixXd jacobian_;
    //Hessian matrix
    Eigen::Matrix3d hessian_;
    Eigen::Vector3d g_;
    //estimation error
    Eigen::VectorXd error_;
    //squared estimation error
    double squaredError_;
    //delta x
    Eigen::Vector3d delta_x_;

    //observations
    std::vector<Eigen::Vector2d> observations_;


};//class NLSSolver

}//namespace nlssolver