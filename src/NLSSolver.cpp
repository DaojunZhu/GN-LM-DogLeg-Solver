#include "NLSSolver.hpp"

namespace nlssolver{

using namespace Eigen;
using namespace std;

bool NLSSolver::solveByGN(){
    cout << "Guass-Newton Solver processing...." << endl;    
    int iter = 0;   
    double current_squared_error;
    double delta_norm;

    while(iter++ < maximumIterations_){
        computeJacobianAndError();
        current_squared_error = error_.squaredNorm();
        cout << "---Current squared error: " << current_squared_error << endl;
        computeHessianAndg();
        solveLinearSystem();
        delta_norm = delta_x_.norm();
        if(delta_norm < epsilon_)
            break;
        updateStates(); 
    }

    cout << "Gauss-Newton Solver ends." << endl;
    cout << "Iterations: " << iter << endl;
    cout << "The optimized value of a,b,c : " << endl
        << "--- a = " << a_ << endl
        << "--- b = " << b_ << endl
        << "--- c = " << c_ << endl;
    return true;
}//solveByGN


bool NLSSolver::solveByLM(){
    cout << "LenvenbergMarquardt Solver processing ..." << endl;
    int iter = 0;
    double current_squared_error=0.0;
    double delta_norm;
    double v = 2.0;
    double rou;
    double tau = 1e-5;
    double lambda;
    int inner_iterations = 10;
    int inner_iter ;
    
    computeJacobianAndError();
    computeHessianAndg();

    //Initial lambda
    double maxDiagonal = 0.;
    for(int i = 0; i < 3; ++i){
        maxDiagonal = max(fabs(hessian_(i,i)),maxDiagonal);
    }
    lambda = tau * maxDiagonal;

    while (iter++ < maximumIterations_)
    {
        current_squared_error = error_.squaredNorm();
        cout << "--Current squared error: " << current_squared_error << endl;
        inner_iter = 0;
        //Try to find a valid step in maximum iterations : inner_iter
        while (inner_iter++ < inner_iterations)
        {
            cout << "----Current lambda: " << lambda << endl;
            Matrix3d damper = lambda * Matrix3d::Identity();
            Vector3d delta = (hessian_+damper).inverse()*g_;
            double new_a = a_ + delta(0);
            double new_b = b_ + delta(1);
            double new_c = c_ + delta(2);
            delta_norm = delta.norm();
            //compute the new error after the step
            double new_squared_error = 0.0;
            for(size_t i = 0; i < observations_.size(); ++i){
                double xi = observations_[i](0);
                double yi = observations_[i](1);
                double exp_y = exp(new_a*xi*xi+new_b*xi+new_c);
                double error_i = exp_y - yi;
                new_squared_error += error_i * error_i;
            }
            //gain ratio
            rou = (current_squared_error-new_squared_error) / 
                (0.5*delta.transpose()*(lambda*delta+g_)+1e-3);

            //a valid iteration step
            if(rou > 0){
                //update states
                a_ = new_a;
                b_ = new_b;
                c_ = new_c;
                //update lamda
                lambda = lambda * max(1.0/3.0, 1-pow((2*rou-1),3));
                v = 2;
                break;
            }
            //An invalid iteration step
            else{
                //update lamda 
                lambda = lambda * v;
                v = 2 * v;
            }
        }

        if(delta_norm < epsilon_)
            break;

        computeJacobianAndError();
        computeHessianAndg();

    }

    cout << "lon Solver ends." << endl;
    cout << "Iterations: " << iter << endl;
    cout << "The optimized value of a,b,c : " << endl
        << "--- a = " << a_ << endl
        << "--- b = " << b_ << endl
        << "--- c = " << c_ << endl;
    return true;
}//solveByLM


bool NLSSolver::solveByDogLeg(){
    cout << "Dog Leg Solver processing ..." << endl;

    //parameters
    int iter = 0;
    double v = 2.;
    double radius = 100;   //initial trust region radius
    double rou;     //gain ratio
    //The number of iterations to find a valid step
    int inner_iterations = 10; 
    int inner_iter = 0;
    
    double current_squared_error = 0.;
    double delta_norm;

    while (iter++ < maximumIterations_)
    {
        computeJacobianAndError();
        computeHessianAndg();

        current_squared_error = 0.5*error_.squaredNorm();
        cout << "--Current squared error: " << current_squared_error << endl;

        //compute step for sdd
        double alpha = g_.squaredNorm() / 
            (jacobian_ * g_).squaredNorm();
        //steepest descent step at current estimation point
        Vector3d h_sd = alpha * g_;
        //gauss-newton step at current estimation point 
        Vector3d h_gn = hessian_.inverse() * g_;

        Vector3d h_dl;

        //iterate to find a good dog leg step
        while(inner_iter++ < inner_iterations)
        {
            //compute dog leg step for 
            //current trust region radius(Delta)
            int flag_choice;
            double belta;
            if(h_gn.norm() <= radius){
                h_dl = h_gn;
                flag_choice = 0;
            }
            else if(h_sd.norm() >= radius){
                h_dl = radius / (h_sd.norm()) * h_sd;
                flag_choice = 1;
            }
            else{
                double c = h_sd.transpose()*(h_gn-h_sd);
                double sqrt_temp;
                sqrt_temp = sqrt(c*c+(h_gn-h_sd).squaredNorm()*
                    (radius*radius-h_sd.squaredNorm()));
                if(c <= 0){
                    belta = (-c+sqrt_temp)/((h_gn-h_sd).squaredNorm()+1e-3);
                }
                else{
                    belta = (radius*radius-h_sd.squaredNorm())/(c+sqrt_temp);
                }
                h_dl = h_sd + belta * (h_gn - h_sd);
                flag_choice = 2;
            }

            delta_norm = h_dl.norm();
            double new_a = a_ + h_dl(0);
            double new_b = b_ + h_dl(1);
            double new_c = c_ + h_dl(2);

            //compute gain ratio
            double new_squared_error = 0.;
            for(auto obs : observations_){
                double xi = obs(0);
                double yi = obs(1);
                double exp_y = exp(new_a*xi*xi+new_b*xi+new_c);
                double error_i = exp_y - yi;
                new_squared_error += error_i * error_i;
            }
            new_squared_error *= 0.5;
            double model_decreased_error;
            switch (flag_choice)
            {
            case 0:
                model_decreased_error = 
                    current_squared_error;
                break;
            case 1:
                model_decreased_error = 
                    (radius*(2*h_sd.norm()-radius)) / (2*alpha);
                break;
            case 2:
                model_decreased_error = 
                    0.5*alpha*(1-belta)*(1-belta)*g_.squaredNorm()+
                        belta*(2-belta)*current_squared_error;
                break;
            }
            rou = (current_squared_error - new_squared_error) / 
                model_decreased_error;
            
            //a valid decreasing step
            if(rou > 0){
                //update states
                a_ = new_a;
                b_ = new_b;
                c_ = new_c;
                //update trust region radius
                radius = radius / max(1.0/3.0,1-pow(2*rou-1,3));
                v = 2;
                break;
            }
            else{
                radius = radius / v;
                v = v * 2;
            }
        }

        //found 
        if(delta_norm < epsilon_)
            break;
        
    }
    
    cout << "Dog Leg Solver ends." << endl;
    cout << "Iterations: " << iter << endl;
    cout << "The optimized value of a,b,c : " << endl
        << "--- a = " << a_ << endl
        << "--- b = " << b_ << endl
        << "--- c = " << c_ << endl;

    return true;
}

}//nlssolver