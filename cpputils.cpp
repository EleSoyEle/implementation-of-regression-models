#include <math.h>
#include <iostream>
#include <cstdlib>  

using namespace std;
//Implementacion del optimizador Adam
class Adam {
    public:
        double lr;
        double beta1=0.9;
        double beta2=0.99;
        double decay=0;
        double* mt = (double*)calloc(1,sizeof(double));
        double* vt = (double*)calloc(1,sizeof(double));
        double* mt_1 = (double*)calloc(1,sizeof(double));
        double* vt_1 = (double*)calloc(1,sizeof(double));
        int steps=0;
        double epsilon=1e-7;

        void build_optimizer(int size){
            mt = (double*)realloc(mt,size*sizeof(double));
            vt = (double*)realloc(vt,size*sizeof(double));
            mt_1 = (double*)realloc(mt_1,size*sizeof(double));
            vt_1 = (double*)realloc(vt_1,size*sizeof(double));
        }

        void apply_gradients(double* parameters,double* gradients,int size){
            steps++;
            for(int i=0;i<size;i++){
                mt[i] = beta1*mt[i]+(1-beta1)*gradients[i];
                vt[i] = beta2*vt[i]+(1-beta2)*pow(gradients[i],2);

                double mt_ = mt[i]/(1-pow(beta1,steps));
                double vt_ = vt[i]/(1-pow(beta2,steps));

                parameters[i]-=lr*mt_/(sqrt(vt_)+epsilon);
            }

        }
        
};