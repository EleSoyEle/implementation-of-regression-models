/*
Al ejecutar cualquier funcion que haga predicciones va a calcular sus gradientes para el backpropagation
Por ejemplo

double* softmax(**args){
    soft = (double*)calloc(2,sizeof(double));
    ...
    soft[0] = softmax_pred
    soft[1] = softmax_grads
    return soft
}
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/opencl.h> 

double* a_zeros(int size){
    return (double*)calloc(size,sizeof(double));
}
double** m_zeros(int s1,int s2){
    double** z = (double**)calloc(s1,sizeof(double*));
    for(int i=0;i<s1;i++){
        z[i]=a_zeros(s2);
    }
    return z;
}
// Establecemos el retorno del gradiente
double* logits(double x){
    double* out = (double*)calloc(2,sizeof(double));
    double t_exp = exp(-x);
    double s = 1/(1+t_exp);
    out[0]=s; //Logit
    out[1]=t_exp*pow(s,2); //dlogit
    return out;
}

//
//m(x1,x2,...,xn)=w1*x1+...+wn*xn+b
//
double** regresion(double** x,double* p, int sx,int sw,int use_logits){
    double** predgrads = m_zeros(sx,2);
    
    for(int i=0;i<sx;i++){
        for(int j=0;j<sw;j++){
            predgrads[i][0] += p[j]*x[i][j];
        
        }
        predgrads[i][0]+=p[sw];
        if(use_logits==1){
            double* lg = logits(predgrads[i][0]);
            predgrads[i][0]=lg[0];
            predgrads[i][1]=lg[1];
        }
        else{
            predgrads[i][1]=1;
        }
    }
    return predgrads;
}

double* MSE(double* y_true,double** y_pred,double** x,int size,int nv){
    double* loss = a_zeros(nv+2);
    for(int i=0;i<size;i++){
        double dif = y_true[i]-y_pred[i][0];
        loss[0]+=pow(dif,2);
        double dsdw = dif*y_pred[i][1];//pred[i][1]==dlogits[i]
        for(int j=0;j<nv;j++){
            loss[j+1] -= dsdw*x[i][j];
        }
        loss[nv+1] -= dsdw;
    }
    loss[0] = loss[0]/size;
    for(int i=0;i<nv+1;i++){
        loss[i+1] = 2*loss[i+1]/size;
    }
    return loss;

}

char* readTextFile(char filename[]){
    FILE* file = fopen(filename,"r");
    if(file == NULL){
        perror("Error al abrir el archivo");
    }
    fseek(file,0,SEEK_END);
    long file_size = ftell(file);
    fseek(file,0,SEEK_SET);
    char* KernelS = (char*)malloc((file_size+1)*sizeof(char));
    fread(KernelS,1,file_size,file);
    fclose(file);
    KernelS[file_size]='\0';
    return KernelS;
}
