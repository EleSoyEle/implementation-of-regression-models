//Calculamos los primeros gradientes para cada dato
__kernel void PsumMSE(
    __global double* y_true,
    __global double* y_pred,
    __global double* x,
    __global double* v_sum,
    __const int size,
    __const int nv){
        int k = get_global_id(0);
        if(k>0){
            double dif = y_true[k]-y_pred[2*k];
            v_sum[k]=dif*y_pred[2*k+1];
        }
        else{
            //Tecnicamente esto es designar a un nucleo especifico hacer una unica tarea jajaj
            for(int i=0;i<size;i++){
                double dif = y_true[i]-y_pred[2*i];
                v_sum[0]+=pow(dif,2);
            }
        }
    }

//Calculamos todos los gradientes
__kernel void GradsMSE(
    __global double* acc_loss,
    __global double* x,
    __global double* grads,
    __const int size,
    __const int nv){
    
    int i = get_global_id(0);
    if(i<nv){
        double ggrad = 0;
        for(int k=0;k<size;k++){
            ggrad+=acc_loss[k+1]*x[nv*k+i];
        }
        grads[i]=-2*ggrad/size;
    }
    else if(i==nv){
        double ggrad = 0;
        for(int k=0;k<size;k++){
            ggrad+=acc_loss[k+1];
        }
        grads[i]=-2*ggrad/size;
    }
    else if(i==nv+1){
        grads[i]=acc_loss[0];
    }
}


__kernel void regresion(
    __global double* x,
    __global double* predgrads,
    __global double* p,
    __const int sx,
    __const int sw,
    __const int use_sigmoid){
    int gid = get_global_id(0);
    int i = gid;
    for(int j=0;j<sw;j++){
        predgrads[i] += p[j]*x[i*sw+j];
        
    }
    predgrads[i]+=p[sw];
    if(use_sigmoid==1){
        double t_exp = exp(-predgrads[i]);
        double s = 1/(1+t_exp);
        predgrads[i]=s; //Prediccion con logits
        predgrads[i+sx]=s*(1-s); //Derivada de logits
    }
    else{
        predgrads[i+sx]=1;
    }
    
}