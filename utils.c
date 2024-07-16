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


void show_array(double* array,int size){
    printf("\n");
    for(int i=0;i<size;i++){
        printf("%f ",array[i]);
    }
    printf("\n");
}

void show_matrix(double** mat,int size[]){
    for(int i=0;i<size[0];i++){
        for(int j=0;j<size[1];j++){
            printf("%f ",mat[i][j]);
        }
        printf("\n");
    }
}


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

/*
Ejemplo:
[[1,2,3],[2,3,4],[3,5,6]] ----> [1,2,3,2,3,4,3,5,6]

*/
double* twod2oned(double** array,int s[2]){
    double* new_array = (double*)calloc(s[0]*s[1],sizeof(double));
    for(int i=0;i<s[0];i++){
        for(int j=0;j<s[1];j++){
            new_array[i*s[1]+j]=array[i][j];
        }
    }
    return new_array;
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

double** aug_pg(double* array,int sx){
    double** pg = (double**)calloc(sx,sizeof(double*));
    for(int i=0;i<sx;i++){
        pg[i] = (double*)calloc(2,sizeof(double));
        pg[i][0]=array[i];
        pg[i][1]=array[i+sx];
    }
    return pg;
}

//Calcula un batch de datos completo con opencl
//m(x1,x2,...,xn)=w1*x1+...+wn*xn+b
cl_int kerr1 = CL_SUCCESS;
double** regresion_cl(
    cl_program program,
    cl_command_queue queue,
    cl_context context,
    double** x,double* p,
    int sx,int sw,
    int use_logits){
    double* predgrads = a_zeros(sx*2);
    double* x1d = twod2oned(x,(int[2]){sx,sw});
    cl_mem buff_x = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(double)*sw*sx,x1d,NULL);
    cl_mem buff_pg = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(double)*sx*2,predgrads,NULL);
    cl_mem buff_p = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(double)*(sw+1),p,NULL);
    cl_kernel kernel = clCreateKernel(program,"regresion",&kerr1);

    if(kerr1!=CL_SUCCESS){
        printf("Error al crear el kernel:%d \n",kerr1);
    }
    clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&buff_x);
    clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&buff_pg);
    clSetKernelArg(kernel,2,sizeof(cl_mem),(void*)&buff_p);
    clSetKernelArg(kernel,3,sizeof(cl_int),&sx);
    clSetKernelArg(kernel,4,sizeof(cl_int),&sw);
    clSetKernelArg(kernel,5,sizeof(cl_int),&use_logits);
    size_t work_size = sx;
    clEnqueueNDRangeKernel(queue,kernel,1,NULL,&work_size,NULL,0,NULL,NULL);
    clEnqueueReadBuffer(queue,buff_pg,CL_TRUE,0,sizeof(double)*sx*2,predgrads,0,NULL,NULL);
    
    double** predgrads_2d = aug_pg(predgrads,sx);
    clReleaseKernel(kernel);
    clReleaseMemObject(buff_p);
    clReleaseMemObject(buff_x);
    clReleaseMemObject(buff_pg);
    return predgrads_2d;
}

//Calcula un batch de datos completo sin opencl
double** regresion(
    double** x,double* p,
    int sx,int sw,
    int use_logits){
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

/*
Funcion de error sin opencl

Aclaracion sobre nv:
nv es el numero de parametros que tiene cada uno de nuestros modelos sin contar el bias
loss es un puntero con el gradiente del error con respecto a cada parametro
loss[0] es el error total del modelo para todo el set de datos

*/
double* MSE(double* y_true,double** y_pred,double** x,int size,int nv){
    double* loss = a_zeros(nv+2);
    for(int i=0;i<size;i++){
        double dif = y_true[i]-y_pred[i][0];
        loss[0]+=pow(dif,2);
        double dsdw = dif*y_pred[i][1];//pred[i][1]==dlogits[i]
        for(int j=0;j<nv;j++){
            loss[j+1] += dsdw*x[i][j];
        }
        loss[nv+1] += dsdw;
    }
    loss[0] = loss[0]/size;
    for(int i=0;i<nv+1;i++){
        loss[i+1] = -2*loss[i+1]/size;
    }
    return loss;

}
cl_int kerr2 = CL_SUCCESS;
double* MSE_CL(
    cl_program program,
    cl_command_queue queue,
    cl_context context,
    double* y_true,double** y_pred,double** x,int size,int nv){
    
    double* y_pred_r = twod2oned(y_pred,(int[2]){size,2});
    double* x_r = twod2oned(x,(int[2]){size,nv});
    //nv es el numero de parametros sin contar el bias, y como se concatena al final el error total del modelo le añadimos 2
    int s_vars = nv+2;

    double* loss = a_zeros(s_vars);
    double* grads = a_zeros(s_vars);
    cl_mem buff1 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(double)*size,y_true,NULL);
    cl_mem buff2 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(double)*size*2,y_pred_r,NULL);
    cl_mem buff3 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(double)*size*nv,x_r,NULL);

    cl_mem out_buff = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(double)*(size+1),loss,NULL);

    cl_kernel kernel1 = clCreateKernel(program,"PsumMSE",NULL);
    cl_kernel kernel2 = clCreateKernel(program,"GradsMSE",&kerr2);
    clSetKernelArg(kernel1,0,sizeof(cl_mem),(void*)&buff1);
    clSetKernelArg(kernel1,1,sizeof(cl_mem),(void*)&buff2);
    clSetKernelArg(kernel1,2,sizeof(cl_mem),(void*)&buff3);
    clSetKernelArg(kernel1,3,sizeof(cl_mem),(void*)&out_buff);
    clSetKernelArg(kernel1,4,sizeof(cl_int),&size);
    clSetKernelArg(kernel1,5,sizeof(cl_int),&nv);

    size_t s = size;
    clEnqueueNDRangeKernel(queue,kernel1,1,NULL,&s,NULL,0,NULL,NULL);
    clEnqueueReadBuffer(queue,out_buff,CL_TRUE,0,sizeof(double)*(size),loss,0,NULL,NULL);

    clFinish(queue);
    //clReleaseKernel(kernel1);

    cl_mem grads_buff = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(double)*(s_vars),grads,NULL);
    clSetKernelArg(kernel2,0,sizeof(cl_mem),(void*)&out_buff);
    clSetKernelArg(kernel2,1,sizeof(cl_mem),(void*)&buff3);
    clSetKernelArg(kernel2,2,sizeof(cl_mem),(void*)&grads_buff);
    clSetKernelArg(kernel2,3,sizeof(cl_int),&size);
    clSetKernelArg(kernel2,4,sizeof(cl_int),&nv);

    size_t s2 = s_vars;
    clEnqueueNDRangeKernel(queue,kernel2,1,NULL,&s2,NULL,0,NULL,NULL);
    clEnqueueReadBuffer(queue,grads_buff,CL_TRUE,0,sizeof(double)*(s_vars),grads,0,NULL,NULL);
    
    clReleaseMemObject(buff1);
    clReleaseMemObject(buff2);
    clReleaseMemObject(buff3);
    clReleaseMemObject(out_buff);
    clReleaseMemObject(grads_buff);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    return grads;
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
