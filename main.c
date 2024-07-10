//El codigo entrena dos tipos de regresiones para n variables de entrada
//El modelo puede ser logistico o lineal

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/opencl.h> 
#include "utils.c"

cl_int cerror = CL_SUCCESS;
int main(){
    int nv,size,m;
    printf("Ingresa el numero de variables de tu regresion: ");
    scanf("%d",&nv);
    printf("Ingresa el numero de datos de entrenamiento: ");
    scanf("%d",&size);
    printf("Con logits?[y:1,n:0]");
    scanf("%d",&m);
    
    double** data = m_zeros(size,nv);
    double* target = a_zeros(size);

    for(int i=0;i<size;i++){
        for(int j=0;j<nv;j++){
            printf("X[%d][%d]: ",i,j);
            scanf("%lf",&data[i][j]);
        }
        printf("Y[%d]",i);
        scanf("%lf",&target[i]);
        printf("\n");
    }
    const char* KernelSource = readTextFile("kernel.cl");
    const cl_uint num = 1;
    cl_device_type devt = CL_DEVICE_TYPE_CPU;
    clGetDeviceIDs(NULL,devt,0,NULL,(cl_uint*)&num);
    cl_device_id devices[1];
    clGetDeviceIDs(NULL,devt,num,devices,NULL);

    cl_context context = clCreateContextFromType(NULL,devt,NULL,NULL,&cerror);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context,devices[0],NULL,NULL);
    cl_program program = clCreateProgramWithSource(context,1,(const char**)&KernelSource,NULL,NULL);

    clBuildProgram(program,num,devices,NULL,NULL,NULL);

    cl_build_status buildStatus;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);
    if (buildStatus != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        printf("Error de compilación:\n%s\n", log);
        free(log);
        return -1;
    }

    double* p = a_zeros(nv+1);
    double lr;

    int epochs;
    printf("Ingresa el numero de pasos: ");
    scanf("%d",&epochs);
    printf("\nIngresa el lr de tu modelo: ");
    scanf("%lf",&lr);
    printf("\nEntrenando...");
    for(int epoch=0;epoch<epochs;epoch++){
        double** pred = regresion_cl(program,queue,context,data,p,size,nv,m);
        //double** pred = regresion(data,p,size,nv,m);

        double* loss = MSE(target,pred,data,size,nv);
        printf("\nLoss: %lf",loss[0]);
        for(int i=0;i<nv+1;i++){
            p[i]=p[i]-lr*loss[i+1];
        }
    }
    printf("\n Valores calculados");
    for(int i=0;i<nv+1;i++){
        printf("\n%lf",p[i]);
    }
}