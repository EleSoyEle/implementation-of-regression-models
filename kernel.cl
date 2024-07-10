__kernel void regresion(
    __global double* x,
    __global double* predgrads,
    __global double* p,
    __const int sx,
    __const int sw,
    __const int use_logits){
    int gid = get_global_id(0);
    int i = gid;
    for(int j=0;j<sw;j++){
        predgrads[i] += p[j]*x[i*sw+j];
        
    }
    predgrads[i]+=p[sw];
    if(use_logits==1){
        double t_exp = exp(-predgrads[i]);
        double s = 1/(1+t_exp);
        predgrads[i]=s; //Prediccion con logits
        predgrads[i+sx]=t_exp*pow(s,2); //Derivada de logits
    }
    else{
        predgrads[i+sx]=1;
    }
    
}