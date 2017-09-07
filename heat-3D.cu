#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include "../b2r_config.h"
//typedef float CELL_DT;

#define CUDA_BLOCK_SIZE    32
#define Alpha         0.1
#define Dt            0.1
#define Dx            1.0
#define Dy            1.0
#define Dz            0.5

#define coef_d2_a        (-1.f/12.f)
#define coef_d2_b        (4.f/3.f)
#define coef_d2_c        (-2.5f)

#define coef_d3_a        (1.f/90.f)
#define coef_d3_b        (-3.f/20.f)
#define coef_d3_c        (1.5f)
#define coef_d3_d        (-49.f/18.f)

#define coef_d4_a        (-1.f/560.f)
#define coef_d4_b        (8.f/315.f)
#define coef_d4_c        (-0.2f)
#define coef_d4_d        (1.6f)
#define coef_d4_e        (-205.f/72.f)
// define as global memory to avoid allocate and free each time, 
// (does not work sometimes, need to figure out why)
CELL_DT *d_temp1, *d_temp2;  
using namespace std;

#define cudaErrchk(ans)  cudaAssert((ans), __FILE__, __LINE__) 
inline void cudaAssert(cudaError_t code, string file, int line){
    if (code != cudaSuccess){
        cerr << "CUDA Error: " << cudaGetErrorString(code) << "; file: " << file << ", line:" << line << endl;
        exit(-1);
    }
}

/*
***************************************************************************************************
* func   name: fdm_heat_diffuse_delta1_verify
* description: cuda kernel function to update (2nd order accurancy) in Z direction with
*              this function is mostly for results verification under different R
* parameters :
*             d_temp1: input
*             d_temp2: output
*             Ngx, Ngy, Ngz: subdomain dimension
*             b2r_i: loop id in subdomain
*             xef, yef, zef: indicate the position of subdomain in the simulation environment
*             b2r_R: parameter R in B2R framework
* return: none
***************************************************************************************************
*/
__global__ void fdm_heat_diffuse_delta1_verify(CELL_DT* d_temp1, CELL_DT* d_temp2, 
                                        int Ngx, int Ngy, int Ngz, int b2r_i, 
                                        int xef, int yef, int zef, int b2r_R)
{
    __shared__ CELL_DT slice[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2]; 

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + 1;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + 1;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    int in_2d  = stride*b2r_i + iy*Ngx + ix;   
    int out_2d;                 // the output global index
    // update_flag is important, for those subdomain on the edge of the environment, 
    // edge (B2R)halo should not be updated
    bool update_flag = true;
    
    // left and right edge
    if(xef == 0){         // inside the environment, update all
        update_flag = update_flag && ix >= b2r_i+1 && ix <= (Ngx-2-b2r_i);
    }else{                // do not update edge
        if(xef == -1){    // leftmost, left part do not update
            update_flag = update_flag && ix >= b2r_R && ix <= (Ngx-2-b2r_i);
        }else{           // rightmost, right part do not update
            update_flag = update_flag && ix >= b2r_i+1 && ix <= (Ngx-1-b2r_R); // max of b2r_i is b2r_R-1
        }
    }
    // top and bottom
    if(yef == 0){        // inside the environment, update all
        update_flag = update_flag && iy >= 1+b2r_i && iy <= (Ngy-2-b2r_i);
    }else{
        if(yef == -1){  // topmost, top edge do not update
            update_flag = update_flag && iy >= b2r_R && iy <= (Ngy-2-b2r_i);
        }else{          // bottommost, bottom edge do not update
            update_flag = update_flag && iy >= 1+b2r_i && iy <= (Ngy-1-b2r_R);
        }
    }
    
    // front and behind
    int zid_st, zid_end;
    if(zef == 0){       // inside the environment, update all
        zid_st = b2r_i+1;
        zid_end = Ngz-2-b2r_i;
    }else{
        if(zef == -1){  // fore-most
            zid_st = b2r_R;
            zid_end = Ngz-2-b2r_i;         
        }else{          // rear-most
            zid_st = b2r_i+1;
            zid_end = Ngz-1-b2r_R;            
        }
    }
    
    CELL_DT behind;
    CELL_DT current = d_temp1[in_2d]; 
    out_2d = in_2d;            // current
    in_2d += stride;           // one layer ahead 
    CELL_DT infront = d_temp1[in_2d]; 
    in_2d += stride;           // two layers ahead 

    for(int i=b2r_i+1; i<=Ngz-2-b2r_i; i++)
    {
        behind = current;         // current layer i = 1 
        current= infront;
        infront= d_temp1[in_2d];  // 1 ahead (current = 1)
        in_2d += stride;
        out_2d += stride;
        __syncthreads();
        if (update_flag)
        {
            if(threadIdx.x == 0){ // Halo left
                slice[ty][tx-1] = d_temp1[out_2d - 1];
            }
            if(threadIdx.x == CUDA_BLOCK_SIZE-1){ // Halo right
                slice[ty][tx+1] = d_temp1[out_2d + 1];
            }
            if(threadIdx.y == 0){ // Halo bottom
                slice[ty-1][tx] = d_temp1[out_2d - Ngx];
            }
            if(threadIdx.y == CUDA_BLOCK_SIZE-1){ // Halo top
                slice[ty+1][tx] = d_temp1[out_2d + Ngx];
            }
        }
        __syncthreads();
        slice[ty][tx] = current;
        __syncthreads();
        // also check z index, for foremost and rearmost subdomain, do not update the B2R's halo
        if (update_flag && i >= zid_st && i <= zid_end){
            d_temp2[out_2d]  = current + (Alpha*Dt)*(
                            (slice[ty][tx-1] - 2*current + slice[ty][tx+1])/(Dx*Dx) +
                            (slice[ty-1][tx] - 2*current + slice[ty+1][tx])/(Dy*Dy) +
                            (behind          - 2*current + infront)/(Dz*Dz));
        }
        __syncthreads();
    }
}
/*
***************************************************************************************************
* func   name: fdm_heat_diffuse_delta1
* description: cuda kernel function to update (2nd order accurancy) in Z direction with
               note: this is a general kernal for all the subdomain, i.e., the (B2R's)halo 
               of edge subdomain also will get updated
* parameters :
*             d_temp1: input
*             d_temp2: output
*             Ngx, Ngy, Ngz: subdomain dimension
*             b2r_i: loop id in subdomain
* return: none
***************************************************************************************************
*/
__global__ void fdm_heat_diffuse_delta1(CELL_DT* d_temp1, CELL_DT* d_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    __shared__ CELL_DT slice[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix;   
    int out_2d;              
    bool update_flag = ix >= b2r_i+1 && ix <= (Ngx-2-b2r_i) && iy >= 1+b2r_i && iy <= (Ngy-2-b2r_i);

    CELL_DT behind;
    CELL_DT current = d_temp1[in_2d]; 
    out_2d = in_2d;             // current
    in_2d += stride;            // one layer ahead 
    CELL_DT infront = d_temp1[in_2d]; 
    in_2d += stride;            // two layers ahead 

    for(int i=b2r_i+1; i<=Ngz-2-b2r_i; i++)
    {
        behind = current;       // current layer i = 1 
        current= infront;
        infront= d_temp1[in_2d];// 1 ahead (current = 1)
        in_2d += stride;
        out_2d += stride;
        __syncthreads();
        // let the thread on the edge also load for halo
        if (update_flag)
        {
            if(threadIdx.x == 0){ // Halo left
                slice[ty][tx-1] = d_temp1[out_2d - 1];
            }
            if(threadIdx.x == CUDA_BLOCK_SIZE-1){ // Halo right
                slice[ty][tx+1] = d_temp1[out_2d + 1];
            }
            if(threadIdx.y == 0){ // Halo bottom
                slice[ty-1][tx] = d_temp1[out_2d - Ngx];
            }
            if(threadIdx.y == CUDA_BLOCK_SIZE-1){ // Halo top
                slice[ty+1][tx] = d_temp1[out_2d + Ngx];
            }
        }
        __syncthreads();
        slice[ty][tx] = current;
        __syncthreads();
        if (update_flag){
            d_temp2[out_2d]  = current + (Alpha*Dt)*(
                            (slice[ty][tx-1] - 2*current + slice[ty][tx+1])/(Dx*Dx) +
                            (slice[ty-1][tx] - 2*current + slice[ty+1][tx])/(Dy*Dy) +
                            (behind          - 2*current + infront)/(Dz*Dz));
        }
        __syncthreads();
    }
}
/*
*************************************************************************************************
* func   name: fdm_heat_diffuse_delta2
* description: cuda kernel function to update (4th order) in Z direction
               note: this is a general kernal for all the subdomain, i.e., the (B2R's)halo 
               of edge subdomain also will get updated
* parameters :
*             d_temp1: input
*             d_temp2: output
*             Ngx, Ngy, Ngz: subdomain dimension
*             b2r_i: loop id in subdomain
* return: none
*************************************************************************************************
*/
__global__ void fdm_heat_diffuse_delta2(CELL_DT* d_temp1, CELL_DT* d_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    __shared__ CELL_DT slice[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix; 
    int out_2d;
    bool update_flag = ix >= B2R_D*(b2r_i+1) && ix <= Ngx-1-(1+b2r_i)*B2R_D && iy >= B2R_D*(1+b2r_i) && iy <= Ngy-1-(1+b2r_i)*B2R_D;

    CELL_DT behind2, behind1, current, infront1, infront2;

    behind1 = d_temp1[in_2d];
    in_2d += stride;

    current = d_temp1[in_2d];
    out_2d = in_2d;
    in_2d += stride;

    infront1 = d_temp1[in_2d];
    in_2d += stride;

    infront2 = d_temp1[in_2d];
    in_2d += stride;

    for(int i=B2R_D*(b2r_i+1); i<Ngz-(1+b2r_i)*B2R_D; i++)
    {
        behind2  = behind1;
        behind1  = current;         // current layer i = 1 
        current  = infront1;
        infront1 = infront2;
        infront2 = d_temp1[in_2d];  // 1 ahead (current = 1)

        in_2d += stride;
        out_2d += stride;
        __syncthreads();
        if (update_flag)
        {
            if(threadIdx.x < B2R_D){ // Halo left
                slice[ty][tx-B2R_D] = d_temp1[out_2d - B2R_D];
            }
            if(threadIdx.x >= CUDA_BLOCK_SIZE-B2R_D){ // Halo right
                slice[ty][tx+B2R_D] = d_temp1[out_2d + B2R_D];
            }
            if(threadIdx.y < B2R_D){ // Halo bottom
                slice[ty-B2R_D][tx] = d_temp1[out_2d - Ngx];
            }
            if(threadIdx.y >= CUDA_BLOCK_SIZE-B2R_D){ // Halo top
                slice[ty+B2R_D][tx] = d_temp1[out_2d + Ngx];
            }
        }
        __syncthreads();
        slice[ty][tx] = current;
        __syncthreads();
        if (update_flag){ // coef: −1/12 4/3 −5/2 4/3 −1/12
            d_temp2[out_2d]  = current + (Alpha*Dt)*(
                            (coef_d2_a*(slice[ty][tx-2]+slice[ty][tx+2]) + coef_d2_b*(slice[ty][tx-1]+slice[ty][tx+1]) + coef_d2_c*current)/(Dx*Dx) +
                            (coef_d2_a*(slice[ty-2][tx]+slice[ty+2][tx]) + coef_d2_b*(slice[ty-1][tx]+slice[ty+1][tx]) + coef_d2_c*current)/(Dy*Dy) +
                            (coef_d2_a*(behind2 + infront2)              + coef_d2_b*(behind1 + infront1)              + coef_d2_c*current)/(Dz*Dz));
        }
        __syncthreads();
    }
}
/*
*************************************************************************************************
* func   name: fdm_heat_diffuse_delta3
* description: cuda kernel function to update (6th order) in Z direction
               note: this is a general kernal for all the subdomain, i.e., the (B2R's)halo 
               of edge subdomain also will get updated
* parameters :
*             d_temp1: input
*             d_temp2: output
*             Ngx, Ngy, Ngz: subdomain dimension
*             b2r_i: loop id in subdomain
* return: none
*************************************************************************************************
*/
__global__ void fdm_heat_diffuse_delta3(CELL_DT* d_temp1, CELL_DT* d_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    __shared__ CELL_DT slice[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix;  
    int out_2d; 
    bool update_flag = ix >= B2R_D*(b2r_i+1) && ix <= Ngx-1-(1+b2r_i)*B2R_D && iy >= B2R_D*(1+b2r_i) && iy <= Ngy-1-(1+b2r_i)*B2R_D;

    CELL_DT behind3, behind2, behind1, current, infront1, infront2, infront3;
    
    behind2 = d_temp1[in_2d]; 
    in_2d += stride;

    behind1 = d_temp1[in_2d];
    in_2d += stride;

    current = d_temp1[in_2d];
    out_2d = in_2d;
    in_2d += stride;

    infront1 = d_temp1[in_2d];
    in_2d += stride;

    infront2 = d_temp1[in_2d];
    in_2d += stride;

    infront3 = d_temp1[in_2d];
    in_2d += stride;
    
    for(int i=B2R_D*(b2r_i+1); i<Ngz-(1+b2r_i)*B2R_D; i++)
    {
        behind3  = behind2;
        behind2  = behind1;
        behind1  = current;         // current layer i = 1 
        current  = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = d_temp1[in_2d];  // 1 ahead (current = 1)

        in_2d += stride;
        out_2d += stride;
        __syncthreads();
        if (update_flag)
        {
            if(threadIdx.x < B2R_D){ // Halo left
                slice[ty][tx-B2R_D] = d_temp1[out_2d - B2R_D];
            }
            if(threadIdx.x >= CUDA_BLOCK_SIZE-B2R_D){ // Halo right
                slice[ty][tx+B2R_D] = d_temp1[out_2d + B2R_D];
            }
            if(threadIdx.y < B2R_D){ // Halo top
                slice[ty-B2R_D][tx] = d_temp1[out_2d - Ngx];
            }
            if(threadIdx.y >= CUDA_BLOCK_SIZE-B2R_D){ // Halo down
                slice[ty+B2R_D][tx] = d_temp1[out_2d + Ngx];
            }
        }
        __syncthreads();
        slice[ty][tx] = current;
        __syncthreads();
        if (update_flag){ 
            d_temp2[out_2d]  = current + (Alpha*Dt)*(
                            (coef_d3_a*(slice[ty][tx-3]+slice[ty][tx+3]) + coef_d3_b*(slice[ty][tx-2]+slice[ty][tx+2]) + coef_d3_c*(slice[ty][tx-1]+slice[ty][tx+1]) + coef_d3_d*current)/(Dx*Dx) +
                            (coef_d3_a*(slice[ty-3][tx]+slice[ty+3][tx]) + coef_d3_b*(slice[ty-2][tx]+slice[ty+2][tx]) + coef_d3_c*(slice[ty-1][tx]+slice[ty+1][tx]) + coef_d3_d*current)/(Dy*Dy) +
                            (coef_d3_a*(behind3 + infront3) + coef_d3_b*(behind2 + infront2) + coef_d3_c*(behind1 + infront1) + coef_d3_d*current)/(Dz*Dz));
        }
        __syncthreads();
    }
}
/*
*************************************************************************************************
* func   name: fdm_heat_diffuse_delta4
* description: cuda kernel function to update (8th order) in Z direction
               note: this is a general kernal for all the subdomain, i.e., the (B2R's)halo 
               of edge subdomain also will get updated
* parameters :
*             d_temp1: input
*             d_temp2: output
*             Ngx, Ngy, Ngz: subdomain dimension
*             b2r_i: loop id in subdomain
* return: none
*************************************************************************************************
*/
__global__ void fdm_heat_diffuse_delta4(CELL_DT* d_temp1, CELL_DT* d_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    __shared__ CELL_DT slice[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix; 
    int out_2d;                     // represent one layer behind (z-direction)
    // although the outmost B2R_D do not need update, the thread should load their data from device memory to shared memory
    bool update_flag = ix >= B2R_D*(b2r_i+1) && ix <= Ngx-1-(1+b2r_i)*B2R_D && iy >= B2R_D*(1+b2r_i) && iy <= Ngy-1-(1+b2r_i)*B2R_D;

    CELL_DT behind4, behind3, behind2, behind1, current, infront1, infront2, infront3, infront4;
    
    behind3 = d_temp1[in_2d]; 
    in_2d += stride;
    
    behind2 = d_temp1[in_2d]; 
    in_2d += stride;

    behind1 = d_temp1[in_2d];
    in_2d += stride;

    current = d_temp1[in_2d];
    out_2d = in_2d;
    in_2d += stride;

    infront1 = d_temp1[in_2d];
    in_2d += stride;

    infront2 = d_temp1[in_2d];
    in_2d += stride;

    infront3 = d_temp1[in_2d];
    in_2d += stride;

    infront4 = d_temp1[in_2d];
    in_2d += stride;
        
    for(int i=B2R_D*(b2r_i+1); i<Ngz-(1+b2r_i)*B2R_D; i++)
    {
        behind4  = behind3;
        behind3  = behind2;
        behind2  = behind1;
        behind1  = current;         // current layer i = 1 
        current  = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = d_temp1[in_2d];  // 1 ahead (current = 1)

        in_2d += stride;
        out_2d += stride;
        __syncthreads();
        if (update_flag)
        {
            if(threadIdx.x < B2R_D){ // Halo left
                slice[ty][tx-B2R_D] = d_temp1[out_2d - B2R_D];
            }
            if(threadIdx.x >= CUDA_BLOCK_SIZE-B2R_D){ // Halo right
                slice[ty][tx+B2R_D] = d_temp1[out_2d + B2R_D];
            }
            if(threadIdx.y < B2R_D){ // Halo top
                slice[ty-B2R_D][tx] = d_temp1[out_2d - Ngx];
            }
            if(threadIdx.y >= CUDA_BLOCK_SIZE-B2R_D){ // Halo down
                slice[ty+B2R_D][tx] = d_temp1[out_2d + Ngx];
            }
        }
        __syncthreads();
        slice[ty][tx] = current;
        __syncthreads();
        if (update_flag){ 
            d_temp2[out_2d]  = current + (Alpha*Dt)*(
                            (coef_d4_a*(slice[ty][tx-4]+slice[ty][tx+4]) + coef_d4_b*(slice[ty][tx-3]+slice[ty][tx+3]) + coef_d4_c*(slice[ty][tx-2]+slice[ty][tx+2]) + coef_d4_d*(slice[ty][tx-1]+slice[ty][tx+1]) + coef_d4_e*current)/(Dx*Dx) +
                            (coef_d4_a*(slice[ty-4][tx]+slice[ty+4][tx]) + coef_d4_b*(slice[ty-3][tx]+slice[ty+3][tx]) + coef_d4_c*(slice[ty-2][tx]+slice[ty+2][tx]) + coef_d4_d*(slice[ty-1][tx]+slice[ty+1][tx]) + coef_d4_e*current)/(Dy*Dy) +
                            (coef_d4_a*(behind4 + infront4)              + coef_d4_b*(behind3 + infront3)              + coef_d4_c*(behind2 + infront2)              + coef_d4_d*(behind1 + infront1)              + coef_d4_e*current)/(Dz*Dz));
        }
        __syncthreads();
    }
}
/*
*************************************************************************************************
* func   name: heat_3D_gpu_init
* description: initialize cuda related variable/environment, this function should be called 
               in model initialization
* parameters :
*             in_grid_size: number of cells in the subdomain grid
* return: none
*************************************************************************************************
*/
void heat_3D_gpu_init(int in_grid_size)
{
    cudaErrchk( cudaMalloc((void**)&d_temp1, in_grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_temp2, in_grid_size) );
}
/*
*************************************************************************************************
* func   name: heat_3D_gpu_finalize
* description: release allocated resource in cuda runtime, 
               this function should be called in model finalize func.
* parameters :
*             none
* return: none
*************************************************************************************************
*/
void heat_3D_gpu_finalize()
{
    cudaFree(d_temp1);
    cudaFree(d_temp2);
}
/*
*************************************************************************************************
* func   name: pinned_host_memAlloc
* description: allocate pinned host memory
* parameters :
*             none
* return: none
*************************************************************************************************
*/
CELL_DT *pinned_host_memAlloc(long byte_size){
    CELL_DT * phost;
    cudaErrchk( cudaMallocHost((void**)&phost, byte_size) );
    return phost;
}
/*
*************************************************************************************************
* func   name: heat_3D_gpu_main
* description: main entry of the model implementation, this function will be called from B2R 
               every R simulation time steps.
* parameters :
*             h_in  : input buffer
*             b2r_R : R of the B2R framework
*             Ngx, Ngy, Ngz
*             edge_flag: indicate if the location of the subdomain, 
*                   -1 -> leftmost,topmost,foremost), 
*                    0 -> inside, 
*                    1 -> right, bottommost, rearmost
*             
* return: none
*************************************************************************************************
*/
double heat_3D_gpu_main(CELL_DT * h_in, int b2r_R, char *ef, int Ngx, int Ngy, int Ngz)
{
    int in_grid_size = sizeof(CELL_DT)*Ngx*Ngy*Ngz;
    // TODO: figure out why global malloc and free failed sometimes!
    heat_3D_gpu_init(in_grid_size);
    CELL_DT *pd_temp;

    cudaErrchk( cudaMemcpy((void *)d_temp1, (void *)h_in, in_grid_size, cudaMemcpyHostToDevice) );
    // copy to d_temp2 is also needed because the hola (our surface will not be updated, while d_temp2 will be the input for the next step)
    cudaErrchk( cudaMemcpy((void *)d_temp2, (void *)d_temp1, in_grid_size, cudaMemcpyDeviceToDevice) );
//    struct timeval compu_s, compu_e;
//    gettimeofday(&compu_s, NULL);  // set as the start of computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Launch configuration:
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    for (int i = 0; i < b2r_R; i++)
    {
        switch(B2R_D){
            case 1:
                //fdm_heat_diffuse_delta1_verify<<<dimGrid, dimBlock>>>(d_temp1, d_temp2, Ngx, Ngy, Ngz, i, ef[0], ef[1], ef[2], b2r_R);
                fdm_heat_diffuse_delta1<<<dimGrid, dimBlock>>>(d_temp1, d_temp2, Ngx, Ngy, Ngz, i);            
                break;
            case 2:
                fdm_heat_diffuse_delta2<<<dimGrid, dimBlock>>>(d_temp1, d_temp2, Ngx, Ngy, Ngz, i);
                break;
            case 3:
                fdm_heat_diffuse_delta3<<<dimGrid, dimBlock>>>(d_temp1, d_temp2, Ngx, Ngy, Ngz, i);
                break;
            case 4:
                fdm_heat_diffuse_delta4<<<dimGrid, dimBlock>>>(d_temp1, d_temp2, Ngx, Ngy, Ngz, i);
                break;
            default:       
                cout << "Error, unsupported delta(B2R_D)" << endl;
                exit(-1);                                             
        }  
        
        cudaErrchk( cudaThreadSynchronize() );

        pd_temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = pd_temp;
    }
//    gettimeofday(&compu_e, NULL);  // computation finished
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Copy from device to host, (the first R*D layer did not get updated, no need copy)
    cudaErrchk( cudaMemcpy((void*) (h_in    + b2r_R*B2R_D*Ngx*Ngy) ,
                           (void*) (d_temp1 + b2r_R*B2R_D*Ngx*Ngy), sizeof(CELL_DT)*Ngx*Ngy*(Ngz-2*b2r_R*B2R_D), 
                           cudaMemcpyDeviceToHost) );

    heat_3D_gpu_finalize();
    //return compu_e.tv_sec - compu_s.tv_sec + (compu_e.tv_usec - compu_s.tv_usec) / 1e6;
    float gpu_compu_elapsed_time_ms;
    cudaEventElapsedTime(&gpu_compu_elapsed_time_ms, start, stop);
    return (double)gpu_compu_elapsed_time_ms / 1000.0;
}

