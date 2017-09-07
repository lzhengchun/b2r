#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

typedef unsigned char CELL_DT;

#define CUDA_BLOCK_SIZE    32

using namespace std;

#define cudaErrchk(ans)  cudaAssert((ans), __FILE__, __LINE__) 
inline void cudaAssert(cudaError_t code, string file, int line){
    if (code != cudaSuccess){
        cerr << "CUDA Error: " << cudaGetErrorString(code) << "; file: " << file << ", line:" << line << endl;
        exit(-1);
    }
}
/*
*********************************************************************
* func   name: gol_live_neighbor_cnt
* description: Add up all live neighbors and return the number of
*              live neighbors
* parameters :
*             none
* return: none
*********************************************************************
*/

__global__ void gol_3d_kernel(CELL_DT* gol_grid_in, CELL_DT* gol_grid_out, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    __shared__ CELL_DT  behind[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2]; 
    __shared__ CELL_DT current[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2]; 
    __shared__ CELL_DT infront[CUDA_BLOCK_SIZE+2][CUDA_BLOCK_SIZE+2]; 

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i || iy >= Ngy-b2r_i || ix < b2r_i || iy < b2r_i)
    {
        return;
    }

    int tx = threadIdx.x + 1;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + 1;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    int in_2d  = stride*b2r_i + iy*Ngx + ix;   
    int out_2d;              
    int live_cnt;
    bool update_flag = ix >= b2r_i+1 && ix <= (Ngx-2-b2r_i) && iy >= 1+b2r_i && iy <= (Ngy-2-b2r_i);

    current[ty][tx] = gol_grid_in[in_2d]; 
    out_2d = in_2d;            // current
    in_2d += stride;           // one layer ahead 
    infront[ty][tx] = gol_grid_in[in_2d]; 
    in_2d += stride;           // two layers ahead 

    for(int i=b2r_i+1; i<=Ngz-2-b2r_i; i++)
    {
        behind[ty][tx] = current[ty][tx];    
        current[ty][tx]= infront[ty][tx];
        infront[ty][tx]= gol_grid_in[in_2d];
        in_2d += stride;
        out_2d += stride;
        __syncthreads();
        if (update_flag)
        {
            if(threadIdx.x == 0){ // Halo left
                current[ty][tx-1] = gol_grid_in[out_2d - 1];
            }
            if(threadIdx.x == CUDA_BLOCK_SIZE-1){ // Halo right
                current[ty][tx+1] = gol_grid_in[out_2d + 1];
            }
            if(threadIdx.y == 0){ // Halo bottom
                current[ty-1][tx] = gol_grid_in[out_2d - Ngx];
            }
            if(threadIdx.y == CUDA_BLOCK_SIZE-1){ // Halo top
                current[ty+1][tx] = gol_grid_in[out_2d + Ngx];
            }
        }
        __syncthreads();
        if (update_flag){ // the update_flag limitted edge, ±1 will not exceed border 
            live_cnt = infront[ty-1][tx-1] + infront[ty-1][tx] + infront[ty-1][tx+1] 
                     + infront[ty][tx-1]   + infront[ty][tx]   + infront[ty][tx+1] 
                     + infront[ty+1][tx-1] + infront[ty+1][tx] + infront[ty+1][tx+1]   
                     + current[ty-1][tx-1] + current[ty-1][tx] + current[ty-1][tx+1] 
                     + current[ty][tx-1]                       + current[ty][tx+1] 
                     + current[ty+1][tx-1] + current[ty+1][tx] + current[ty+1][tx+1]    
                     +  behind[ty-1][tx-1] + behind[ty-1][tx]  + behind[ty-1][tx+1] 
                     +  behind[ty][tx-1]   + behind[ty][tx]    + behind[ty][tx+1] 
                     +  behind[ty+1][tx-1] + behind[ty+1][tx]  + behind[ty+1][tx+1];                           
        }
        if(current[ty][tx] && live_cnt <= 1){                   // with only 1 or less neighbours die, as if by lonliness.
           gol_grid_out[out_2d] = 0;
        }else if(0 == current[ty][tx] && live_cnt == 5){        // If 5 cells surround an empty cell, they breed
           gol_grid_out[out_2d] = 1;
        }else if(current[ty][tx] && live_cnt >= 8){             // If a cell has 8 or more neighbours, it dies from overcrowding.
           gol_grid_out[out_2d] = 0; 
        }
        __syncthreads();
    }
}

/*
*********************************************************************
* func   name: gol_main
* description: update cells, rules include:
            (1) Cells (in this case, cubes) with only 1
            or less neighbours die, as if by lonliness.
            (2) If 5 cells surround an empty cell, they breed
            and fill it.
            (3) If a cell has 8 or more neighbours,
            it dies from overcrowding.
* parameters :
*             none
* return: none
*********************************************************************
*/
double gol_3D_gpu_main(CELL_DT * h_in, int b2r_R, char *ef, int Ngx, int Ngy, int Ngz)
{
    int in_grid_size = sizeof(CELL_DT)*Ngx*Ngy*Ngz;
    CELL_DT *d_temp1, *d_temp2;  
    CELL_DT *pd_temp;
    cudaErrchk( cudaMalloc((void**)&d_temp1, in_grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_temp2, in_grid_size) );
    cudaErrchk( cudaMemcpy((void *)d_temp1, (void *)h_in, in_grid_size, cudaMemcpyHostToDevice) );

    // copy to d_temp2 is also needed because the hola (our surface will not be updated, while d_temp2 will be the input for the next step)
    cudaErrchk( cudaMemcpy((void *)d_temp2, (void *)d_temp1, in_grid_size, cudaMemcpyDeviceToDevice) );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);                   // set as the start of computation
    // Launch configuration:
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    for (int i = 0; i < b2r_R; i++)
    {
        gol_3d_kernel<<<dimGrid, dimBlock>>>(d_temp1, d_temp2, Ngx, Ngy, Ngz, i);
        cudaErrchk( cudaThreadSynchronize() );
        pd_temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = pd_temp;
    }
    cudaEventRecord(stop, 0);                    // computation finished
    cudaEventSynchronize(stop);
    // Copy from device to host
    cudaErrchk( cudaMemcpy((void*) h_in, (void*) d_temp1, in_grid_size, cudaMemcpyDeviceToHost) );
    float gpu_compu_elapsed_time_ms;
    cudaEventElapsedTime(&gpu_compu_elapsed_time_ms, start, stop);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    return (double)gpu_compu_elapsed_time_ms / 1000.0;
}