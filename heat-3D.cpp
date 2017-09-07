//
//  heat-3D.cpp
//  b2r
//
//  Created by Zhengchun Liu on 1/22/16.
//  Copyright © 2016 research. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include "../b2r_config.h"
typedef float CELL_DT;

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

using namespace std;

inline int gg_id(int x, int y, int z, int Ngx, int Ngy)
{
    return z*(Ngx*Ngy) + y*Ngx + x;
}

//from back to front, to avoid confliction with GPU in joint layer, i.e., avoid reading updated data
void cpu_fdm_heat_diffuse1(float* h_temp1, float* h_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
   // cout << "thread: " << omp_get_thread_num() << " doing for Ngz = " << Ngz << " from " << (long)h_temp1 << endl;
    #pragma omp parallel for num_threads(CPU_N_THRD)
    for (int ix=b2r_i+1; ix <= Ngx-2-b2r_i; ix++)
        for (int iy=b2r_i+1; iy <= Ngy-2-b2r_i; iy++)
            //for (int iz=b2r_i+1; iz <= Ngz-2-b2r_i; iz++)
            for (int iz=Ngz-2-b2r_i; iz >= b2r_i+1; iz--)
            {
                h_temp2[gg_id(ix, iy, iz, Ngx, Ngy)] = h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)] + (Alpha*Dt)*(
                (h_temp1[gg_id(ix-1, iy, iz, Ngx, Ngy)] - 2*h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+1, iy, iz, Ngx, Ngy)]) / (Dx*Dx) +
                (h_temp1[gg_id(ix, iy-1, iz, Ngx, Ngy)] - 2*h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+1, iz, Ngx, Ngy)]) / (Dy*Dy) +
                (h_temp1[gg_id(ix, iy, iz-1, Ngx, Ngy)] - 2*h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+1, Ngx, Ngy)]) / (Dz*Dz));
            }
}

void cpu_fdm_heat_diffuse2(float* h_temp1, float* h_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    #pragma omp parallel for num_threads(CPU_N_THRD)
    for (int ix=b2r_i*2+1; ix <= Ngx-2-b2r_i*2; ix++)
        for (int iy=b2r_i*2+1; iy <= Ngy-2-b2r_i*2; iy++)
            //for (int iz=b2r_i*2+1; iz <= Ngz-2-b2r_i*2; iz++)
            for (int iz=Ngz-2-b2r_i*2; iz >= b2r_i*2+1; iz--)
            {
                h_temp2[gg_id(ix, iy, iz, Ngx, Ngy)] =  h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)] + (Alpha*Dt)*(
                            ( coef_d2_a * (h_temp1[gg_id(ix-2, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+2, iy, iz, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix-1, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+1, iy, iz, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dx*Dx) + 
                            ( coef_d2_a * (h_temp1[gg_id(ix, iy-2, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+2, iz, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix, iy-1, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+1, iz, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dy*Dy) + 
                            ( coef_d2_a * (h_temp1[gg_id(ix, iy, iz-2, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+2, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix, iy, iz-1, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+1, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dz*Dz));
            }
}

void cpu_fdm_heat_diffuse3(float* h_temp1, float* h_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    #pragma omp parallel for num_threads(CPU_N_THRD)  
    for (int ix=b2r_i*3+1; ix <= Ngx-2-b2r_i*3; ix++)
        for (int iy=b2r_i*3+1; iy <= Ngy-2-b2r_i*3; iy++)
            //for (int iz=b2r_i*3+1; iz <= Ngz-2-b2r_i*3; iz++)
            for (int iz=Ngz-2-b2r_i*3; iz >= b2r_i*3+1; iz--)
            {
                h_temp2[gg_id(ix, iy, iz, Ngx, Ngy)] =  h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)] + (Alpha*Dt)*(
                            ( coef_d2_a * (h_temp1[gg_id(ix-3, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+3, iy, iz, Ngx, Ngy)]) +
                              coef_d2_a * (h_temp1[gg_id(ix-2, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+2, iy, iz, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix-1, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+1, iy, iz, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dx*Dx) + 
                            ( coef_d2_a * (h_temp1[gg_id(ix, iy-3, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+3, iz, Ngx, Ngy)]) + 
                              coef_d2_a * (h_temp1[gg_id(ix, iy-2, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+2, iz, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix, iy-1, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+1, iz, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dy*Dy) + 
                            ( coef_d2_a * (h_temp1[gg_id(ix, iy, iz-3, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+3, Ngx, Ngy)]) + 
                              coef_d2_a * (h_temp1[gg_id(ix, iy, iz-2, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+2, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix, iy, iz-1, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+1, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dz*Dz));
            }
}

void cpu_fdm_heat_diffuse4(float* h_temp1, float* h_temp2, int Ngx, int Ngy, int Ngz, int b2r_i)
{
    #pragma omp parallel for num_threads(CPU_N_THRD)  
    for (int ix=b2r_i*4+1; ix <= Ngx-2-b2r_i*4; ix++)
        for (int iy=b2r_i*4+1; iy <= Ngy-2-b2r_i*4; iy++)
            //for (int iz=b2r_i*4+1; iz <= Ngz-2-b2r_i*4; iz++)
            for (int iz=Ngz-2-b2r_i*4; iz >= b2r_i*4+1; iz--)
            {
                h_temp2[gg_id(ix, iy, iz, Ngx, Ngy)] =  h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)] + (Alpha*Dt)*(
                            ( coef_d2_a * (h_temp1[gg_id(ix-4, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+4, iy, iz, Ngx, Ngy)]) +
                              coef_d2_a * (h_temp1[gg_id(ix-3, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+3, iy, iz, Ngx, Ngy)]) +
                              coef_d2_a * (h_temp1[gg_id(ix-2, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+2, iy, iz, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix-1, iy, iz, Ngx, Ngy)] + h_temp1[gg_id(ix+1, iy, iz, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dx*Dx) + 
                            ( coef_d2_a * (h_temp1[gg_id(ix, iy-4, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+4, iz, Ngx, Ngy)]) + 
                              coef_d2_a * (h_temp1[gg_id(ix, iy-3, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+3, iz, Ngx, Ngy)]) + 
                              coef_d2_a * (h_temp1[gg_id(ix, iy-2, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+2, iz, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix, iy-1, iz, Ngx, Ngy)] + h_temp1[gg_id(ix, iy+1, iz, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dy*Dy) + 
                            ( coef_d2_a * (h_temp1[gg_id(ix, iy, iz-4, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+4, Ngx, Ngy)]) + 
                              coef_d2_a * (h_temp1[gg_id(ix, iy, iz-3, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+3, Ngx, Ngy)]) + 
                              coef_d2_a * (h_temp1[gg_id(ix, iy, iz-2, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+2, Ngx, Ngy)]) + 
                              coef_d2_b * (h_temp1[gg_id(ix, iy, iz-1, Ngx, Ngy)] + h_temp1[gg_id(ix, iy, iz+1, Ngx, Ngy)]) + 
                              coef_d2_c * (h_temp1[gg_id(ix, iy, iz, Ngx, Ngy)]) ) / (Dz*Dz));
            }
}

float heat_3D_cpu_main(CELL_DT * h_in, int b2r_R, int b2r_D, int Ngx, int Ngy, int Ngz)
{
    struct timeval cpu_s, cpu_e;
    float elapsedtime;
    gettimeofday(&cpu_s, NULL); 
    
    CELL_DT *p_in, *p_out, *p_temp;
    p_in = h_in;
    p_out = new CELL_DT[Ngx * Ngy * Ngz];
    memcpy(p_out, h_in, sizeof(CELL_DT)*Ngx*Ngy*Ngz);
    
    for (int i = 0; i < b2r_R; i++){
        switch(b2r_D){
            case 1:
                cpu_fdm_heat_diffuse1(p_in, p_out, Ngx, Ngy, Ngz, i);
                break;
            case 2:
                cpu_fdm_heat_diffuse2(p_in, p_out, Ngx, Ngy, Ngz, i);
                break;
            case 3:
                cpu_fdm_heat_diffuse3(p_in, p_out, Ngx, Ngy, Ngz, i);
                break;
            case 4:
                cpu_fdm_heat_diffuse4(p_in, p_out, Ngx, Ngy, Ngz, i);
                break;
            default:
                cout << "unsupported delta" << endl;
                exit(-1);                                                                
        }
        
        p_temp = p_in;
        p_in = p_out;
        p_out = p_temp;
    }
    // the last slice did not get updated, and will be updated by GPU
     
    if(b2r_R%2){
        memcpy(h_in + b2r_R*B2R_D*Ngx*Ngy, 
               p_in + b2r_R*B2R_D*Ngx*Ngy, 
               sizeof(CELL_DT)*Ngx*Ngy*(Ngz-2*b2r_R*B2R_D) );
        delete p_in;
    }else{
        delete p_out;
    }
    
    gettimeofday(&cpu_e, NULL);      
    elapsedtime = cpu_e.tv_sec - cpu_s.tv_sec + (float)(cpu_e.tv_usec - cpu_s.tv_usec) / 1.0e6;
    return elapsedtime;    
}



