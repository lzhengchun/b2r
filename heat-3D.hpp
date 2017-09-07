//
//  heat-3D.hpp

#ifndef heat_3D_h
#define heat_3D_h
#include "../b2r.h"
#include <fstream>
#include <mpi.h>

using namespace std;
double heat_3D_gpu_main(CELL_DT * h_in, int b2r_R, char *edge_flag, int Ngx, int Ngy, int Ngz);
float heat_3D_cpu_main(CELL_DT * h_in, int b2r_R, int b2r_D, int Ngx, int Ngy, int Ngz);
void heat_3D_gpu_init(int in_grid_size);
void heat_3D_gpu_finalize();
double heat_3D_gpu_diffsync_main(int b2r_R, char *ef, int Ngx, int Ngy, int Ngz, char *sync_flag, char **recv_buf, char **send_buf);
CELL_DT *pinned_host_memAlloc(long byte_size);
/*
 *********************************************************************
 * func   name: b2r_block_init
 * description: this function initlizes all the cells mostly for
                sync testing/verification.
 * parameters :
 *             none
 * return: none
 *********************************************************************
 */
void b2r_block_init(int rank)
{
    for (int z = 0; z < B2R_BLOCK_SIZE_Z; ++z)
        for(int r = 0; r < B2R_BLOCK_SIZE_Y; r++)
        {
            for(int c = 0; c < B2R_BLOCK_SIZE_X; c++)
            {
                int global_id = z * B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y + r*B2R_BLOCK_SIZE_X + c;
                b2r_env_grid[global_id] = 10.0;
            }
        }
}

/*
 *********************************************************************
 * func   name: model_init
 * description: this function initializes cells according to model
                requirements, this is to initialize from point view
                of the model, no B2R issue needs to be considered.
 * parameters :
 *             none
 * return     : none
 *********************************************************************
 */
void model_init(int rank)
{
    b2r_block_init(rank);    // initialize hola for testing purpose
    int Rd = B2R_R*B2R_D;
    
    for (int z = Rd; z < B2R_BLOCK_SIZE_Z-Rd; ++z)
        for(int r = Rd; r < B2R_BLOCK_SIZE_Y-Rd ; r++)
        {
            for(int c = Rd; c < B2R_BLOCK_SIZE_X-Rd; c++)
            {
                int global_id = z * B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y + r*B2R_BLOCK_SIZE_X + c;
                b2r_env_grid[global_id] = 20.0;
            }
        }
    //heat_3D_gpu_init(item_size*B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*B2R_BLOCK_SIZE_Z);
}

/*
 *********************************************************************
 * func   name: print_block_data
 * description: this function prints / output results in local blocks,
                for results verification
 * parameters :
 *             none
 * return: none
 *********************************************************************
 */
void print_block_data()
{
    char buf[20];
    int global_id;
    int Rd = B2R_R*B2R_D;
    for (int z = Rd; z < B2R_BLOCK_SIZE_Z-Rd; ++z)
    {
        // cout << "slice: [" << z << ", :, :]" << endl;
        for(int r = Rd; r < B2R_BLOCK_SIZE_Y-Rd; r++)
        {
            for(int c = Rd; c < B2R_BLOCK_SIZE_X-Rd; c++)
            {
                global_id = z * B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y + r*B2R_BLOCK_SIZE_X + c;
                //cout << (int)b2r_env_grid[r][c] << " ";
#if _ENV_3D_
                //                snprintf (buf, sizeof(buf), "(%02d,%02d,%02d)", b2r_env_grid[global_id].x, b2r_env_grid[global_id].y, b2r_env_grid[global_id].z);
                snprintf (buf, sizeof(buf), "%0.2f", b2r_env_grid[global_id]);
#else
                snprintf (buf, sizeof(buf), "(%02d,%02d)", b2r_env_grid[global_id].x, b2r_env_grid[global_id].y);
#endif
                cout << buf << ",";
            }
            //cout << endl;
        }
    }
}

void print_block_data_step(int rank, float step){
    ofstream output_file;
    char filename[100];
    sprintf( filename, "R-%d-rank-%d-i-%0.1f.txt", B2R_R, rank, step);
    output_file.open(filename);
    char buf[20];
    int global_id;
    for (int z = 0; z < B2R_BLOCK_SIZE_Z; ++z)
    {
        output_file << "slice: [" << z << ", :, :]" << endl;
        for(int r = 0; r < B2R_BLOCK_SIZE_Y; r++)
        {
            for(int c = 0; c < B2R_BLOCK_SIZE_X; c++)
            {
                global_id = z * B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y + r*B2R_BLOCK_SIZE_X + c;
                snprintf (buf, sizeof(buf), "%0.2f", b2r_env_grid[global_id]);
                output_file << buf << ",";
            }
            output_file << endl;
        }
    }    
    output_file.close();
}

/*
 *********************************************************************
 * func   name: model_finalize
 * description:   release some model scopic resources
 * parameters :
 *             none
 * return: none
 *********************************************************************
 */
void model_finalize()
{
   // heat_3D_gpu_finalize();
}

/*
 *********************************************************************
 * func   name: print_sent_data
 * description: mostly for debuging, this function prints / output,
                packed data in sending buffer
 * parameters :
 *             none
 * return: none
 *********************************************************************
 */
/*
void print_sent_data()
{
    char buf[10] = "";
    int B2R_RD = (B2R_D*B2R_R), global_id;
    int sub_matrix_size_r[10] = {B2R_RD, B2R_RD, B2R_RD, B2R_B_Y , B2R_B_Y , B2R_RD, B2R_RD, B2R_RD, B2R_B_Y, B2R_B_Y};
    int sub_matrix_size_c[10] = {B2R_RD, B2R_B_X , B2R_RD, B2R_RD, B2R_RD, B2R_RD, B2R_B_X , B2R_RD, B2R_B_X, B2R_B_X};
    
    CELL_DT* pdata;
    for (int m = 0; m < 10; ++m){
        pdata = (CELL_DT*)pad_send_buf[m];
        cout << "sent sub_matrix: " << m << endl;
        for (int z = 0; z < B2R_B_Z; ++z)
        {
            if(m > 7 && z >= B2R_R*B2R_D)
                break;
            cout << "slice: [" << z << ", :, :]" << endl;
            for(int r = 0; r < sub_matrix_size_r[m]; r++)
            {
                for(int c = 0; c < sub_matrix_size_c[m]; c++)
                {
                    global_id = z * sub_matrix_size_r[m]*sub_matrix_size_c[m] + r*sub_matrix_size_c[m] + c;
                    //cout << (int)b2r_env_grid[r][c] << " ";
                    snprintf (buf, 10, "%f", pdata[global_id]);
                    cout << buf << " ";
                }
                cout << endl;
            }
        }
    }
}
 */
/*
 *********************************************************************
 * func   name: print_recv_data
 * description: mostly for debuging, this function prints / output,
                packed data in sending buffer
 * parameters:
 *             none
 * return: none
 *********************************************************************
 */
/*
void print_recv_data()
{
    char buf[10] = "";
    int B2R_RD = (B2R_D*B2R_R), global_id;
    int sub_matrix_size_r[10] = {B2R_RD, B2R_RD, B2R_RD, B2R_B_Y , B2R_B_Y , B2R_RD, B2R_RD, B2R_RD, B2R_B_Y, B2R_B_Y};
    int sub_matrix_size_c[10] = {B2R_RD, B2R_B_X , B2R_RD, B2R_RD, B2R_RD, B2R_RD, B2R_B_X , B2R_RD, B2R_B_X, B2R_B_X};
    
    CELL_DT* pdata;
    for (int m = 10; m < 18; ++m){
        pdata = (CELL_DT*)pad_recv_buf[m];
        cout << "recv sub_matrix: " << m << endl;
        for (int z = 0; z < B2R_RD; ++z)
        {
            cout << "slice: [" << z << ", :, :]" << endl;
            for(int r = 0; r < sub_matrix_size_r[m-10]; r++)
            {
                for(int c = 0; c < sub_matrix_size_c[m-10]; c++)
                {
                    global_id = z * sub_matrix_size_r[m-10]*sub_matrix_size_c[m-10] + r*sub_matrix_size_c[m-10] + c;
                    //cout << (int)b2r_env_grid[r][c] << " ";
                    snprintf (buf, 10, "%f", pdata[global_id]);
                    cout << buf << " ";
                }
                cout << endl;
            }
        }
    }
}
*/
#endif
