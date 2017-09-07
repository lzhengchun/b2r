#ifndef _FDM3D_HPP_
#define _FDM3D_HPP_
#include <cmath>
using namespace std;

double gol_3D_gpu_main(CELL_DT * h_in, int b2r_R, char *edge_flag, int Ngx, int Ngy, int Ngz);

/*
*********************************************************************
* func   name: gol_model_init
* description: this function initializes cells according to model
               requirements, this is to initialize from point view
               of the model, no B2R issue needs to be considered.
* parameters :
*             none
* return: none
*********************************************************************
*/
void model_init(int rank)
{
    for (int z = 0; z < B2R_BLOCK_SIZE_Z; ++z)
        for(int r = 0; r < B2R_BLOCK_SIZE_Y; r++)
        {
            for(int c = 0; c < B2R_BLOCK_SIZE_X; c++)
            {
                int global_id = z * B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y + r*B2R_BLOCK_SIZE_X + c;
                b2r_env_grid[global_id] = rand() % 2;
            }
        }
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

int gol_live_neighbor_cnt(int x, int y, int j)
{
    return 0;
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
    int global_id;
    int Rd = B2R_R*B2R_D;
    for (int z = Rd; z < B2R_BLOCK_SIZE_Z-Rd; ++z)
    {
        for(int r = Rd; r < B2R_BLOCK_SIZE_Y-Rd; r++)
        {
            for(int c = Rd; c < B2R_BLOCK_SIZE_X-Rd; c++)
            {
                global_id = z * B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y + r*B2R_BLOCK_SIZE_X + c;
                cout << b2r_env_grid[global_id] << ",";
            }
            cout << endl;
        }
    }
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
void print_sent_data()
{
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
                    cout << pdata[global_id] << " ";
                }
                cout << endl;
            }
        }
    }
}
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
void print_recv_data()
{
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
                    cout << pdata[global_id] << " ";
                }
                cout << endl;
            }
        }
    }
}

#endif
