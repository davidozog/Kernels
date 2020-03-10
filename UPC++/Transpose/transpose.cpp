/*
Copyright (c) 2020, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
* Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    transpose

PURPOSE: This program tests the efficiency with which a square matrix
         can be transposed and stored in another matrix. The matrices
         are distributed identically.

USAGE:   Program inputs are the matrix order, the number of times to
         repeat the operation, and the communication mode

         transpose <# iterations> <matrix order> [tile size]

         An optional parameter specifies the tile size used to divide the
         individual matrix blocks for improved cache and TLB performance.

         The output consists of diagnostics to make sure the
         transpose worked and timing statistics.

FUNCTIONS CALLED:

         Other than UPC++ or standard C/C++ functions, the following
         functions are used in this program:

          wtime()           Portable wall-timer interface.
          bail_out()        Determine global error and exit if nonzero.

HISTORY: UPC++ variant written by David Ozog, March 2020.
         Based on the SHMEM variant written by Tom St. John, July 2015.
         Based on the MPI variant by Tim Mattson, Rob VdW, etc. 1999-2015.
*******************************************************************/

/******************************************************************
                     Layout nomenclature
                     -------------------

o Each rank owns one block of columns (Colblock) of the overall
  matrix to be transposed, as well as of the transposed matrix.
o Colblock is stored contiguously in the memory of the rank.
  The stored format is column major, which means that matrix
  elements (i,j) and (i+1,j) are adjacent, and (i,j) and (i,j+1)
  are "order" words apart
o Colblock is logically composed of #ranks Blocks, but a Block is
  not stored contiguously in memory. Conceptually, the Block is
  the unit of data that gets communicated between ranks. Block i of
  rank j is locally transposed and gathered into a buffer called Work,
  which is sent to rank i, where it is scattered into Block j of the
  transposed matrix.
o When tiling is applied to reduce TLB misses, each block gets
  accessed by tiles.
o The original and transposed matrices are called A and B

 -----------------------------------------------------------------
|           |           |           |                             |
| Colblock  |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |  Block    |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |Tile|      |           |                             |
|           |    |      |           |   Overall Matrix            |
|           |----       |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
 -----------------------------------------------------------------*/

#include <par-res-kern_general.h>
#include <par-res-kern_upcxx.hpp>

#ifdef USE_CUDA
const upcxx::memory_kind memkind = upcxx::memory_kind::cuda_device;
#else
const upcxx::memory_kind memkind = upcxx::memory_kind::host;
#endif

typedef upcxx::dist_object<upcxx::global_ptr<double, memkind>> distributred_matrix;

#define A(i,j)        A_p[(i+istart)+order*(j)]
#define B(i,j)        B_p[(i+istart)+order*(j)]
#define Work_in(phase, i,j)  Work_in_p[phase-1][i+Block_order*(j)]
#define Work_out(i,j) Work_out_p[i+Block_order*(j)]

int main(int argc, char ** argv)
{
  long Block_order;        /* number of columns owned by rank       */
  int Block_size;          /* size of a single block                */
  int Colblock_size;       /* size of column block                  */
  int Tile_order=32;       /* default Tile order                    */
  int tiling;              /* boolean: true if tiling is used       */
  int Num_procs;           /* number of ranks                       */
  int order;               /* order of overall matrix               */
  int send_to, recv_from;  /* ranks with which to communicate       */
  long bytes;              /* combined size of matrices             */
  int my_ID;               /* rank                                  */
  int root=0;              /* rank of root                          */
  int iterations;          /* number of times to do the transpose   */
  long i, j, it, jt, istart;/* dummies                              */
  int iter;                /* index of iteration                    */
  int phase;               /* phase inside staged communication     */
  int colstart;            /* starting column for owning rank       */
  int error;               /* error flag                            */
  double * RESTRICT A_p;   /* original matrix column block          */
  double * RESTRICT B_p;   /* transposed matrix column block        */
  upcxx::global_ptr<double, memkind> *Work_in; /* global in buffers */
  double **Work_in_p;      /* workspace for the transpose function  */
  upcxx::global_ptr<double, memkind> Work_out; /* global out buffer */
  double *Work_out_p;      /* workspace for the transpose function  */
  double epsilon = 1.e-8;  /* error tolerance                       */
  double local_trans_time, avgtime,
         *trans_time;      /* timing parameters                     */
  double abserr_local,     /* local and aggregate error             */
         abserr_total;
  int *arguments;          /* command line arguments                */
#ifdef USE_CUDA
  upcxx::device_allocator<upcxx::cuda_device> gpu_alloc;
#endif

/*********************************************************************
** Initialize the UPC++ environment
*********************************************************************/
  upcxx::init();

  my_ID = upcxx::rank_me();
  Num_procs = upcxx::rank_n();

  if (upcxx::rank_me() == 0) {
    std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
    std::cout << "UPC++ matrix transpose: B = A^T" << std::endl;
  }

  size_t n_args = 3;
  arguments = new int[n_args];

/*********************************************************************
** Process, test and broadcast input parameters
*********************************************************************/
  error = 0;
  if (my_ID == root) {
    if (argc != 3 && argc != 4) {
      std::cout << "Usage: " << *argv <<
              " <# iterations> <matrix order> [Tile size]" << std::endl;
      error = 1; goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    arguments[0] = iterations;
    if (iterations < 1) {
      std::cout << "ERROR: iterations must be >= 1 : " << iterations << std::endl;
      error = 1; goto ENDOFTESTS;
    }

    order = atoi(*++argv);
    arguments[1] = order;
    if (order < Num_procs) {
      std::cout << "ERROR: matrix order " << order << " should at least # procs "
                << Num_procs << std::endl;
      error = 1; goto ENDOFTESTS;
    }
    if (order % Num_procs) {
      std::cout << "ERROR: matrix order " << order <<
                  " should be divisible by # procs " << Num_procs << std::endl;
      error = 1; goto ENDOFTESTS;
    }

    if (argc == 4) Tile_order = atoi(*++argv);
    arguments[2] = Tile_order;

    ENDOFTESTS:;
  }
  bail_out(error);

  if (my_ID == root) {
    std::cout << "Number of ranks      = " << Num_procs << std::endl;
    std::cout << "Matrix order         = " << order << std::endl;
    std::cout << "Number of iterations = " << iterations << std::endl;
    if ((Tile_order > 0) && (Tile_order < order))
    std::cout << "Tile size            = " << Tile_order << std::endl;
    else std::cout << "Untiled" << std::endl;
  }

  /*  Broadcast the root's input data to all ranks */
  upcxx::broadcast(arguments, n_args, root).wait();

  iterations = arguments[0];
  order = arguments[1];
  Tile_order = arguments[2];

  upcxx::barrier();

  delete(arguments);

  /* a non-positive tile size means no tiling of the local transpose */
  tiling = (Tile_order > 0) && (Tile_order < order);
  bytes = 2 * sizeof(double) * order * order;

/*********************************************************************
** The matrix is broken up into column blocks that are mapped one to a
** rank.  Each column block is made up of Num_procs smaller square
** blocks of order block_order.
*********************************************************************/

  Block_order    = order / Num_procs;
  colstart       = Block_order * my_ID;
  Colblock_size  = order * Block_order;
  Block_size     = Block_order * Block_order;

/*********************************************************************
** Create the column block of the test matrix, the row block of the
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  A_p = new double[Colblock_size];
  if (A_p == NULL) {
    std::cout << " Error allocating space for original matrix on node " << my_ID << std::endl;
    error = 1;
  }
  bail_out(error);

  B_p = new double[Colblock_size];
  if (B_p == NULL) {
    std::cout << " Error allocating space for transpose matrix on node " << my_ID << std::endl;
    error = 1;
  }
  bail_out(error);

  if (Num_procs > 1) {
    Work_in    = new upcxx::global_ptr<double, memkind>[Num_procs - 1];
    Work_in_p  = new double *[Num_procs - 1];

#ifdef USE_CUDA
    auto gpu_device = upcxx::cuda_device(0);

    /* upcxx::device_allocator requires padding the size to include at least an extra page */
    size_t buff_size = Block_size * (Num_procs - 1) * sizeof(double);
    const size_t MiB = 1024 * 1024;
    buff_size += MiB - (buff_size % MiB);

    upcxx::device_allocator<upcxx::cuda_device> gpu_alloc(gpu_device, buff_size);
    Work_out = gpu_alloc.allocate<double>(Block_size);
#else
    Work_out = upcxx::new_array<double>(Block_size);
#endif

    assert(Work_out);
    Work_out_p = Work_out.local();

    if ((Work_in_p == NULL) || (Work_out_p == NULL)) {
      std::cout << " Error allocating space for work on node " << my_ID << std::endl;
      error = 1;
    }
    bail_out(error);
    for (i = 0; i < (Num_procs - 1); i++) {
#ifdef USE_CUDA
      Work_in[i] = gpu_alloc.allocate<double>(Block_size);
#else
      Work_in[i] = upcxx::new_array<double>(Block_size);
#endif
      assert(Work_in_p);
      Work_in_p[i] = Work_in[i].local();
      if (Work_in_p[i] == NULL) {
        std::cout << " Error allocating space for work on node "
                  <<   my_ID << std::endl;
        error = 1;
      }
      bail_out(error);
    }
  }

  /* Fill the original column matrices */
  istart = 0;
  for (j = 0; j < Block_order; j++)
    for (i = 0; i < order; i++)  {
      A(i,j) = (double) (order*(j+colstart) + i);
      B(i,j) = 0.0;
  }

  upcxx::barrier();

  for (iter = 0; iter <= iterations; iter++) {

    /* start timer after a warmup iteration */
    if (iter == 1) {
      upcxx::barrier();
      local_trans_time = wtime();
    }

    /* do the local transpose */
    istart = colstart;
    if (!tiling) {
      for (i=0; i<Block_order; i++)
        for (j=0; j<Block_order; j++) {
          B(j,i) += A(i,j);
          A(i,j) += 1.0;
	}
    }
    else {
      for (i=0; i<Block_order; i+=Tile_order)
        for (j=0; j<Block_order; j+=Tile_order)
          for (it=i; it<MIN(Block_order,i+Tile_order); it++)
            for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
              B(jt,it) += A(it,jt);
              A(it,jt) += 1.0;
            }
    }

    for (phase=1; phase<Num_procs; phase++){
      recv_from = (my_ID + phase            )%Num_procs;
      send_to   = (my_ID - phase + Num_procs)%Num_procs;

      istart = send_to*Block_order;
      if (!tiling) {
        for (i=0; i<Block_order; i++)
          for (j=0; j<Block_order; j++){
	    Work_out(j,i) = A(i,j);
            A(i,j) += 1.0;
	  }
      }
      else {
        for (i=0; i<Block_order; i+=Tile_order)
          for (j=0; j<Block_order; j+=Tile_order)
            for (it=i; it<MIN(Block_order,i+Tile_order); it++)
              for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
                Work_out(jt,it) = A(it,jt);
                A(it,jt) += 1.0;
	      }
      }

      upcxx::barrier();

      distributred_matrix Work(Work_in[phase-1]);

      auto Work_dest = Work.fetch(send_to).wait();
#ifdef USE_CUDA
      upcxx::copy(&Work_out_p[0], Work_dest, Block_size).wait();
#else
      upcxx::rput(&Work_out_p[0], Work_dest, Block_size).wait();
#endif

      upcxx::barrier();

      istart = recv_from*Block_order;
      /* scatter received block to transposed matrix; no need to tile */
      for (j=0; j<Block_order; j++)
        for (i=0; i<Block_order; i++)
          B(i,j) += Work_in(phase, i,j);

    }  /* end of phase loop  */

  } /* end of iterations */

  local_trans_time = wtime() - local_trans_time;

  upcxx::barrier();

  trans_time = upcxx::reduce_all(&local_trans_time, upcxx::op_max).wait();

  abserr_local = 0.0;
  istart = 0;
  double addit = 0.5 * ( (iterations+1.) * (double)iterations );
  for (j=0;j<Block_order;j++) for (i=0;i<order; i++) {
      abserr_local += fabs(B(i,j) - (double)((order*i + j+colstart)*(iterations+1)+addit));
  }

  upcxx::barrier();

  abserr_total = upcxx::reduce_all(abserr_local, upcxx::op_add).wait();

  if (abserr_total <= epsilon) {
    avgtime = trans_time[0]/(double)iterations;
    if (my_ID == root) {
        std::cout << "Solution validates" << std::endl;
        std::cout << "Rate (MB/s): " << 1.0E-06 * bytes / avgtime
                  << " Avg time (s): " << avgtime << std::endl;
#ifdef VERBOSE
        std::cout << "Summed errors: " << abserr_total << std::endl;
#endif
    }
  } else {
    error = 1;
    if (my_ID == root) {
        std::cout << "ERROR: Aggregate squared error " << abserr_total
                  << " exceeds threshold " << epsilon << std::endl;
    }
    std::cout << std::flush;
    std::cout << "ERROR: PE=" << my_ID << ", error = " << abserr_local << std::endl;
  }

  bail_out(error);

  if (Num_procs > 1) {
      for(i = 0; i < Num_procs - 1; i++)
#ifdef USE_CUDA
	gpu_alloc.deallocate(Work_in[i]);

      gpu_alloc.deallocate(Work_out);
#else
	upcxx::delete_array(Work_in[i]);

      upcxx::delete_array(Work_out);
#endif

      delete Work_in_p;
      delete Work_in;
  }

  upcxx::finalize();
  exit(EXIT_SUCCESS);
}
