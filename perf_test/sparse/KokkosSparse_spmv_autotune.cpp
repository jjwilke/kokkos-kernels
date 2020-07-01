/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse_spmv.hpp>
#include "KokkosKernels_default_types.hpp"
#include <spmv/Kokkos_SPMV.hpp>
#include <spmv/Kokkos_SPMV_Inspector.hpp>

#ifdef HAVE_CUSPARSE
#include <CuSparse_SPMV.hpp>
#endif

#ifdef HAVE_MKL
#include <MKL_SPMV.hpp>
#endif

#ifdef KOKKOS_ENABLE_OPENMP
#include <OpenMPStatic_SPMV.hpp>
#include <OpenMPDynamic_SPMV.hpp>
#include <OpenMPSmartStatic_SPMV.hpp>
#endif

typedef default_scalar Scalar;
typedef default_lno_t Ordinal;
typedef default_size_type Offset;
typedef default_layout Layout;

#ifndef MAX_THREADS_PER_BLOCK
#define MAX_THREADS_PER_BLOCK 128
#endif

#ifndef MIN_BLOCKS_PER_SM
#define MIN_BLOCKS_PER_SM 8
#endif

#ifndef POLICY_TYPE
#define POLICY_TYPE Kokkos::Static
#endif

#ifndef TEAM_SIZE
#define TEAM_SIZE 32
#endif

#ifndef VECTOR_LENGTH
#define VECTOR_LENGTH 8
#endif

#ifdef KOKKOSKERNELS_ETI_ONLY
#error not okay
#else
#warning okay
#endif 

#ifdef KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#warning okay
#else
#error not okay
#endif

template<typename AType, typename XType, typename YType>
void matvec(AType& A, XType x, YType y){
  using exec_space = typename AType::execution_space;
  using Policy = Kokkos::TeamPolicy<exec_space, Kokkos::LaunchBounds<128, 2>, Kokkos::Schedule<Kokkos::Dynamic>>;
  auto tuner = KokkosSparse::TeamTuner<Policy>();
  KokkosSparse::spmv_tuned(KokkosSparse::NoTranspose,1.0,A,x,0.0,y,tuner);
}

void test_crs_matrix_singlevec(Ordinal numRows, Ordinal numCols, const char* filename)
{
  typedef KokkosSparse::CrsMatrix<Scalar, Ordinal, Kokkos::DefaultExecutionSpace, void, Offset> matrix_type;
  typedef typename Kokkos::View<Scalar*, Layout> mv_type;
  typedef typename mv_type::HostMirror h_mv_type;

  srand(17312837);
  matrix_type A;
  if (filename) {
    A = KokkosKernels::Impl::read_kokkos_crst_matrix<matrix_type>(filename);
  } else {
    Offset nnz = 10 * numRows;
    //note: the help text says the bandwidth is fixed at 0.01 * numRows
    A = KokkosKernels::Impl::kk_generate_sparse_matrix<matrix_type>(numRows, numCols, nnz, 0, 0.01 * numRows);
  }
  numRows = A.numRows();
  numCols = A.numCols();
  Offset nnz = A.nnz();
  mv_type x("X", numCols);
  mv_type y("Y", numRows);
  h_mv_type h_x = Kokkos::create_mirror_view(x);
  h_mv_type h_y = Kokkos::create_mirror_view(y);
  h_mv_type h_y_compare = Kokkos::create_mirror(y);

  typename matrix_type::StaticCrsGraphType::HostMirror h_graph = Kokkos::create_mirror(A.graph);
  typename matrix_type::values_type::HostMirror h_values = Kokkos::create_mirror_view(A.values);

  for(int i=0; i<numCols;i++) {
    h_x(i) = (Scalar) (1.0*(rand()%40)-20.);
  }
  for(int i=0; i<numRows;i++) {
    h_y(i) = (Scalar) (1.0*(rand()%40)-20.);
  }

  Kokkos::deep_copy(x,h_x);
  Kokkos::deep_copy(y,h_y);
  Kokkos::deep_copy(A.graph.entries,h_graph.entries);
  Kokkos::deep_copy(A.values,h_values);
  // Benchmark
  double min_time = 1.0e32;
  double max_time = 0.0;
  double ave_time = 0.0;
  int loop = 5;
  for (int i=0;i<loop;i++) {
    Kokkos::Timer timer;
    matvec(A,x,y);
    Kokkos::fence();
    double time = timer.seconds();
    ave_time += time;
    if(time>max_time) max_time = time;
    if(time<min_time) min_time = time;
  }

  // Performance Output
  double matrix_size = 1.0*((nnz*(sizeof(Scalar)+sizeof(Ordinal)) + numRows*sizeof(Offset)))/1024/1024;
  double vector_size = 2.0*numRows*sizeof(Scalar)/1024/1024;
  double vector_readwrite = (nnz+numCols)*sizeof(Scalar)/1024/1024;

  double problem_size = matrix_size+vector_size;
  printf("NNZ NumRows NumCols ProblemSize(MB) AveBandwidth(GB/s) MinBandwidth(GB/s) MaxBandwidth(GB/s) AveGFlop MinGFlop MaxGFlop aveTime(ms) maxTime(ms) minTime(ms)\n");
  printf("%i %i %i %6.2lf ( %6.2lf %6.2lf %6.2lf ) ( %6.3lf %6.3lf %6.3lf ) ( %6.3lf %6.3lf %6.3lf ) RESULT\n",nnz, numRows,numCols,problem_size,
          (matrix_size+vector_readwrite)/ave_time*loop/1024, (matrix_size+vector_readwrite)/max_time/1024,(matrix_size+vector_readwrite)/min_time/1024,
          2.0*nnz*loop/ave_time/1e9, 2.0*nnz/max_time/1e9, 2.0*nnz/min_time/1e9,
          ave_time/loop*1000, max_time*1000, min_time*1000);
}

int main(int argc, char **argv)
{
  long long int size = 110503; // a prime number
  char* filename = NULL;

  Kokkos::initialize(argc,argv);

  test_crs_matrix_singlevec(size,size,filename);

  Kokkos::finalize();
}

