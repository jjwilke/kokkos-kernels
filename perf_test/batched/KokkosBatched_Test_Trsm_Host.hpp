/// \author Kyungjoo Kim (kyukim@sandia.gov)

#define __KOKKOSBATCHED_INTEL_MKL__
//#define __KOKKOSBATCHED_INTEL_MKL_BATCHED__

#include <iomanip>

#if defined(__KOKKOSBATCHED_INTEL_MKL__)
#include "mkl.h"
#endif

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosBatched_Vector.hpp"

#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

namespace KokkosBatched {
  namespace Experimental {
    namespace PerfTest {

#undef FLOP_MUL
#undef FLOP_ADD

      // no complex yet
#if defined( KokkosBatched_Test_Trsm_Host_Complex )      
#define FLOP_MUL 6.0
#define FLOP_ADD 2.0
      typedef Kokkos::complex<double> value_type;
#endif

#if defined( KokkosBatched_Test_Trsm_Host_Real )      
#define FLOP_MUL 1.0
#define FLOP_ADD 1.0      
      typedef double value_type;
#endif

      double FlopCountLower(int mm, int nn) {
        double m = (double)mm;    double n = (double)nn;
        return (FLOP_MUL*(0.5*m*n*(n+1.0)) +
                FLOP_ADD*(0.5*m*n*(n-1.0)));
      }

      double FlopCountUpper(int mm, int nn) {
        double m = (double)mm;    double n = (double)nn;
        return (FLOP_MUL*(0.5*m*n*(n+1.0)) +
                FLOP_ADD*(0.5*m*n*(n-1.0)));
      }

      template<int test, int BlkSize, int NumCols, typename HostSpaceType, typename AlgoTagType>
      void Trsm(const int NN) {
        typedef Kokkos::Schedule<Kokkos::Static> ScheduleType;

        constexpr int VectorLength = DefaultVectorLength<value_type,typename HostSpaceType::memory_space>::value;
        const int N = NN/VectorLength;

        {
          std::string value_type_name;
          if (std::is_same<value_type,double>::value)                   value_type_name = "double";
          if (std::is_same<value_type,Kokkos::complex<double> >::value) value_type_name = "Kokkos::complex<double>";

#if   defined(__AVX__) || defined(__AVX2__)
          std::cout << "AVX or AVX2 is defined: datatype " << value_type_name <<  " a vector length " << VectorLength << "\n";
#elif defined(__AVX512F__)
          std::cout << "AVX512 is defined: datatype " << value_type_name <<  " a vector length " << VectorLength << "\n";
#else
          std::cout << "SIMD (compiler vectorization) is defined: datatype " << value_type_name <<  " a vector length " << VectorLength << "\n";
#endif
        }

        switch (test) {
        case 0: std::cout << "TestID = Left,  Lower, NoTrans,    UnitDiag\n"; break;
        case 1: std::cout << "TestID = Left,  Lower, NoTrans, NonUnitDiag\n"; break;
        case 2: std::cout << "TestID = Right, Upper, NoTrans,    UnitDiag\n"; break;
        case 3: std::cout << "TestID = Right, Upper, NoTrans, NonUnitDiag\n"; break;
        case 4: std::cout << "TestID = Left,  Upper, NoTrans, NonUnitDiag\n"; break;
        }

        // when m == n, lower upper does not matter (unit and nonunit)
        double flop = 0;
        switch (test) {
        case 0:
        case 1:
          flop = FlopCountLower(BlkSize,NumCols);
          break;
        case 2:
        case 3:
        case 4:
          flop = FlopCountUpper(BlkSize,NumCols);
          break;
        }
        flop *= (N*VectorLength);

        const double tmax = 1.0e15;

        const int iter_begin = -10, iter_end = 100;
        Kokkos::Impl::Timer timer;

        ///
        /// Reference version using MKL DTRSM
        ///
        Kokkos::View<value_type***,Kokkos::LayoutRight,HostSpaceType> bref;
        Kokkos::View<value_type***,Kokkos::LayoutRight,HostSpaceType>
          amat("amat", N*VectorLength, BlkSize, BlkSize),
          bmat("bmat", N*VectorLength, BlkSize, NumCols);

        typedef Vector<SIMD<value_type>,VectorLength> VectorType;
        Kokkos::View<VectorType***,Kokkos::LayoutRight,HostSpaceType>
          amat_simd("amat_simd", N, BlkSize, BlkSize),
          bmat_simd("bmat_simd", N, BlkSize, NumCols); 
      
        Random<value_type> random;

        for (int k=0;k<N*VectorLength;++k) {
          const int k0 = k/VectorLength, k1 = k%VectorLength;
          for (int i=0;i<BlkSize;++i)
            for (int j=0;j<BlkSize;++j) {
              amat(k, i, j) = random.value() + 4.0*(i==j);
              amat_simd(k0, i, j)[k1] = amat(k, i, j);
            }
          for (int i=0;i<BlkSize;++i)
            for (int j=0;j<NumCols;++j) {
              bmat(k, i, j) = random.value(); 
              bmat_simd(k0, i, j)[k1] = bmat(k, i, j);
            }
        }
      
        // for KNL
        constexpr size_t LLC_CAPACITY = 34*1024*1024;
        Flush<LLC_CAPACITY> flush;

        ///
        /// Reference version using MKL DTRSM
        /// 
#if defined(__KOKKOSBATCHED_INTEL_MKL__)
        {
          Kokkos::View<value_type***,Kokkos::LayoutRight,HostSpaceType>
            a("a", N*VectorLength, BlkSize, BlkSize),
            b("b", N*VectorLength, BlkSize, NumCols);

          {
            double tavg = 0, tmin = tmax;
            for (int iter=iter_begin;iter<iter_end;++iter) {
              // flush
              flush.run();

              // initialize matrices
              Kokkos::deep_copy(a, amat);
              Kokkos::deep_copy(b, bmat);

              HostSpaceType::fence();
              timer.reset();

              Kokkos::RangePolicy<HostSpaceType,ScheduleType> policy(0, N*VectorLength);
              Kokkos::parallel_for
                (policy,
                 KOKKOS_LAMBDA(const int k) {
                  auto aa = Kokkos::subview(a, k, Kokkos::ALL(), Kokkos::ALL());
                  auto bb = Kokkos::subview(b, k, Kokkos::ALL(), Kokkos::ALL());

                  switch (test) {
                  case 0:
                    cblas_dtrsm(CblasRowMajor,
                                CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                                BlkSize, NumCols,
                                1.0,
                                (double*)aa.data(), aa.stride_0(),
                                (double*)bb.data(), bb.stride_0());
                    break;
                  case 1:
                    cblas_dtrsm(CblasRowMajor,
                                CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                                BlkSize, NumCols,
                                1.0,
                                (double*)aa.data(), aa.stride_0(),
                                (double*)bb.data(), bb.stride_0());
                    break;
                  case 2:
                    cblas_dtrsm(CblasRowMajor,
                                CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
                                BlkSize, NumCols,
                                1.0,
                                (double*)aa.data(), aa.stride_0(),
                                (double*)bb.data(), bb.stride_0());
                    break;
                  case 3:
                    cblas_dtrsm(CblasRowMajor,
                                CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                                BlkSize, NumCols,
                                1.0,
                                (double*)aa.data(), aa.stride_0(),
                                (double*)bb.data(), bb.stride_0());
                    break;
                  case 4:
                    cblas_dtrsm(CblasRowMajor,
                                CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                BlkSize, NumCols,
                                1.0,
                                (double*)aa.data(), aa.stride_0(),
                                (double*)bb.data(), bb.stride_0());
                    break;
                  }
                });

              HostSpaceType::fence();
              const double t = timer.seconds();
              tmin = std::min(tmin, t);
              tavg += (iter >= 0)*t;
            }
            tavg /= iter_end;

            double sum = 0;
            for (int i=0;i<b.dimension(0);++i)
              for (int j=0;j<b.dimension(1);++j)
                for (int k=0;k<b.dimension(2);++k)
                  sum += std::abs(bmat(i,j,k));

            std::cout << std::setw(10) << "MKL TRSM"
                      << " BlkSize = " << std::setw(3) << BlkSize
                      << " NumCols = " << std::setw(3) << NumCols
                      << " time = " << std::scientific << tmin
                      << " avg flop/s = " << (flop/tavg)
                      << " max flop/s = " << (flop/tmin)
                      << " sum abs(B)  = " << sum
                      << std::endl;

            bref = b;
          }
        }
#if defined(__KOKKOSBATCHED_INTEL_MKL_BATCHED__)
        {
          Kokkos::View<value_type***,Kokkos::LayoutRight,HostSpaceType>
            a("a", N*VectorLength, BlkSize, BlkSize),
            b("b", N*VectorLength, BlkSize, NumCols);
        
          value_type 
            *aa[N*VectorLength],
            *bb[N*VectorLength];

          for (int k=0;k<N*VectorLength;++k) {
            aa[k] = &a(k, 0, 0);
            bb[k] = &b(k, 0, 0);
          }
        
          {
            double tavg = 0, tmin = tmax;
          
            MKL_INT blksize[1] = { BlkSize };
            MKL_INT numcols[1] = { NumCols };

            MKL_INT lda[1] = { a.stride_1() };
            MKL_INT ldb[1] = { b.stride_1() };

            double one[1] = { 1.0 };
            MKL_INT size_per_grp[1] = { N*VectorLength };

            for (int iter=iter_begin;iter<iter_end;++iter) {
              // flush
              flush.run();

              // initialize matrices
              Kokkos::deep_copy(a, amat);
              Kokkos::deep_copy(b, bmat);

              HostSpaceType::fence();
              timer.reset();

              switch (test) {
              case 0: {
                CBLAS_SIDE side[1] = {CblasLeft};
                CBLAS_UPLO uplo[1] = {CblasLower};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasUnit};

                cblas_dtrsm_batch(CblasRowMajor,
                                  side, uplo, transA, diag,
                                  blksize, numcols,
                                  one,
                                  (const double**)aa, lda,
                                  (double**)bb, ldb,
                                  1, size_per_grp);
                break;
              }
              case 1: {
                CBLAS_SIDE side[1] = {CblasLeft};
                CBLAS_UPLO uplo[1] = {CblasLower};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasNonUnit};

                cblas_dtrsm_batch(CblasRowMajor,
                                  side, uplo, transA, diag,
                                  blksize, numcols,
                                  one,
                                  (const double**)aa, lda,
                                  (double**)bb, ldb,
                                  1, size_per_grp);
                break;
              }
              case 2: {
                CBLAS_SIDE side[1] = {CblasRight};
                CBLAS_UPLO uplo[1] = {CblasUpper};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasUnit};

                cblas_dtrsm_batch(CblasRowMajor,
                                  side, uplo, transA, diag,
                                  blksize, numcols,
                                  one,
                                  (const double**)aa, lda,
                                  (double**)bb, ldb,
                                  1, size_per_grp);
                break;
              }
              case 3: {
                CBLAS_SIDE side[1] = {CblasRight};
                CBLAS_UPLO uplo[1] = {CblasUpper};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasNonUnit};

                cblas_dtrsm_batch(CblasRowMajor,
                                  side, uplo, transA, diag,
                                  blksize, numcols,
                                  one,
                                  (const double**)aa, lda,
                                  (double**)bb, ldb,
                                  1, size_per_grp);
                break;
              }
              case 4: {
                CBLAS_SIDE side[1] = {CblasLeft};
                CBLAS_UPLO uplo[1] = {CblasUpper};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasNonUnit};

                cblas_dtrsm_batch(CblasRowMajor,
                                  side, uplo, transA, diag,
                                  blksize, numcols,
                                  one,
                                  (const double**)aa, lda,
                                  (double**)bb, ldb,
                                  1, size_per_grp);
                break;
              }
              }
            
              HostSpaceType::fence();
              const double t = timer.seconds();
              tmin = std::min(tmin, t);
              tavg += (iter >= 0)*t;
            }
            tavg /= iter_end;

            double diff = 0;
            for (int i=0;i<bref.dimension(0);++i)
              for (int j=0;j<bref.dimension(1);++j)
                for (int k=0;k<bref.dimension(2);++k)
                  diff += std::abs(bref(i,j,k) - b(i,j,k));

            std::cout << std::setw(10) << "MKL Batch"
                      << " BlkSize = " << std::setw(3) << BlkSize
                      << " NumCols = " << std::setw(3) << NumCols
                      << " time = " << std::scientific << tmin
                      << " avg flop/s = " << (flop/tavg)
                      << " max flop/s = " << (flop/tmin)
                      << " diff to ref = " << diff
                      << std::endl;
          }
        }
#endif

#if defined(__KOKKOSBATCHED_INTEL_MKL_COMPACT_BATCHED__)
        {
          Kokkos::View<VectorType***,Kokkos::LayoutRight,HostSpaceType>
            a("a", N, BlkSize, BlkSize),
            b("b", N, BlkSize, NumCols);
        
          {
            double tavg = 0, tmin = tmax;
          
            MKL_INT blksize[1] = { BlkSize };
            MKL_INT numcols[1] = { NumCols };

            MKL_INT lda[1] = { a.stride_1() };
            MKL_INT ldb[1] = { b.stride_1() };
          
            double one[1] = { 1.0 };
            MKL_INT size_per_grp[1] = { N*VectorLength };

            compact_t A_p, B_p;
            A_p.layout = CblasRowMajor;
            A_p.rows = blksize;
            A_p.cols = blksize;
            A_p.stride = lda;
            A_p.group_count = 1;
            A_p.size_per_group = size_per_grp;
            A_p.format = VectorLength;
            A_p.mat = (double*)a.data();
          
            B_p.layout = CblasRowMajor;
            B_p.rows = blksize;
            B_p.cols = numcols;
            B_p.stride = ldb;
            B_p.group_count = 1;
            B_p.size_per_group = size_per_grp;
            B_p.format = VectorLength;
            B_p.mat = (double*)b.data();

            for (int iter=iter_begin;iter<iter_end;++iter) {
              // flush
              flush.run();

              // initialize matrices
              Kokkos::deep_copy(a, amat_simd);
              Kokkos::deep_copy(b, bmat_simd);

              HostSpaceType::fence();
              timer.reset();

              switch (test) {
              case 0: {
                CBLAS_SIDE side[1] = {CblasLeft};
                CBLAS_UPLO uplo[1] = {CblasLower};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasUnit};

                cblas_dtrsm_compute_batch(side, uplo, transA, diag,
                                          one,
                                          &A_p,
                                          &B_p);
                break;
              }
              case 1: {
                CBLAS_SIDE side[1] = {CblasLeft};
                CBLAS_UPLO uplo[1] = {CblasLower};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasNonUnit};

                cblas_dtrsm_compute_batch(side, uplo, transA, diag,
                                          one,
                                          &A_p,
                                          &B_p);
                break;
              }
              case 2: {
                CBLAS_SIDE side[1] = {CblasRight};
                CBLAS_UPLO uplo[1] = {CblasUpper};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasUnit};

                cblas_dtrsm_compute_batch(side, uplo, transA, diag,
                                          one,
                                          &A_p,
                                          &B_p);
                break;
              }
              case 3: {
                CBLAS_SIDE side[1] = {CblasRight};
                CBLAS_UPLO uplo[1] = {CblasUpper};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasNonUnit};

                cblas_dtrsm_compute_batch(side, uplo, transA, diag,
                                          one,
                                          &A_p,
                                          &B_p);
                break;
              }
              case 4: {
                CBLAS_SIDE side[1] = {CblasLeft};
                CBLAS_UPLO uplo[1] = {CblasUpper};
                CBLAS_TRANSPOSE transA[1] = {CblasNoTrans};
                CBLAS_DIAG diag[1] = {CblasNonUnit};

                cblas_dtrsm_compute_batch(side, uplo, transA, diag,
                                          one,
                                          &A_p,
                                          &B_p);
                break;
              }
              }
            
              HostSpaceType::fence();
              const double t = timer.seconds();
              tmin = std::min(tmin, t);
              tavg += (iter >= 0)*t;
            }
            tavg /= iter_end;

            double diff = 0;
            for (int i=0;i<bref.dimension(0);++i)
              for (int j=0;j<bref.dimension(1);++j)
                for (int k=0;k<bref.dimension(2);++k)
                  diff += std::abs(bref(i,j,k) - b(i/VectorLength,j,k)[i%VectorLength]);

            std::cout << std::setw(10) << "MKL Cmpt"
                      << " BlkSize = " << std::setw(3) << BlkSize
                      << " NumCols = " << std::setw(3) << NumCols
                      << " time = " << std::scientific << tmin
                      << " avg flop/s = " << (flop/tavg)
                      << " max flop/s = " << (flop/tmin)
                      << " diff to ref = " << diff
                      << std::endl;
          }
        }
#endif

#endif

        ///
        /// Plain version (comparable to micro BLAS version)
        ///
        {
          Kokkos::View<value_type***,Kokkos::LayoutRight,HostSpaceType>
            a("a", N*VectorLength, BlkSize, BlkSize),
            b("b", N*VectorLength, BlkSize, NumCols);

          {
            double tavg = 0, tmin = tmax;
            for (int iter=iter_begin;iter<iter_end;++iter) {
              // flush
              flush.run();

              // initialize matrices
              Kokkos::deep_copy(a, amat);
              Kokkos::deep_copy(b, bmat);

              HostSpaceType::fence();
              timer.reset();

              Kokkos::RangePolicy<HostSpaceType,ScheduleType> policy(0, N*VectorLength);
              Kokkos::parallel_for
                (policy,
                 KOKKOS_LAMBDA(const int k) {
                  auto aa = Kokkos::subview(a, k, Kokkos::ALL(), Kokkos::ALL());
                  auto bb = Kokkos::subview(b, k, Kokkos::ALL(), Kokkos::ALL());

                  switch (test) {
                  case 0: 
                    SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 1:
                    SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::NonUnit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 2:
                    SerialTrsm<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::Unit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 3:
                    SerialTrsm<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 4:
                    SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  }
                });

              HostSpaceType::fence();
              const double t = timer.seconds();
              tmin = std::min(tmin, t);
              tavg += (iter >= 0)*t;
            }
            tavg /= iter_end;

            double diff = 0;
            for (int i=0;i<bref.dimension(0);++i)
              for (int j=0;j<bref.dimension(1);++j)
                for (int k=0;k<bref.dimension(2);++k)
                  diff += std::abs(bref(i,j,k) - b(i,j,k));

            std::cout << std::setw(10) << "KK Scalar"
                      << " BlkSize = " << std::setw(3) << BlkSize
                      << " NumCols = " << std::setw(3) << NumCols
                      << " time = " << std::scientific << tmin
                      << " avg flop/s = " << (flop/tavg)
                      << " max flop/s = " << (flop/tmin)
                      << " diff to ref = " << diff
                      << std::endl;
          }
        }

        ///
        /// SIMD with appropriate data layout
        ///
        {
          Kokkos::View<VectorType***,Kokkos::LayoutRight,HostSpaceType>
            a("a", N, BlkSize, BlkSize),
            b("b", N, BlkSize, NumCols);

          {
            double tavg = 0, tmin = tmax;
            for (int iter=iter_begin;iter<iter_end;++iter) {
              // flush
              flush.run();

              // initialize matrices
              Kokkos::deep_copy(a, amat_simd);
              Kokkos::deep_copy(b, bmat_simd);

              HostSpaceType::fence();
              timer.reset();

              Kokkos::RangePolicy<HostSpaceType,ScheduleType> policy(0, N);
              Kokkos::parallel_for
                (policy,
                 KOKKOS_LAMBDA(const int k) {
                  auto aa = Kokkos::subview(a, k, Kokkos::ALL(), Kokkos::ALL());
                  auto bb = Kokkos::subview(b, k, Kokkos::ALL(), Kokkos::ALL());

                  switch (test) {
                  case 0:
                    SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 1:
                    SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::NonUnit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 2:
                    SerialTrsm<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::Unit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 3:
                    SerialTrsm<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  case 4:
                    SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,AlgoTagType>::
                      invoke(1.0, aa, bb);
                    break;
                  }
                });

              HostSpaceType::fence();
              const double t = timer.seconds();
              tmin = std::min(tmin, t);
              tavg += (iter >= 0)*t;
            }
            tavg /= iter_end;

            double diff = 0;
            for (int i=0;i<bref.dimension(0);++i)
              for (int j=0;j<bref.dimension(1);++j)
                for (int k=0;k<bref.dimension(2);++k)
                  diff += std::abs(bref(i,j,k) - b(i/VectorLength,j,k)[i%VectorLength]);

            std::cout << std::setw(10) << "KK Vector"
                      << " BlkSize = " << std::setw(3) << BlkSize
                      << " NumCols = " << std::setw(3) << NumCols
                      << " time = " << std::scientific << tmin
                      << " avg flop/s = " << (flop/tavg)
                      << " max flop/s = " << (flop/tmin)
                      << " diff to ref = " << diff
                      << std::endl;
          }
        }
        std::cout << "\n\n";
      }
    }
  }
}

