/*
 * Tensor-core betweenness centrality for unweighted graphs.
 *
 * The CUDA-core kernels in bcgpugcsc_sc.cu compute f <- A'f, a sparse
 * matrix-VECTOR product.  Tensor cores multiply matrices, not vectors, so this
 * file runs TCK roots at once: F is n x TCK and the BFS step becomes
 * FT <- A'F, a sparse matrix-matrix product.  TCK is 16 because that is the
 * N dimension of the m16n16k8 tf32 WMMA shape.
 *
 * Precision.  Path counts reach 24973 on fb_artist, so fp16 is unusable twice
 * over: its exact-integer range stops at 2048 and sigma on some graphs would
 * overflow its 65504 maximum outright.  tf32 has fp32's exponent range, which
 * removes the overflow, but still carries only an 11-bit mantissa.  So every
 * F block is split as F = F1 + F2 + F3 with each term exact in tf32, and three
 * MMAs are issued per tile.  33 mantissa bits > fp32's 24, so the product is
 * exact to fp32.  The adjacency tile itself is 0/1 and needs no splitting.
 * The three extra MMAs are free: this kernel is memory-bound, not compute-bound.
 *
 * Blocking.  A' is cut into row-windows of 16 vertices.  A geometric 16x16
 * tiling is useless here: on fb_artist a nonzero tile holds 1.12 nonzeros, so
 * loading the 16 consecutive columns of F that the tile spans wastes 93% of
 * the bandwidth.  Instead each row-window keeps the sorted list of source
 * vertices its 16 rows actually reference and cuts THAT list into chunks of
 * 16.  Every column of a chunk is then guaranteed to carry a nonzero, which
 * raises the density the MMA sees from 0.4%-1.8% to at least 6.25% and drops
 * the F traffic by the same factor.  Tiles are still never stored dense --
 * only their nonzeros are, and each warp scatters them into a shared-memory
 * tile just before the MMA.  One warp owns one row-window and accumulates
 * across it in a single fragment, so no atomics are needed on the output.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <mma.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "bcgpug.cuh"
extern "C" {
#include "bcug.h"
}
using namespace nvcuda;

#define TCK             16   /* roots per batch == wmma N */
#define TC_TILE         256  /* 16x16 */
#define TC_WARPS 4

/*****************************************************************************/
/* FT <- A'F over one row-window, accumulated in a single tf32 fragment.      */
__global__ void tcSpMMKernel (const int *__restrict__ browptr,
                              const int *__restrict__ chunkcol,
                              const int *__restrict__ bentptr,
                              const unsigned char *__restrict__ bent,
                              const float *__restrict__ F,
                              float *__restrict__ FT,
                              int nbrows,int nsplit){

  int wid  = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int brow = blockIdx.x * TC_WARPS + wid;

  extern __shared__ float sh[];
  float *At = sh + wid * (TC_TILE + 3*TC_TILE);
  float *Fs = At + TC_TILE;

  wmma::fragment<wmma::accumulator,16,16,8,float> acc;
  wmma::fill_fragment(acc,0.0f);

  if (brow < nbrows){
    for (int b = browptr[brow]; b < browptr[brow+1]; b++){

      for (int t = lane; t < TC_TILE; t += 32) At[t] = 0.0f;
      __syncwarp();
      /* bent packs (local_row << 4) | chunk_slot, which is the tile offset */
      for (int e = bentptr[b] + lane; e < bentptr[b+1]; e += 32)
        atomicAdd(&At[bent[e]],1.0f);

      /* slot s of this chunk holds source vertex chunkcol[b*16+s]; lanes that
         share a slot read its 16 consecutive roots, so the gather is a 64-byte
         coalesced load per slot rather than one strided element per lane */
      const int *cc = chunkcol + (size_t)b * 16;
      for (int t = lane; t < TC_TILE; t += 32){
        float x  = F[(size_t)cc[t >> 4] * TCK + (t & 15)];
        float h1 = wmma::__float_to_tf32(x);
        float r1 = x - h1;
        float h2 = wmma::__float_to_tf32(r1);
        Fs[t]           = h1;
        Fs[TC_TILE+t]   = h2;
        Fs[2*TC_TILE+t] = r1 - h2;
      }
      __syncwarp();

      for (int s = 0; s < nsplit; s++){
        for (int kk = 0; kk < 16; kk += 8){
          wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> fb;
          wmma::load_matrix_sync(fa,At + kk,16);
          wmma::load_matrix_sync(fb,Fs + s*TC_TILE + kk*16,16);
          for (int i = 0; i < fa.num_elements; i++) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
          for (int i = 0; i < fb.num_elements; i++) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
          wmma::mma_sync(acc,fa,fb,acc);
        }
      }
      __syncwarp();
    }
    wmma::store_matrix_sync(FT + (size_t)brow*TC_TILE,acc,16,wmma::mem_row_major);
  }
}

/*****************************************************************************/
/* Elementwise stages.  Index i runs over v*TCK + j, vertex v and root slot j.
 * The reference masks visited vertices inside the SpMV; the MMA cannot mask
 * mid-product, so the same test is applied here instead -- equivalent, because
 * a masked-out vertex contributed ft = 0 either way.                          */
__global__ void tcBfsFunctionsKernel (float *F,const float *FT,float *Sigma,
                                      int *S,int *c,int nv,int d){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < nv){
    float ft = (Sigma[i] < 0.01f) ? FT[i] : 0.0f;
    F[i] = 0.0f;
    if (ft > 0.9f){
      *c = 1;
      F[i] = ft;
      Sigma[i] += ft;
      S[i] = d;
    }
  }
}

__global__ void tcDeltaUKernel (const int *S,const float *Delta,float *DeltaU,
                                const float *Sigma,int nv,int d){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < nv){
    DeltaU[i] = 0.0f;
    if (S[i] == d && Sigma[i] > 0.1f)
      DeltaU[i] = (1.0f + Delta[i])/Sigma[i];
  }
}

__global__ void tcDeltaKernel (const int *S,float *Delta,const float *DeltaUT,
                               const float *Sigma,int nv,int d){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < nv && DeltaUT[i] > 0.0f){
    if (S[i] == d-1 && Sigma[i] > 0.9f)
      Delta[i] += DeltaUT[i]*Sigma[i];
  }
}

__global__ void tcBcKernel (float *bc,const float *Delta,const int *roots,
                            int nroots,int n){
  int v = blockIdx.x*blockDim.x + threadIdx.x;
  if (v < n){
    float s = 0.0f;
    for (int j = 0; j < nroots; j++){
      float dv = Delta[(size_t)v*TCK + j];
      if (v != roots[j] && dv > 0.0f) s += dv;
    }
    if (s > 0.0f) bc[v] += s;
  }
}

__global__ void tcInitRootsKernel (float *F,float *Sigma,const int *roots,int nroots){
  int j = threadIdx.x;
  if (j < nroots){
    F    [(size_t)roots[j]*TCK + j] = 1.0f;
    Sigma[(size_t)roots[j]*TCK + j] = 1.0f;
  }
}

/*****************************************************************************/
/* Host side: build the condensed blocking of A' from the CSC arrays, then
 * drive the BFS and delta stages a batch of TCK roots at a time.            */
struct TcEnt { unsigned long long key; unsigned char lrow; };

extern "C"
int bc_gpu_ug_tc (int *I_h,int *CP_h,int *S_h,float *sigma_h,float *bc_h,
                  int nr,int rs,int nz,int n,int repetition){

  int nsplit = 3;
  const char *env = getenv("TURBOBC_TC_SPLIT");
  if (env){ nsplit = atoi(env); if (nsplit < 1) nsplit = 1; if (nsplit > 3) nsplit = 3; }

  int nb   = (n + 15)/16;      /* row-windows of 16 vertices */
  int npad = nb*16;

  /* ---- condensed blocking --------------------------------------------- */
  /* key = (row-window << 32) | source vertex.  Sorting on it groups a
     window's edges together and orders them by source, so the pass below can
     number the distinct sources 0,1,2,... and pack 16 of them per chunk.    */
  std::vector<TcEnt> ents(nz);
  {
    size_t t = 0;
    for (int v = 0; v < n; v++)
      for (int k = CP_h[v]; k < CP_h[v+1]; k++){
        int u = I_h[k];                         /* edge u -> v, so A'[v][u] */
        ents[t].key  = ((unsigned long long)(v>>4) << 32) | (unsigned int)u;
        ents[t].lrow = (unsigned char)(v & 15);
        t++;
      }
  }
  std::sort(ents.begin(),ents.end(),
            [](const TcEnt &a,const TcEnt &b){ return a.key < b.key; });

  std::vector<int>           h_chunkcol, h_bentptr, h_browptr(nb+1,0);
  std::vector<unsigned char> h_bent;
  h_bent.reserve(nz);
  h_bentptr.push_back(0);
  int nblocks = 0;
  {
    size_t i = 0;
    for (int br = 0; br < nb; br++){
      h_browptr[br] = nblocks;
      int slot = 16;                    /* 16 => current chunk is full */
      unsigned int prev = 0;
      int first = 1;
      while (i < (size_t)nz && (int)(ents[i].key >> 32) == br){
        unsigned int u = (unsigned int)(ents[i].key & 0xffffffffu);
        if (first || u != prev){
          if (slot == 16){
            if (nblocks) h_bentptr.push_back((int)h_bent.size());
            /* unused trailing slots stay at vertex 0; their adjacency column
               is zero, so they contribute nothing to the product */
            h_chunkcol.resize(h_chunkcol.size()+16,0);
            nblocks++; slot = 0;
          }
          h_chunkcol[h_chunkcol.size()-16+slot] = (int)u;
          prev = u; first = 0; slot++;
        }
        h_bent.push_back((unsigned char)((ents[i].lrow << 4) | (slot-1)));
        i++;
      }
    }
    h_browptr[nb] = nblocks;
    h_bentptr.push_back((int)h_bent.size());
  }
  printf("bcgputc::bc_gpu_ug_tc:: %d row-windows, %d condensed 16x16 chunks, "
         "%.3f%% tile density, tf32 split = %d\n",
         nb,nblocks,100.0*nz/((double)nblocks*256.0),nsplit);

  /* ---- device memory --------------------------------------------------- */
  int *browptr_d,*chunkcol_d,*bentptr_d,*S_d,*roots_d,*c;
  unsigned char *bent_d;
  float *F_d,*FT_d,*Sigma_d,*Delta_d,*DeltaU_d,*DeltaUT_d,*bc_d;
  size_t velem = (size_t)npad*TCK;

  checkCudaErrors(cudaMalloc(&browptr_d,(nb+1)*sizeof(int)));
  checkCudaErrors(cudaMalloc(&chunkcol_d,(size_t)nblocks*16*sizeof(int)));
  checkCudaErrors(cudaMalloc(&bentptr_d,(nblocks+1)*sizeof(int)));
  checkCudaErrors(cudaMalloc(&bent_d,nz*sizeof(unsigned char)));
  checkCudaErrors(cudaMemcpy(browptr_d,h_browptr.data(),(nb+1)*sizeof(int),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(chunkcol_d,h_chunkcol.data(),(size_t)nblocks*16*sizeof(int),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bentptr_d,h_bentptr.data(),(nblocks+1)*sizeof(int),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bent_d,h_bent.data(),nz*sizeof(unsigned char),cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&F_d,velem*sizeof(float)));
  checkCudaErrors(cudaMalloc(&FT_d,velem*sizeof(float)));
  checkCudaErrors(cudaMalloc(&Sigma_d,velem*sizeof(float)));
  checkCudaErrors(cudaMalloc(&Delta_d,velem*sizeof(float)));
  checkCudaErrors(cudaMalloc(&DeltaU_d,velem*sizeof(float)));
  checkCudaErrors(cudaMalloc(&DeltaUT_d,velem*sizeof(float)));
  checkCudaErrors(cudaMalloc(&S_d,velem*sizeof(int)));
  checkCudaErrors(cudaMalloc(&roots_d,TCK*sizeof(int)));
  checkCudaErrors(cudaMalloc(&bc_d,n*sizeof(float)));
  checkCudaErrors(cudaMemset(bc_d,0,n*sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&c,sizeof(int)));

  int nv       = n*TCK;
  int gridEl   = (nv + 255)/256;
  int gridV    = (n + 255)/256;
  int gridSpMM = (nb + TC_WARPS - 1)/TC_WARPS;
  size_t shmem = (size_t)TC_WARPS*(TC_TILE + 3*TC_TILE)*sizeof(float);

  float t_bfs_spmv=0,t_bfs_spmv_t=0,t_bfsfunctions=0,t_bfsfunctions_t=0;
  float t_delta_u=0,t_delta_u_t=0,t_bc_spmv=0,t_bc_spmv_t=0;
  float t_delta=0,t_delta_t=0,t_bc=0,t_bc_t=0;
  cudaEvent_t start,stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  std::vector<int> h_roots(TCK,0);
  std::vector<float> h_Sigma; std::vector<int> h_S;
  int dmax_report = 0;

  for (int base = 0; base < nr; base += TCK){
    int nroots = (nr - base < TCK) ? (nr - base) : TCK;
    for (int j = 0; j < nroots; j++) h_roots[j] = (nr == 1) ? rs : base + j;
    checkCudaErrors(cudaMemcpy(roots_d,h_roots.data(),TCK*sizeof(int),cudaMemcpyHostToDevice));

    int d = 0;
    for (int rep = 0; rep < repetition; rep++){
      checkCudaErrors(cudaMemset(F_d,0,velem*sizeof(float)));
      checkCudaErrors(cudaMemset(Sigma_d,0,velem*sizeof(float)));
      checkCudaErrors(cudaMemset(Delta_d,0,velem*sizeof(float)));
      checkCudaErrors(cudaMemset(S_d,0,velem*sizeof(int)));
      tcInitRootsKernel<<<1,TCK>>>(F_d,Sigma_d,roots_d,nroots);

      *c = 1; d = 0;
      while (*c){
        d++; *c = 0;
        cudaEventRecord(start);
        tcSpMMKernel<<<gridSpMM,TC_WARPS*32,shmem>>>
          (browptr_d,chunkcol_d,bentptr_d,bent_d,F_d,FT_d,nb,nsplit);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_bfs_spmv,start,stop); t_bfs_spmv_t += t_bfs_spmv;

        cudaEventRecord(start);
        tcBfsFunctionsKernel<<<gridEl,256>>>(F_d,FT_d,Sigma_d,S_d,c,nv,d);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_bfsfunctions,start,stop); t_bfsfunctions_t += t_bfsfunctions;
      }
      d = d - 1;
      if (d > dmax_report) dmax_report = d;

      while (d > 1){
        cudaEventRecord(start);
        tcDeltaUKernel<<<gridEl,256>>>(S_d,Delta_d,DeltaU_d,Sigma_d,nv,d);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_delta_u,start,stop); t_delta_u_t += t_delta_u;

        cudaEventRecord(start);
        tcSpMMKernel<<<gridSpMM,TC_WARPS*32,shmem>>>
          (browptr_d,chunkcol_d,bentptr_d,bent_d,DeltaU_d,DeltaUT_d,nb,nsplit);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_bc_spmv,start,stop); t_bc_spmv_t += t_bc_spmv;

        cudaEventRecord(start);
        tcDeltaKernel<<<gridEl,256>>>(S_d,Delta_d,DeltaUT_d,Sigma_d,nv,d);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_delta,start,stop); t_delta_t += t_delta;
        d--;
      }
    }
    cudaEventRecord(start);
    tcBcKernel<<<gridV,256>>>(bc_d,Delta_d,roots_d,nroots,n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_bc,start,stop); t_bc_t += t_bc;

    /* the reference leaves sigma and S holding the last root's BFS */
    if (base + nroots >= nr){
      h_Sigma.resize(velem); h_S.resize(velem);
      checkCudaErrors(cudaMemcpy(h_Sigma.data(),Sigma_d,velem*sizeof(float),cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(h_S.data(),S_d,velem*sizeof(int),cudaMemcpyDeviceToHost));
      int jlast = nroots - 1;
      for (int v = 0; v < n; v++){
        sigma_h[v] = h_Sigma[(size_t)v*TCK + jlast];
        S_h[v]     = h_S[(size_t)v*TCK + jlast];
      }
    }
  }
  checkCudaErrors(cudaMemcpy(bc_h,bc_d,n*sizeof(float),cudaMemcpyDeviceToHost));

  float t_bfs_avg   = (t_bfs_spmv_t + t_bfsfunctions_t)/repetition;
  float t_delta_avg = (t_delta_u_t + t_bc_spmv_t + t_delta_t)/repetition;
  float total_BC_t  = t_bfs_avg + t_delta_avg + t_bc_t;

  printf("\nbcgputc::bc_gpu_ug_tc:BFS  d = %d\n",dmax_report);
  printf("bcgputc::bc_gpu_ug_tc:time F <-- A'F (tensor core)  = %lfms \n",t_bfs_spmv_t/repetition);
  printf("bcgputc::bc_gpu_ug_tc:time bfs functions  = %lfms \n",t_bfsfunctions_t/repetition);
  printf("bcgputc::bc_gpu_ug_tc:average time BFS  = %lfms \n",t_bfs_avg);
  printf("bcgputc::bc_gpu_ug_tc:time delta_u <-- (1+delta)/sigma  = %lfms \n",t_delta_u_t/repetition);
  printf("bcgputc::bc_gpu_ug_tc:time delta_ut <-- A'delta_u (tensor core)  = %lfms \n",t_bc_spmv_t/repetition);
  printf("bcgputc::bc_gpu_ug_tc:time delta <-- delta + delta_ut*sigma  = %lfms \n",t_delta_t/repetition);
  printf("bcgputc::bc_gpu_ug_tc:average time delta   = %lfms \n",t_delta_avg);
  printf("bcgputc::bc_gpu_ug_tc:time bc <-- bc + delta = %lfms \n",t_bc_t);
  printf("bcgputc::bc_gpu_ug_tc:total BC time = %lfms \n",total_BC_t);
  printf("bcgputc::bc_gpu_ug_tc:total BC time/vertex = %lfms \n",total_BC_t/nr);

  cudaFree(browptr_d); cudaFree(chunkcol_d); cudaFree(bentptr_d); cudaFree(bent_d);
  cudaFree(F_d); cudaFree(FT_d); cudaFree(Sigma_d); cudaFree(Delta_d);
  cudaFree(DeltaU_d); cudaFree(DeltaUT_d); cudaFree(S_d); cudaFree(roots_d);
  cudaFree(bc_d); cudaFree(c);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return 0;
}
