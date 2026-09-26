// Included in gcn_cuda.cu's anonymous namespace.
// One edge-feature contribution per thread. Atomics balance skewed rows at
// the cost of contention and floating-point summation order variability.
__global__ void aggregate_edge_feature_kernel(
    const int* rows, const int* columns, const int* destinations,
    const float* input, float* output, std::size_t entries,
    std::size_t features, int mode) {
  const std::size_t first=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const std::size_t stride=static_cast<std::size_t>(gridDim.x)*blockDim.x;
  for(std::size_t i=first;i<entries*features;i+=stride){
    const auto e=i/features;const auto f=i%features;
    const int v=destinations[e],u=columns[e];
    float scale=1.0F;
    if(mode==static_cast<int>(Aggregation::Mean)) scale=1.0F/float(rows[v+1]-rows[v]);
    else if(mode==static_cast<int>(Aggregation::SymmetricGcn)){
      const int du=rows[u+1]-rows[u];
      scale=du?rsqrtf(float(rows[v+1]-rows[v])*float(du)):0.0F;
    }
    atomicAdd(output+static_cast<std::size_t>(v)*features+f,scale*input[static_cast<std::size_t>(u)*features+f]);
  }
}

// A 16x16 tile reuses feature/weight values in shared memory. Every thread
// reaches each barrier, including partial tiles; large N uses a grid stride.
__global__ void dense_tiled_kernel(const float* input,const float* weights,
    float* output,std::size_t vertices,std::size_t fin,std::size_t fout,int activation){
  __shared__ float a[16][16];
  __shared__ float b[16][16];
  const unsigned tx=threadIdx.x,ty=threadIdx.y;
  for(std::size_t tile_row=blockIdx.y;tile_row<(vertices+15)/16;tile_row+=gridDim.y){
    const auto row=tile_row*16+ty;
    for(std::size_t tile_col=blockIdx.x;tile_col<(fout+15)/16;tile_col+=gridDim.x){
      const auto col=tile_col*16+tx;
      float sum=0;
      for(std::size_t k=0;k<fin;k+=16){
        a[ty][tx]=(row<vertices && k+tx<fin)?input[row*fin+k+tx]:0;
        b[ty][tx]=(k+ty<fin && col<fout)?weights[(k+ty)*fout+col]:0;
        __syncthreads();
        #pragma unroll
        for(unsigned j=0;j<16;++j)sum+=a[ty][j]*b[j][tx];
        __syncthreads();
      }
      if(row<vertices && col<fout)output[row*fout+col]=activation==static_cast<int>(Activation::Relu)?fmaxf(0.0F,sum):sum;
    }
  }
}
