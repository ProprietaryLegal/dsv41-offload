// KX-GROUPED standalone kernels. CUTLASS headers retain their upstream BSD license.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

using at::Tensor;
using Coord = cutlass::gemm::GemmCoord;

template<class T> using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
 T,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,8,
 T,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,8,
 float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
 cutlass::gemm::GemmShape<32,64,32>,cutlass::gemm::GemmShape<32,32,32>,
 cutlass::gemm::GemmShape<16,8,16>,
 cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
 cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,3,
 cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
template<class T> using Gemm = cutlass::gemm::device::GemmGrouped<Kernel<T>>;

static void tensors(const std::vector<Tensor>& ts) {
 TORCH_CHECK(!ts.empty(), "empty argument list");
 for (auto& t:ts) TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==ts[0].device(),
                            "all tensors must be contiguous on the same CUDA device");
}
static cudaStream_t stream(const Tensor& t) {return at::cuda::getCurrentCUDAStream(t.get_device());}

// Only called at setup, outside capture; never used to clear/recover a poisoned context.
int blocks(Tensor x) {
 tensors({x}); c10::cuda::CUDAGuard guard(x.device());
 int b=x.scalar_type()==at::kHalf ? Gemm<cutlass::half_t>::sufficient() :
                                 Gemm<cutlass::bfloat16_t>::sufficient();
 TORCH_CHECK(b>0,"CUTLASS reported no resident blocks"); return b;
}
template<class T> void launch_gemm(Tensor sizes,Tensor ptr,Tensor ld,int groups,int nblocks,cudaStream_t s) {
 Gemm<T> op;
 auto p=reinterpret_cast<void**>(ptr.data_ptr<int64_t>());
 auto l=ld.data_ptr<int64_t>();
 typename Gemm<T>::Arguments a(reinterpret_cast<Coord*>(sizes.data_ptr<int>()),groups,nblocks,
  {1.0f,0.0f},reinterpret_cast<T**>(p),reinterpret_cast<T**>(p+groups),
  reinterpret_cast<float**>(p+groups*2),reinterpret_cast<float**>(p+groups*3),
  l,l+groups,l+2*groups,l+3*groups,nullptr);
 auto st=op.initialize(a,nullptr,s);
 TORCH_CHECK(st==cutlass::Status::kSuccess,"CUTLASS initialize failed: ",cutlassGetStatusString(st));
 st=op.run(s);
 TORCH_CHECK(st==cutlass::Status::kSuccess,"CUTLASS run failed: ",cutlassGetStatusString(st));
}
void gemm(Tensor x,Tensor sizes,Tensor ptr,Tensor ld,int nblocks) {
 tensors({x,sizes,ptr,ld}); c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(sizes.scalar_type()==at::kInt && ptr.scalar_type()==at::kLong && ld.scalar_type()==at::kLong,
             "bad metadata dtype");
 int g=sizes.size(0);
 TORCH_CHECK(sizes.dim()==2 && sizes.size(1)==3 && ptr.numel()==4*g && ld.numel()==4*g && nblocks>0,
             "bad GEMM metadata shape");
 if(x.scalar_type()==at::kHalf) launch_gemm<cutlass::half_t>(sizes,ptr,ld,g,nblocks,stream(x));
 else {TORCH_CHECK(x.scalar_type()==at::kBFloat16,"only BF16/FP16 supported");
       launch_gemm<cutlass::bfloat16_t>(sizes,ptr,ld,g,nblocks,stream(x));}
}

template<class T> __global__ void route_meta(const T* x,const int* ids,const T* w1,const T* w3,const T* w2,
 T* sx,int* inverse,int* mapping,int* counts,int* offsets,int* status,
 Coord* gu,Coord* dn,int64_t* pg,int64_t* pd,int64_t* lg,int64_t* ld,
 float* g,float* u,T* act,float* d,int E,int R,int H,int J,int top,int G) {
 __shared__ int c[128];
 int e=threadIdx.x;
 if(e==0) { *status=0; for(int r=0;r<R;++r) if(ids[r]<0||ids[r]>=E) *status=1; }
 int n=0;
 if(e<E) {for(int r=0;r<R;++r) n+=ids[r]==e; c[e]=n; counts[e]=n;}
 __syncthreads();
 // Empty entries are well-formed zero-M GEMMs. Two phases avoid metadata write races.
 for(int a=e;a<2*G;a+=blockDim.x) {
  gu[a]=Coord(0,J,H); pg[a]=(int64_t)sx; pg[a+2*G]=(int64_t)w1;
  pg[a+4*G]=pg[a+6*G]=(int64_t)g;
  lg[a]=H; lg[a+2*G]=H; lg[a+4*G]=lg[a+6*G]=J;
 }
 for(int a=e;a<G;a+=blockDim.x) {
  dn[a]=Coord(0,H,J); pd[a]=(int64_t)act;pd[a+G]=(int64_t)w2;
  pd[a+2*G]=pd[a+3*G]=(int64_t)d;
  ld[a]=J;ld[a+G]=J;ld[a+2*G]=ld[a+3*G]=H;
 }
 __syncthreads();
 if(e>=E) return;
 int off=0,group=0;for(int i=0;i<e;++i){off+=c[i];group+=c[i]>0;} offsets[e]=off;
 if(n==0) return;
 int j=off;
 for(int r=0;r<R;++r) if(ids[r]==e) {inverse[j]=r;mapping[r]=j++;}
 for(int proj=0;proj<2;++proj) {
  int a=2*group+proj;
  gu[a]=Coord(n,J,H);pg[a]=(int64_t)(sx+(int64_t)off*H);
  pg[a+2*G]=(int64_t)((proj?w3:w1)+(int64_t)e*J*H);
  pg[a+4*G]=pg[a+6*G]=(int64_t)((proj?u:g)+(int64_t)off*J);
 }
 dn[group]=Coord(n,H,J);pd[group]=(int64_t)(act+(int64_t)off*J);
 pd[group+G]=(int64_t)(w2+(int64_t)e*H*J);
 pd[group+2*G]=pd[group+3*G]=(int64_t)(d+(int64_t)off*H);
}
template<class T> __global__ void gather_x(const T* x,T* sx,const int* inverse,const int* status,int R,int H,int top) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<R*H) sx[i]=*status?T(0):x[(inverse[i/H]/top)*H+i%H];
}
void prepare(Tensor x,Tensor ids,Tensor w1,Tensor w3,Tensor w2,std::vector<Tensor> s) {
 // s: sorted_x,inverse,mapping,counts,offsets,status,gu_sizes,dn_sizes,gu_ptr,dn_ptr,gu_ld,dn_ld,g,u,act,d
 std::vector<Tensor> all={x,ids,w1,w3,w2};all.insert(all.end(),s.begin(),s.end());tensors(all);
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(s.size()==16,"expected 16 workspace tensors");
 TORCH_CHECK(ids.scalar_type()==at::kInt && w1.scalar_type()==x.scalar_type() &&
             w3.scalar_type()==x.scalar_type() && w2.scalar_type()==x.scalar_type(),"bad expert dtypes");
 int E=w1.size(0),H=x.size(1),J=w1.size(1),R=ids.numel(),G=std::min(E,R),top=ids.size(1);
 TORCH_CHECK(E>0&&E<=128&&ids.size(0)==x.size(0)&&w1.size(2)==H&&H%8==0&&J%8==0,
             "unsupported expert shape");
 #define PREP(T) route_meta<T><<<1,128,0,stream(x)>>>(reinterpret_cast<T*>(x.data_ptr()),ids.data_ptr<int>(), \
 reinterpret_cast<T*>(w1.data_ptr()),reinterpret_cast<T*>(w3.data_ptr()),reinterpret_cast<T*>(w2.data_ptr()), \
 reinterpret_cast<T*>(s[0].data_ptr()),s[1].data_ptr<int>(),s[2].data_ptr<int>(),s[3].data_ptr<int>(), \
 s[4].data_ptr<int>(),s[5].data_ptr<int>(),reinterpret_cast<Coord*>(s[6].data_ptr()), \
 reinterpret_cast<Coord*>(s[7].data_ptr()),s[8].data_ptr<int64_t>(),s[9].data_ptr<int64_t>(), \
 s[10].data_ptr<int64_t>(),s[11].data_ptr<int64_t>(),s[12].data_ptr<float>(),s[13].data_ptr<float>(), \
 reinterpret_cast<T*>(s[14].data_ptr()),s[15].data_ptr<float>(),E,R,H,J,top,G); \
 gather_x<T><<<(R*H+255)/256,256,0,stream(x)>>>(reinterpret_cast<T*>(x.data_ptr()), \
 reinterpret_cast<T*>(s[0].data_ptr()),s[1].data_ptr<int>(),s[5].data_ptr<int>(),R,H,top)
 if(x.scalar_type()==at::kHalf){PREP(cutlass::half_t);}else{PREP(cutlass::bfloat16_t);}
 #undef PREP
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<class T> __global__ void activation(const float* g,const float* u,T* a,int n,float limit) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=fminf(g[i],limit),z=fminf(fmaxf(u[i],-limit),limit);
 a[i]=T((isfinite(g[i])&&isfinite(u[i]))?((v/(1.0f+expf(-v)))*z):NAN);}
}
void activate(Tensor g,Tensor u,Tensor a,double limit) {
 tensors({g,u,a});c10::cuda::CUDAGuard guard(g.device());int n=g.numel();
 if(a.scalar_type()==at::kHalf)activation<cutlass::half_t><<<(n+255)/256,256,0,stream(g)>>>(g.data_ptr<float>(),u.data_ptr<float>(),reinterpret_cast<cutlass::half_t*>(a.data_ptr()),n,limit);
 else activation<cutlass::bfloat16_t><<<(n+255)/256,256,0,stream(g)>>>(g.data_ptr<float>(),u.data_ptr<float>(),reinterpret_cast<cutlass::bfloat16_t*>(a.data_ptr()),n,limit);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void combine_k(const float* d,const int* map,const float* weights,const int* status,float* out,int T,int H,int top) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<T*H){if(*status){out[i]=NAN;return;}float v=0;
 for(int s=0;s<top;++s){int r=i/H*top+s;v=__fadd_rn(v,__fmul_rn(d[(int64_t)map[r]*H+i%H],weights[r]));}out[i]=v;}
}
void combine(Tensor d,Tensor map,Tensor weights,Tensor status,Tensor out) {
 tensors({d,map,weights,status,out});c10::cuda::CUDAGuard guard(d.device());int n=out.numel();
 combine_k<<<(n+255)/256,256,0,stream(d)>>>(d.data_ptr<float>(),map.data_ptr<int>(),weights.data_ptr<float>(),status.data_ptr<int>(),out.data_ptr<float>(),out.size(0),out.size(1),weights.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// One warp/output, a single FP32 reduction tree. No TF32 or half accumulator.
template<class T,bool Pair> __global__ void small_k(const T* x,const T* w1,const T* w3,const int* ids,
 float* g,float* u,int R,int K,int N,int E,int top) {
 int r=blockIdx.y,col=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(col>=N)return;
 int e=ids[r];if(e<0||e>=E){if(lane==0){g[r*N+col]=NAN;if(Pair)u[r*N+col]=NAN;}return;}
 int row=Pair?r/top:r;
 float a=0,b=0;for(int k=lane;k<K;k+=32){float v=float(x[(int64_t)row*K+k]);
 a=fmaf(v,float(w1[((int64_t)e*N+col)*K+k]),a);
 if(Pair)b=fmaf(v,float(w3[((int64_t)e*N+col)*K+k]),b);}
 for(int d=16;d>0;d/=2){a+=__shfl_down_sync(0xffffffff,a,d);if(Pair)b+=__shfl_down_sync(0xffffffff,b,d);}
 if(lane==0){g[r*N+col]=a;if(Pair)u[r*N+col]=b;}
}
void small(Tensor x,Tensor w1,Tensor w3,Tensor ids,Tensor g,Tensor u,bool pair) {
 tensors({x,w1,w3,ids,g,u});c10::cuda::CUDAGuard guard(x.device());int R=ids.numel(),K=x.size(-1),N=w1.size(1),E=w1.size(0),top=ids.size(1);
 dim3 grid((N+7)/8,R);auto s=stream(x);
 #define SMALL(T,P) small_k<T,P><<<grid,256,0,s>>>(reinterpret_cast<T*>(x.data_ptr()),reinterpret_cast<T*>(w1.data_ptr()),reinterpret_cast<T*>(w3.data_ptr()),ids.data_ptr<int>(),g.data_ptr<float>(),u.data_ptr<float>(),R,K,N,E,top)
 if(x.scalar_type()==at::kHalf){if(pair){SMALL(cutlass::half_t,true);}else{SMALL(cutlass::half_t,false);}}
 else {if(pair){SMALL(cutlass::bfloat16_t,true);}else{SMALL(cutlass::bfloat16_t,false);}}
 #undef SMALL
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<class T> __global__ void dense_meta(const T* x,const T* w,float* out,Coord* size,int64_t* p,int64_t* ld,int G,int M,int N,int K,bool strided) {
 int g=threadIdx.x+blockIdx.x*blockDim.x;if(g<G){size[g]=Coord(M,N,K);p[g]=(int64_t)(x+(int64_t)g*(strided?K:M*K));
 p[g+G]=(int64_t)(w+(int64_t)g*N*K);p[g+2*G]=p[g+3*G]=(int64_t)(out+(int64_t)g*(strided?N:M*N));
 ld[g]=strided?G*K:K;ld[g+G]=K;ld[g+2*G]=ld[g+3*G]=strided?G*N:N;}
}
void dense_prepare(Tensor x,Tensor w,Tensor out,Tensor sizes,Tensor ptr,Tensor ld,bool strided) {
 tensors({x,w,out,sizes,ptr,ld});c10::cuda::CUDAGuard guard(x.device());int G=w.size(0),M=strided?x.size(0):x.size(1),N=w.size(1),K=w.size(2);
 #define DENSE(T) dense_meta<T><<<(G+127)/128,128,0,stream(x)>>>(reinterpret_cast<T*>(x.data_ptr()),reinterpret_cast<T*>(w.data_ptr()),out.data_ptr<float>(),reinterpret_cast<Coord*>(sizes.data_ptr()),ptr.data_ptr<int64_t>(),ld.data_ptr<int64_t>(),G,M,N,K,strided)
 if(x.scalar_type()==at::kHalf){DENSE(cutlass::half_t);}else{DENSE(cutlass::bfloat16_t);}
 #undef DENSE
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Native head: FP32 base + FP32 Markov, then FP32 comparison.
// BF16 boundaries are available only through the explicit engine control.
template<bool EngineBf16Control> __global__ void head_k(const float* base,const float* bias,float* vals,int64_t* ids,int V,int offset) {
 int row=blockIdx.y,chunk=blockIdx.x;float best=-INFINITY;int winner=INT_MAX;
 for(int j=chunk*1024+threadIdx.x;j<min(V,(chunk+1)*1024);j+=blockDim.x){
  float a=base[row*V+j],b=bias[row*V+j];
  if(EngineBf16Control){a=__bfloat162float(__float2bfloat16_rn(a));b=__bfloat162float(__float2bfloat16_rn(b));}
  float v=__fadd_rn(a,b);
  if(EngineBf16Control)v=__bfloat162float(__float2bfloat16_rn(v));
  if(!isfinite(v))v=NAN;
  if(isnan(v)){best=NAN;winner=min(winner,j);}else if(!isnan(best)&&(v>best||(v==best&&j<winner))){best=v;winner=j;}
 }
 __shared__ float vs[256];__shared__ int ix[256];vs[threadIdx.x]=best;ix[threadIdx.x]=winner;__syncthreads();
 for(int d=128;d;d/=2){if(threadIdx.x<d){float b=vs[threadIdx.x+d];int j=ix[threadIdx.x+d];
 if((isnan(b)&&!isnan(vs[threadIdx.x]))||(!isnan(vs[threadIdx.x])&&(b>vs[threadIdx.x]||(b==vs[threadIdx.x]&&j<ix[threadIdx.x])))){
 vs[threadIdx.x]=b;ix[threadIdx.x]=j;}}__syncthreads();}
 if(threadIdx.x==0){int i=row*gridDim.x+chunk;vals[i]=vs[0];ids[i]=ix[0]+offset;}
}
template<bool EngineBf16Control> void head(Tensor base,Tensor bias,Tensor vals,Tensor ids,int offset) {
 tensors({base,bias,vals,ids});c10::cuda::CUDAGuard guard(base.device());int V=base.size(1);
 head_k<EngineBf16Control><<<dim3((V+1023)/1024,base.size(0)),256,0,stream(base)>>>(base.data_ptr<float>(),bias.data_ptr<float>(),vals.data_ptr<float>(),ids.data_ptr<int64_t>(),V,offset);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void small_prepare_k(const int* ids,int* mapping,int* inverse,int* status,int R,int E) {
 if(threadIdx.x==0){*status=0;for(int r=0;r<R;++r){mapping[r]=inverse[r]=r;if(ids[r]<0||ids[r]>=E)*status=1;}}
}
void small_prepare(Tensor ids,Tensor mapping,Tensor inverse,Tensor status,int E) {
 tensors({ids,mapping,inverse,status});c10::cuda::CUDAGuard guard(ids.device());
 small_prepare_k<<<1,32,0,stream(ids)>>>(ids.data_ptr<int>(),mapping.data_ptr<int>(),inverse.data_ptr<int>(),status.data_ptr<int>(),ids.numel(),E);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void head_reduce_k(const float* vals,const int64_t* ids,float* out,int64_t* oi,int chunks) {
 int row=blockIdx.x;float best=-INFINITY;int64_t id=INT64_MAX;
 for(int c=0;c<chunks;++c){float v=vals[row*chunks+c];int64_t j=ids[row*chunks+c];
 if(isnan(v)){best=NAN;id=-1;break;}if(v>best||(v==best&&j<id)){best=v;id=j;}}
 out[row]=best;oi[row]=id;
}
void head_reduce(Tensor vals,Tensor ids,Tensor out,Tensor oi) {
 tensors({vals,ids,out,oi});c10::cuda::CUDAGuard guard(vals.device());
 head_reduce_k<<<vals.size(0),1,0,stream(vals)>>>(vals.data_ptr<float>(),ids.data_ptr<int64_t>(),out.data_ptr<float>(),oi.data_ptr<int64_t>(),vals.size(1));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Software E4M3 QDQ: SM86 has no native FP8 tensor/conversion requirement.
__device__ float round_bf16(float v) {return __bfloat162float(__float2bfloat16_rn(v));}
__device__ float e4m3(float v) {
 float a=fabsf(v); if(a==0)return v;
 int exponent;frexpf(a,&exponent);
 float step=ldexpf(1.0f,max(-9,exponent-4));
 float q=nearbyintf(a/step)*step;return copysignf(fminf(q,448.0f),v);
}
template<class T> __global__ void qdq_k(const T* x,T* out,int N) {
 int i=(blockIdx.x*8+threadIdx.x/32)*32+threadIdx.x%32;
 if(i>=N)return; float v=float(x[i]),a=fmaxf(fabsf(v),1.e-4f);
 for(int d=16;d;d/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
 // ceil(log2) using bits avoids log approximation at exact power-of-two boundaries.
 float z=a*(1.0f/448.0f);int bits=__float_as_int(z);
 int exponent=((bits>>23)&255)-127+((bits&0x7fffff)!=0);
 float scale=ldexpf(1.0f,exponent);
 out[i]=T(isfinite(v)?(e4m3(v/scale)*scale):NAN);
}
void qdq(Tensor x,Tensor out) {
 tensors({x,out});c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.numel()==out.numel()&&x.size(-1)%32==0&&x.scalar_type()==out.scalar_type(),"QDQ requires equal type/shape and 32-wide groups");
 int n=x.numel();
 if(x.scalar_type()==at::kHalf)qdq_k<cutlass::half_t><<<(n+255)/256,256,0,stream(x)>>>(reinterpret_cast<cutlass::half_t*>(x.data_ptr()),reinterpret_cast<cutlass::half_t*>(out.data_ptr()),n);
 else qdq_k<cutlass::bfloat16_t><<<(n+255)/256,256,0,stream(x)>>>(reinterpret_cast<cutlass::bfloat16_t*>(x.data_ptr()),reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),n);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<class T> __global__ void native_act_k(const float* g,const float* u,const int* inverse,const float* rw,T* out,int n,int J,float limit) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=fminf(round_bf16(g[i]),limit),z=fminf(fmaxf(round_bf16(u[i]),-limit),limit);
 float a=__fmul_rn(__fmul_rn(v/(1.0f+expf(-v)),z),rw[inverse[i/J]]);
 out[i]=T((isfinite(g[i])&&isfinite(u[i])&&isfinite(rw[inverse[i/J]]))?round_bf16(a):NAN);}
}
void native_activate(Tensor g,Tensor u,Tensor inverse,Tensor weights,Tensor out,double limit) {
 tensors({g,u,inverse,weights,out});c10::cuda::CUDAGuard guard(g.device());int n=g.numel(),J=g.size(1);
 #define NACT(T) native_act_k<T><<<(n+255)/256,256,0,stream(g)>>>(g.data_ptr<float>(),u.data_ptr<float>(),inverse.data_ptr<int>(),weights.data_ptr<float>(),reinterpret_cast<T*>(out.data_ptr()),n,J,limit)
 if(out.scalar_type()==at::kHalf){NACT(cutlass::half_t);}else{NACT(cutlass::bfloat16_t);}
 #undef NACT
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void native_combine_k(const float* d,const int* ids,const int* map,const int* status,float* out,int T,int H,bool round_down) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<T*H){if(*status){out[i]=NAN;return;}int r=i/H*3;int order[3]={0,1,2};
 // Official MoE loops through experts in ascending ID. Duplicate IDs are not generated by top-k.
 for(int a=0;a<2;++a)for(int b=a+1;b<3;++b)if(ids[r+order[a]]>ids[r+order[b]]){int z=order[a];order[a]=order[b];order[b]=z;}
 float v=0;for(int s=0;s<3;++s){float z=d[(int64_t)map[r+order[s]]*H+i%H];v=__fadd_rn(v,round_down?round_bf16(z):z);}out[i]=v;}
}
void native_combine(Tensor d,Tensor ids,Tensor map,Tensor status,Tensor out,bool round_down) {
 tensors({d,ids,map,status,out});c10::cuda::CUDAGuard guard(d.device());int n=out.numel();
 TORCH_CHECK(ids.size(1)==3,"native combine is top-3");
 native_combine_k<<<(n+255)/256,256,0,stream(d)>>>(d.data_ptr<float>(),ids.data_ptr<int>(),map.data_ptr<int>(),status.data_ptr<int>(),out.data_ptr<float>(),out.size(0),out.size(1),round_down);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void slot_scatter_k(const float* down,const int* map,const int* status,float* slots,int R,int H) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<R*H)slots[i]=*status?NAN:down[(int64_t)map[i/H]*H+i%H];
}
void slot_scatter(Tensor down,Tensor mapping,Tensor status,Tensor out) {
 tensors({down,mapping,status,out});c10::cuda::CUDAGuard guard(down.device());int n=out.numel();
 slot_scatter_k<<<(n+255)/256,256,0,stream(down)>>>(down.data_ptr<float>(),mapping.data_ptr<int>(),status.data_ptr<int>(),out.data_ptr<float>(),mapping.numel(),down.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("small_prepare",&small_prepare);m.def("head_reduce",&head_reduce);m.def("qdq",&qdq);m.def("native_activate",&native_activate);m.def("native_combine",&native_combine);m.def("slot_scatter",&slot_scatter);m.def("blocks",&blocks);m.def("gemm",&gemm);m.def("prepare",&prepare);m.def("activate",&activate);m.def("combine",&combine);m.def("small",&small);m.def("dense_prepare",&dense_prepare);m.def("head",&head<false>);m.def("head_engine_bf16_control",&head<true>);}
