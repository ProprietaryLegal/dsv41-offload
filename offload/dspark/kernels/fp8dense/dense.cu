// SPDX-License-Identifier: Apache-2.0
// MMA operand mapping follows kernels/mxfp4/regfed.cu (Strata attribution there).
// FP8 decoder, dense layouts, split-K and grouped addressing are original.
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>
using u32=uint32_t;
using u64=uint64_t;
namespace py=pybind11;
static void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
static cudaStream_t stream(u64 p) {return reinterpret_cast<cudaStream_t>(p);}

__device__ __forceinline__ u32 scalar_decode(u32 c,int scale) {
    const u32 mag=c&127;
    u32 bits=(mag<<4)+0x3c00;
    if(mag<8) bits=((__byte_perm(0x3b3b3b00U,0x3c3c3c3cU,mag)&255)<<8)
                        |(__byte_perm(0xc0800000U,0x60402000U,mag)&255);
    if(mag) bits+=((scale-127)*128);
    return bits|((c&128)<<8);
}
__device__ __forceinline__ u32 pair_decode_fast(u32 c,int scale) {
    // E4M3's exponent/mantissa bits, spread into BF16 fields without bias,
    // represent code * 2^-120, INCLUDING E4M3 subnormal codes. One exact
    // BF16x2 multiplication supplies both the E4M3 bias and E8M0 exponent.
    // The multiplier is finite for scale <=134 (all native dense scales).
    u32 p=__byte_perm(c,0,0x4140U);
    u32 raw=((p&0x007f007fU)<<4)|((p&0x00800080U)<<8);
    u32 sb=static_cast<u32>((scale+120)<<7)*0x00010001U;
    __nv_bfloat162 v=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&raw),
                            *reinterpret_cast<const __nv_bfloat162*>(&sb));
    return *reinterpret_cast<const u32*>(&v);
}
__device__ __forceinline__ u32 pair_decode(u32 c,int scale) {
    if(scale<=134) return pair_decode_fast(c,scale);
    return scalar_decode(c&255,scale)|(scalar_decode((c>>8)&255,scale)<<16);
}
__device__ __forceinline__ void mma(float (&d)[4], const u32 (&a)[4],u32 b0,u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__global__ void prepack_kernel(uint2* dst,const unsigned char* src,int n,int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int kt=k/16;
    if(i>=(n/16)*kt*32) return;
    int lane=i&31,tile=(i>>5)%kt,nt=(i>>5)/kt,g=lane>>2,t=lane&3;
    u32 words[2]={0,0};
    #pragma unroll
    for(int j=0;j<4;++j) {
        int row=nt*16+g+((j&1)?8:0),col=tile*16+t*2+((j>=2)?8:0);
        const auto* p=src+static_cast<size_t>(row)*k+col;
        words[j/2]|=static_cast<u32>(*reinterpret_cast<const uint16_t*>(p))<<((j%2)*16);
    }
    dst[i]=make_uint2(words[0],words[1]);
}

// One warp computes N16 x M8. Split-K adds independent work without atomics;
// the caller's FP32 scratch is reduced in a fixed ascending order.
template<int ROW_TILES,bool F32,bool SPLIT,bool FAST>
__global__ __launch_bounds__(128) void dense_kernel(
    const __nv_bfloat16* x,const uint2* w,const unsigned char* scales,void* output,
    int m,int n,int k,int xs,int ys,int groups,int splits,int split_shift,int chunk) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int nt=blockIdx.x*4+warp;
    if(nt>=n/16) return;
    int mt=blockIdx.y*8*ROW_TILES;
    int split,gid;
    if(split_shift>=0) {split=blockIdx.z&(splits-1);gid=blockIdx.z>>split_shift;}
    else {split=blockIdx.z%splits;gid=blockIdx.z/splits;}
    int g=lane>>2,t=lane&3;
    float acc[ROW_TILES][4]={};
    int tiles=k/16,start=chunk*split,end=min(start+chunk,tiles);
    const size_t wb=(static_cast<size_t>(gid)*(n/16)+nt)*tiles*32;
    for(int tile=start;tile<end;++tile) {
        uint2 words=w[wb+tile*32+lane];
        int sc=scales[(static_cast<size_t>(gid)*(n/32)+nt/2)*(k/32)+tile/2];
        u32 a[4];
        if constexpr(FAST) {
            a[0]=pair_decode_fast(words.x,sc);a[1]=pair_decode_fast(words.x>>16,sc);
            a[2]=pair_decode_fast(words.y,sc);a[3]=pair_decode_fast(words.y>>16,sc);
        } else {
            a[0]=pair_decode(words.x,sc);a[1]=pair_decode(words.x>>16,sc);
            a[2]=pair_decode(words.y,sc);a[3]=pair_decode(words.y>>16,sc);
        }
        #pragma unroll
        for(int r=0;r<ROW_TILES;++r) {
            int row=mt+r*8+g;
            u32 b0=0,b1=0;
            if(row<m) {
                const auto* p=reinterpret_cast<const u32*>(x+static_cast<size_t>(row)*xs+gid*k+tile*16+t*2);
                b0=p[0];b1=p[4];
            }
            mma(acc[r],a,b0,b1);
        }
    }
    #pragma unroll
    for(int r=0;r<ROW_TILES;++r) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            int row=mt+r*8+t*2+(j&1),col=nt*16+g+((j>=2)?8:0);
            if(row<m) {
                size_t idx;
                if constexpr(SPLIT) idx=((static_cast<size_t>(split)*m+row)*groups+gid)*n+col;
                else idx=static_cast<size_t>(row)*ys+gid*n+col;
                if constexpr(F32 || SPLIT) reinterpret_cast<float*>(output)[idx]=acc[r][j];
                else reinterpret_cast<__nv_bfloat16*>(output)[idx]=__float2bfloat16_rn(acc[r][j]);
            }
        }
    }
}
template<bool F32> __global__ void reduce_kernel(const float* in,void* out,int m,int n,int groups,int splits,int ys) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,total=m*n*groups;
    if(i>=total) return;
    float sum=in[i];
    for(int s=1;s<splits;++s) sum+=in[static_cast<size_t>(s)*total+i];
    int row=i/(n*groups),col=i%(n*groups);
    if constexpr(F32) reinterpret_cast<float*>(out)[static_cast<size_t>(row)*ys+col]=sum;
    else reinterpret_cast<__nv_bfloat16*>(out)[static_cast<size_t>(row)*ys+col]=__float2bfloat16_rn(sum);
}
__global__ void decoder_kernel(const unsigned char* codes,const unsigned char* scales,uint16_t* out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) out[i]=pair_decode(codes[i]*0x101U,scales[i]);
}
static void prepack(u64 dst,u64 src,int n,int k,u64 st) {
    if(n<=0||n%32||k<=0||k%32) throw std::runtime_error("prepack requires N,K positive multiples of 32");
    prepack_kernel<<<((n/16)*(k/16)*32+255)/256,256,0,stream(st)>>>(
        reinterpret_cast<uint2*>(dst),reinterpret_cast<const unsigned char*>(src),n,k);
    check(cudaGetLastError());
}
template<int R,bool F32,bool SPLIT,bool FAST> static void launch_t(u64 x,u64 w,u64 s,u64 y,int m,int n,int k,int xs,int ys,int groups,int splits,u64 st) {
    dim3 grid((n/16+3)/4,(m+8*R-1)/(8*R),groups*splits);
    dense_kernel<R,F32,SPLIT,FAST><<<grid,128,0,stream(st)>>>(reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const uint2*>(w),reinterpret_cast<const unsigned char*>(s),reinterpret_cast<void*>(y),
        m,n,k,xs,ys,groups,splits,(splits&(splits-1))?-1:__builtin_ctz(static_cast<unsigned>(splits)),(k/16+splits-1)/splits);
}
template<bool F32,bool SPLIT,bool FAST> static void launch_r(u64 x,u64 w,u64 s,u64 y,int m,int n,int k,int xs,int ys,int groups,int splits,int rows,u64 st) {
    switch(rows) {
    case 1: launch_t<1,F32,SPLIT,FAST>(x,w,s,y,m,n,k,xs,ys,groups,splits,st);break;
    case 2: launch_t<2,F32,SPLIT,FAST>(x,w,s,y,m,n,k,xs,ys,groups,splits,st);break;
    case 4: launch_t<4,F32,SPLIT,FAST>(x,w,s,y,m,n,k,xs,ys,groups,splits,st);break;
    case 8: launch_t<8,F32,SPLIT,FAST>(x,w,s,y,m,n,k,xs,ys,groups,splits,st);break;
    case 16: launch_t<16,F32,SPLIT,FAST>(x,w,s,y,m,n,k,xs,ys,groups,splits,st);break;
    default: throw std::runtime_error("rows must be 1,2,4,8,16");
    }
}
template<bool FAST> static void linear_impl(u64 x,u64 w,u64 s,u64 y,u64 scratch,int m,int n,int k,int xs,int ys,int groups,int splits,int rows,bool f32,u64 st) {
    if(m<=0||m>128||n<=0||n%32||k<=0||k%32||groups<=0||groups>4||splits<=0||splits>32||xs<groups*k||ys<groups*n)
        throw std::runtime_error("invalid FP8 dense shape/strides/split");
    if(splits>1) {
        if(!scratch) throw std::runtime_error("split-K requires caller scratch");
        launch_r<true,true,FAST>(x,w,s,scratch,m,n,k,xs,ys,groups,splits,rows,st);
        check(cudaGetLastError());
        if(f32) reduce_kernel<true><<<(m*n*groups+255)/256,256,0,stream(st)>>>(reinterpret_cast<const float*>(scratch),reinterpret_cast<void*>(y),m,n,groups,splits,ys);
        else reduce_kernel<false><<<(m*n*groups+255)/256,256,0,stream(st)>>>(reinterpret_cast<const float*>(scratch),reinterpret_cast<void*>(y),m,n,groups,splits,ys);
    } else if(f32) launch_r<true,false,FAST>(x,w,s,y,m,n,k,xs,ys,groups,1,rows,st);
    else launch_r<false,false,FAST>(x,w,s,y,m,n,k,xs,ys,groups,1,rows,st);
    check(cudaGetLastError());
}
static void linear(u64 x,u64 w,u64 s,u64 y,u64 scratch,int m,int n,int k,int xs,int ys,int groups,int splits,int rows,bool f32,bool fast,u64 st) {
    if(fast) linear_impl<true>(x,w,s,y,scratch,m,n,k,xs,ys,groups,splits,rows,f32,st);
    else linear_impl<false>(x,w,s,y,scratch,m,n,k,xs,ys,groups,splits,rows,f32,st);
}
static void test_decode(u64 c,u64 s,u64 out,int count,u64 st) {
    decoder_kernel<<<(count+255)/256,256,0,stream(st)>>>(reinterpret_cast<const unsigned char*>(c),
        reinterpret_cast<const unsigned char*>(s),reinterpret_cast<uint16_t*>(out),count);
    check(cudaGetLastError());
}

// Tiny-M SIMT control: one warp per output row, reuse each weight for all M.
// Raw row-major FP8 keeps the four consecutive bytes per lane coalesced.
template<int MAXM,bool F32> __global__ __launch_bounds__(128) void simt_kernel(
    const __nv_bfloat16* x,const unsigned char* w,const unsigned char* sc,void* output,
    int m,int n,int k,int xs,int ys,int groups) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),gid=blockIdx.y;
    if(row>=n) return;
    float acc[MAXM]={};
    for(int base=0;base<k;base+=128) {
        int col=base+lane*4;
        u32 codes=*reinterpret_cast<const u32*>(w+(static_cast<size_t>(gid)*n+row)*k+col);
        int scale=sc[((static_cast<size_t>(gid)*n+row)/32)*(k/32)+col/32];
        u32 a=pair_decode_fast(codes,scale),b=pair_decode_fast(codes>>16,scale);
        float weights[4]={__uint_as_float(a<<16),__uint_as_float(a&0xffff0000U),
                          __uint_as_float(b<<16),__uint_as_float(b&0xffff0000U)};
        #pragma unroll
        for(int r=0;r<MAXM;++r) if(r<m) {
            uint2 xx=*reinterpret_cast<const uint2*>(x+static_cast<size_t>(r)*xs+gid*k+col);
            float v[4]={__uint_as_float(xx.x<<16),__uint_as_float(xx.x&0xffff0000U),
                        __uint_as_float(xx.y<<16),__uint_as_float(xx.y&0xffff0000U)};
            #pragma unroll
            for(int j=0;j<4;++j) acc[r]=__fmaf_rn(v[j],weights[j],acc[r]);
        }
    }
    #pragma unroll
    for(int r=0;r<MAXM;++r) if(r<m) {
        #pragma unroll
        for(int delta=16;delta;delta>>=1) acc[r]+=__shfl_down_sync(0xffffffffU,acc[r],delta);
        if(lane==0) {
            size_t idx=static_cast<size_t>(r)*ys+gid*n+row;
            if constexpr(F32) reinterpret_cast<float*>(output)[idx]=acc[r];
            else reinterpret_cast<__nv_bfloat16*>(output)[idx]=__float2bfloat16_rn(acc[r]);
        }
    }
}
template<bool F32> static void simt_launch(u64 x,u64 w,u64 s,u64 y,int m,int n,int k,int xs,int ys,int groups,u64 st) {
    dim3 grid((n+3)/4,groups);
    if(m<=8) simt_kernel<8,F32><<<grid,128,0,stream(st)>>>(reinterpret_cast<const __nv_bfloat16*>(x),reinterpret_cast<const unsigned char*>(w),reinterpret_cast<const unsigned char*>(s),reinterpret_cast<void*>(y),m,n,k,xs,ys,groups);
    else if(m<=16) simt_kernel<16,F32><<<grid,128,0,stream(st)>>>(reinterpret_cast<const __nv_bfloat16*>(x),reinterpret_cast<const unsigned char*>(w),reinterpret_cast<const unsigned char*>(s),reinterpret_cast<void*>(y),m,n,k,xs,ys,groups);
    else simt_kernel<20,F32><<<grid,128,0,stream(st)>>>(reinterpret_cast<const __nv_bfloat16*>(x),reinterpret_cast<const unsigned char*>(w),reinterpret_cast<const unsigned char*>(s),reinterpret_cast<void*>(y),m,n,k,xs,ys,groups);
}
static void simt(u64 x,u64 w,u64 s,u64 y,int m,int n,int k,int xs,int ys,int groups,bool f32,u64 st) {
    if(m<=0||m>20||n<=0||n%32||k<=0||k%128||groups<=0||groups>4||xs<groups*k||ys<groups*n)
        throw std::runtime_error("invalid tiny-M SIMT shape");
    if(f32) simt_launch<true>(x,w,s,y,m,n,k,xs,ys,groups,st);
    else simt_launch<false>(x,w,s,y,m,n,k,xs,ys,groups,st);
    check(cudaGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("prepack",&prepack);m.def("linear",&linear);m.def("test_decode",&test_decode);m.def("simt",&simt);}
