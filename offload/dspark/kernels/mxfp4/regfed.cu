// SPDX-License-Identifier: Apache-2.0
// Fragment prepack, prmt decoder and MMA mapping adapted from ro99/strata
// 015b075079c51a7aec670ee24924f920f5e7bb2b, backend_kernels.cuh.
// See LICENSE.strata / NOTICE. Original changes: bounded device-ID schedule,
// direct BF16 activation loads, per-slot output, signed-zero preservation,
// framework-independent pointer API, no global device allocations or caches.
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cmath>

using u32 = uint32_t;
using u64 = uint64_t;
namespace py = pybind11;
static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static cudaStream_t stream(u64 p) {return reinterpret_cast<cudaStream_t>(p);}

__device__ __forceinline__ void decode(u32 word, u32 sl, u32 sh, u32 (&out)[4]) {
    const u32 mag = word & 0x77777777U;
    const u32 ha = __byte_perm(0x3f3f3f00U, 0x40404040U, mag);
    const u32 la = __byte_perm(0xc0800000U, 0xc0804000U, mag);
    const u32 hb = __byte_perm(0x3f3f3f00U, 0x40404040U, mag>>16);
    const u32 lb = __byte_perm(0xc0800000U, 0xc0804000U, mag>>16);
    const u32 pa = __byte_perm(la,ha,0x5140U), qa = __byte_perm(la,ha,0x7362U);
    const u32 pb = __byte_perm(lb,hb,0x5140U), qb = __byte_perm(lb,hb,0x7362U);
    const u32 v[4] = {__byte_perm(pa,pb,0x5410U), __byte_perm(pa,pb,0x7632U),
                      __byte_perm(qa,qb,0x5410U), __byte_perm(qa,qb,0x7632U)};
    #pragma unroll
    for (int i=0;i<4;++i) {
        const u32 scale = (i&1) ? sh : sl;
        const __nv_bfloat162 val = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v[i]),
                                          *reinterpret_cast<const __nv_bfloat162*>(&scale));
        out[i] = *reinterpret_cast<const u32*>(&val) ^ ((word << (12-i*4)) & 0x80008000U);
    }
}
__device__ __forceinline__ u32 scale_pair(u32 code) {return (code<<7)*0x00010001U;}
__device__ __forceinline__ void mma(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}

__global__ void prepack_codes(u32* dst, const unsigned char* src, int n, int k) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    const int kt=k/16;
    if (i >= (n/16)*kt*32) return;
    const int lane=i&31, tile=(i>>5)%kt, nt=(i>>5)/kt, group=lane>>2, t=lane&3;
    u32 word=0;
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int row=nt*16+group+((j&1)?8:0), col=tile*16+t*2+((j>=2)?8:0);
        const u32 pair=src[static_cast<size_t>(row)*(k/2)+col/2];
        word |= (pair&15)<<(j*4);
        word |= (pair>>4)<<((j+4)*4);
    }
    dst[((static_cast<size_t>(nt)*(kt/4)+tile/4)*32+lane)*4+tile%4]=word;
}
__global__ void prepack_scales(unsigned char* dst, const unsigned char* src, int n, int groups) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n*groups) return;
    const int row=i%16, g=(i/16)%groups, nt=(i/16)/groups;
    dst[i]=src[(nt*16+row)*groups+g];
}

// Each pivot owns eight successive occurrences of its expert. Non-leaders
// have an empty schedule, so repeated experts reuse one decoded weight tile.
__global__ void schedule_kernel(const int* ids, int* schedule, int slots, int experts, int* status) {
    const int pivot=blockIdx.x*blockDim.x+threadIdx.x;
    if (pivot>=slots) return;
    int* dst=schedule+pivot*8;
    #pragma unroll
    for(int j=0;j<8;++j) dst[j]=-1;
    const int e=ids[pivot];
    if (e<0 || e>=experts) {atomicOr(status,1); return;}
    int prior=0;
    for(int j=0;j<pivot;++j) prior += ids[j]==e;
    if (prior%8) return;
    int count=0;
    for(int j=pivot;j<slots && count<8;++j) if(ids[j]==e) dst[count++]=j;
}

template<bool FUSED> __global__ __launch_bounds__(128) void linear(
        const __nv_bfloat16* x, const int* ids, const int* schedule,
        const uint4* w, const unsigned char* s, const uint4* up, const unsigned char* us,
        float* out, float* uout, int n, int k, int slots, int experts, int top, bool per_slot) {
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const int work=blockIdx.x*4+warp, ntotal=n/16;
    if(work>=ntotal*slots) return;
    const int pivot=work/ntotal, nt=work%ntotal, e=ids[pivot];
    if(e<0 || e>=experts) {
        if(lane<16) {out[static_cast<size_t>(pivot)*n+nt*16+lane]=NAN;
            if constexpr(FUSED) uout[static_cast<size_t>(pivot)*n+nt*16+lane]=NAN;}
        return;
    }
    if(schedule[pivot*8]<0) return;
    const int group=lane>>2, t=lane&3;
    const int input_slot=schedule[pivot*8+group];
    const int row=per_slot?input_slot:input_slot/top;
    const int blocks=k/64, groups=k/32;
    float acc[4]={0,0,0,0}, ua[4]={0,0,0,0};
    for(int block=0;block<blocks;++block) {
        const size_t wi=((static_cast<size_t>(e)*ntotal+nt)*blocks+block)*32+lane;
        const uint4 packed=w[wi];
        uint4 uv;
        if constexpr(FUSED) uv=up[wi];
        const u32 words[4]={packed.x,packed.y,packed.z,packed.w};
        u32 uwords[4];
        if constexpr(FUSED) {uwords[0]=uv.x;uwords[1]=uv.y;uwords[2]=uv.z;uwords[3]=uv.w;}
        #pragma unroll
        for(int j=0;j<4;++j) {
            const size_t si=((static_cast<size_t>(e)*ntotal+nt)*groups+block*2+j/2)*16+group;
            u32 a[4]; decode(words[j],scale_pair(s[si]),scale_pair(s[si+8]),a);
            u32 b0=0,b1=0;
            if(input_slot>=0) {
                const auto* ptr=reinterpret_cast<const u32*>(x+static_cast<size_t>(row)*k+block*64+j*16+t*2);
                b0=ptr[0];b1=ptr[4];
            }
            mma(acc,a,b0,b1);
            if constexpr(FUSED) {
                u32 aup[4];decode(uwords[j],scale_pair(us[si]),scale_pair(us[si+8]),aup);
                mma(ua,aup,b0,b1);
            }
        }
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int slot=schedule[pivot*8+t*2+(j&1)];
        if(slot>=0) {
            const size_t idx=static_cast<size_t>(slot)*n+nt*16+group+((j>=2)?8:0);
            out[idx]=acc[j];
            if constexpr(FUSED) uout[idx]=ua[j];
        }
    }
}

__global__ void decoder_test(const unsigned char* codes,const unsigned char* scales,
                             unsigned short* out,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const u32 code=codes[i], word=code*0x11111111U;
    u32 result[4];
    decode(word,scale_pair(scales[i]),scale_pair(scales[i]),result);
    out[i]=static_cast<unsigned short>(result[0]);
}

static void prepack(u64 wp,u64 sp,u64 srcp,u64 srcs,int n,int k,u64 st) {
    if(n<=0 || n%16 || k<=0 || k%64) throw std::runtime_error("regfed requires N%16=0,K%64=0");
    prepack_codes<<<((n/16)*(k/16)*32+255)/256,256,0,stream(st)>>>(
        reinterpret_cast<u32*>(wp),reinterpret_cast<const unsigned char*>(srcp),n,k);
    check(cudaGetLastError());
    prepack_scales<<<(n*(k/32)+255)/256,256,0,stream(st)>>>(
        reinterpret_cast<unsigned char*>(sp),reinterpret_cast<const unsigned char*>(srcs),n,k/32);
    check(cudaGetLastError());
}
static void schedule(u64 ids,u64 dst,int slots,int experts,u64 status,u64 st) {
    if(slots<=0 || slots>64 || experts<=0) throw std::runtime_error("invalid bounded routing schedule");
    schedule_kernel<<<(slots+127)/128,128,0,stream(st)>>>(reinterpret_cast<const int*>(ids),
        reinterpret_cast<int*>(dst),slots,experts,reinterpret_cast<int*>(status));
    check(cudaGetLastError());
}
static void launch(u64 x,u64 ids,u64 sched,u64 w,u64 s,u64 up,u64 us,u64 out,u64 uout,
                   int n,int k,int slots,int experts,int top,bool per_slot,bool fused,u64 st) {
    if(n<=0 || n%16 || k<=0 || k%64 || slots<=0 || slots>64 || experts<=0 || top<=0 || top>3)
        throw std::runtime_error("invalid regfed shape");
    const int grid=((n/16)*slots+3)/4;
    if(fused) linear<true><<<grid,128,0,stream(st)>>>(reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const int*>(ids),reinterpret_cast<const int*>(sched),
        reinterpret_cast<const uint4*>(w),reinterpret_cast<const unsigned char*>(s),
        reinterpret_cast<const uint4*>(up),reinterpret_cast<const unsigned char*>(us),
        reinterpret_cast<float*>(out),reinterpret_cast<float*>(uout),n,k,slots,experts,top,per_slot);
    else linear<false><<<grid,128,0,stream(st)>>>(reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const int*>(ids),reinterpret_cast<const int*>(sched),
        reinterpret_cast<const uint4*>(w),reinterpret_cast<const unsigned char*>(s),
        nullptr,nullptr,reinterpret_cast<float*>(out),nullptr,n,k,slots,experts,top,per_slot);
    check(cudaGetLastError());
}
static void test_decode(u64 c,u64 s,u64 out,int count,u64 st) {
    decoder_test<<<(count+255)/256,256,0,stream(st)>>>(reinterpret_cast<const unsigned char*>(c),
        reinterpret_cast<const unsigned char*>(s),reinterpret_cast<unsigned short*>(out),count);
    check(cudaGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("prepack",&prepack);m.def("schedule",&schedule);m.def("linear",&launch);m.def("test_decode",&test_decode);
}
