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

// Compact bounded schedule, one CTA; no cross-CTA counter initialization race.
template<class Id,int GROUP=8> __global__ void make_schedule(const Id* ids, int* dst, int* status, int slots, int experts) {
    __shared__ int count;
    if(threadIdx.x==0) count=0;
    __syncthreads();
    const int p=threadIdx.x;
    if(p<slots) {
        const auto e=ids[p];
        if(e<0 || e>=experts) atomicOr(status,1);
        int prior=0;
        for(int i=0;i<p;++i) prior+=ids[i]==e;
        // Invalid IDs also get schedule entries to poison their output slots.
        if(prior%GROUP==0) {
            int* entry=dst+1+atomicAdd(&count,1)*(GROUP+1);
            entry[0]=(e<0 || e>=experts)?-1:static_cast<int>(e);
            int n=0;
            for(int i=p;i<slots && n<GROUP;++i) if(ids[i]==e) entry[1+n++]=i;
            while(n<GROUP) entry[1+n++]=-1;
        }
    }
    __syncthreads();
    if(threadIdx.x==0) dst[0]=count;
}

__device__ __forceinline__ float prop_max(float a,float b) {
    return isnan(a)?a:(isnan(b)?b:fmaxf(a,b));
}
__device__ __forceinline__ float silu_route(float g, float u, float rw) {
    g=__bfloat162float(__float2bfloat16_rn(g));
    u=__bfloat162float(__float2bfloat16_rn(u));
    if(isnan(g) || isnan(u) || isnan(rw)) return NAN;
    g=fminf(g,10.0f);u=fminf(fmaxf(u,-10.0f),10.0f);
    // expf is libdevice full precision, compiler fast math/FMA disabled.
    float h=rw*((g/(1.0f+expf(-g)))*u);
    return __bfloat162float(__float2bfloat16_rn(h));
}
__device__ __forceinline__ float quantize(float v, float amax) {
    if(isnan(v) || isnan(amax)) return NAN;
    const float a=fmaxf(amax,1.e-4f)*(1.0f/448.0f);
    const u32 bits=__float_as_uint(a);
    const int e=((bits>>23)&255)-127+((bits&0x7fffff)!=0);
    const float sc=__uint_as_float((e+127)<<23);
    float q=fminf(fmaxf(v/sc,-448.0f),448.0f);
    const int qe=max(static_cast<int>((__float_as_uint(q)>>23)&255)-127,-6);
    const float step=__uint_as_float((qe-3+127)<<23);
    return (rintf(q/step)*step)*sc;
}

// Each warp owns one full 32-channel QDQ group and up to eight routes.
// MMA outputs for route pair 2*t+{0,1} span group={0..7}, two row halves,
// two 16-channel tiles. Reduce amax across those 32 channels using warp shuffles.
template<bool DEBUG,int WARPS> __global__ __launch_bounds__(WARPS*32) void gate_up(
    const __nv_bfloat16* x,const float* rw,const int* schedule,
    const uint4* w,const unsigned char* s,const uint4* up,const unsigned char* us,
    __nv_bfloat16* h,float* gout,float* uout,int n,int k,int slots) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int work=blockIdx.x*WARPS+warp, tiles=n/32;
    if(work>=schedule[0]*tiles) return;
    const int task=work/tiles,nt0=(work%tiles)*2;
    const int* entry=schedule+1+task*9;
    const int e=entry[0],group=lane>>2,t=lane&3;
    const int input_slot=entry[1+group];
    if(e<0) {
        #pragma unroll
        for(int r=0;r<8;++r) if(entry[1+r]>=0) {
            const size_t ix=static_cast<size_t>(entry[1+r])*n+nt0*16+lane;
            h[ix]=__float2bfloat16_rn(NAN);
            if constexpr(DEBUG) {gout[ix]=NAN;uout[ix]=NAN;}
        }
        return;
    }
    float acc[2][4]={{0,0,0,0},{0,0,0,0}},ua[2][4]={{0,0,0,0},{0,0,0,0}};
    const int blocks=k/64,groups=k/32,ntotal=n/16;
    for(int block=0;block<blocks;++block) {
        const size_t wi0=((static_cast<size_t>(e)*ntotal+nt0)*blocks+block)*32+lane;
        const size_t wi1=wi0+static_cast<size_t>(blocks)*32;
        // Load the complete 128-bit fragments before selecting each K16 word.
        // Scalar word loads at stride16 waste transaction bandwidth.
        const uint4 p0=w[wi0],p1=w[wi1],u0=up[wi0],u1=up[wi1];
        const u32 words[2][4]={{p0.x,p0.y,p0.z,p0.w},{p1.x,p1.y,p1.z,p1.w}};
        const u32 uwords[2][4]={{u0.x,u0.y,u0.z,u0.w},{u1.x,u1.y,u1.z,u1.w}};
        #pragma unroll
        for(int j=0;j<4;++j) {
            u32 b0=0,b1=0;
            if(input_slot>=0) {
                const auto* ptr=reinterpret_cast<const u32*>(x+static_cast<size_t>(input_slot/3)*k+block*64+j*16+t*2);
                b0=ptr[0];b1=ptr[4];
            }
            #pragma unroll
            for(int z=0;z<2;++z) {
                const size_t si=((static_cast<size_t>(e)*ntotal+nt0+z)*groups+block*2+j/2)*16+group;
                u32 a[4],au[4];
                decode(words[z][j],scale_pair(s[si]),scale_pair(s[si+8]),a);
                decode(uwords[z][j],scale_pair(us[si]),scale_pair(us[si+8]),au);
                mma(acc[z],a,b0,b1);mma(ua[z],au,b0,b1);
            }
        }
    }
    float v[2][4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int slot=entry[1+t*2+(j&1)];
        const float weight=slot>=0?rw[slot]:0.0f;
        #pragma unroll
        for(int z=0;z<2;++z) {
            v[z][j]=silu_route(acc[z][j],ua[z][j],weight);
            if constexpr(DEBUG) if(slot>=0) {
                const size_t ix=static_cast<size_t>(slot)*n+(nt0+z)*16+group+((j>=2)?8:0);
                gout[ix]=acc[z][j];uout[ix]=ua[z][j];
            }
        }
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        float a=prop_max(prop_max(fabsf(v[0][r]),fabsf(v[0][r+2])),prop_max(fabsf(v[1][r]),fabsf(v[1][r+2])));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,4));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,8));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,16));
        const int slot=entry[1+t*2+r];
        if(slot>=0) {
            #pragma unroll
            for(int z=0;z<2;++z) {
                const size_t ix=static_cast<size_t>(slot)*n+(nt0+z)*16+group;
                h[ix]=__float2bfloat16_rn(quantize(v[z][r],a));
                h[ix+8]=__float2bfloat16_rn(quantize(v[z][r+2],a));
            }
        }
    }
}


// A pair of 16-channel MMA warps shares only its eight amax scalars. Keep
// the original compact register tile while enforcing complete 32-channel QDQ.
template<bool DEBUG,int WARPS,bool COMPENSATED=false> __global__ __launch_bounds__(WARPS*32) void gate_up_shared(
    const __nv_bfloat16* x,const float* rw,const int* schedule,
    const uint4* w,const unsigned char* s,const uint4* up,const unsigned char* us,
    __nv_bfloat16* h,float* gout,float* uout,int n,int k,int slots) {
    __shared__ float maxima[WARPS][8];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int work=blockIdx.x*WARPS+warp,ntotal=n/16;
    if(blockIdx.x*WARPS>=schedule[0]*ntotal) return; // uniform CTA exit
    const bool active=WARPS==4 || work<schedule[0]*ntotal;
    const int task=work/ntotal,nt=work%ntotal;
    const int* entry=schedule+1+task*9;
    const int e=active?entry[0]:-1,group=lane>>2,t=lane&3;
    float acc[4]={0,0,0,0},ua[4]={0,0,0,0},v[4]={0,0,0,0};
    float sum[4]={0,0,0,0},usum[4]={0,0,0,0},correction[4]={0,0,0,0},ucorrection[4]={0,0,0,0};
    if(active && e>=0) {
        const int input_slot=entry[1+group],blocks=k/64,groups=k/32;
        for(int block=0;block<blocks;++block) {
            if constexpr(COMPENSATED) {
                #pragma unroll
                for(int j=0;j<4;++j) {acc[j]=0;ua[j]=0;}
            }
            const size_t wi=((static_cast<size_t>(e)*ntotal+nt)*blocks+block)*32+lane;
            const uint4 packed=w[wi],uv=up[wi];
            const u32 words[4]={packed.x,packed.y,packed.z,packed.w};
            const u32 uwords[4]={uv.x,uv.y,uv.z,uv.w};
            #pragma unroll
            for(int j=0;j<4;++j) {
                const size_t si=((static_cast<size_t>(e)*ntotal+nt)*groups+block*2+j/2)*16+group;
                u32 a[4],au[4];decode(words[j],scale_pair(s[si]),scale_pair(s[si+8]),a);
                decode(uwords[j],scale_pair(us[si]),scale_pair(us[si+8]),au);
                u32 b0=0,b1=0;
                if(input_slot>=0) {
                    const auto* ptr=reinterpret_cast<const u32*>(x+static_cast<size_t>(input_slot/3)*k+block*64+j*16+t*2);
                    b0=ptr[0];b1=ptr[4];
                }
                mma(acc,a,b0,b1);mma(ua,au,b0,b1);
            }
            if constexpr(COMPENSATED) {
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const float y=acc[j]-correction[j],u=ua[j]-ucorrection[j];
                    const float next=sum[j]+y,unext=usum[j]+u;
                    correction[j]=(next-sum[j])-y;ucorrection[j]=(unext-usum[j])-u;
                    sum[j]=next;usum[j]=unext;
                }
            }
        }
        if constexpr(COMPENSATED) {
            #pragma unroll
            for(int j=0;j<4;++j) {acc[j]=sum[j];ua[j]=usum[j];}
        }
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int slot=entry[1+t*2+(j&1)];
            v[j]=silu_route(acc[j],ua[j],slot>=0?rw[slot]:0.0f);
            if constexpr(DEBUG) if(slot>=0) {
                const size_t ix=static_cast<size_t>(slot)*n+nt*16+group+((j>=2)?8:0);
                gout[ix]=acc[j];uout[ix]=ua[j];
            }
        }
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        float a=prop_max(fabsf(v[r]),fabsf(v[r+2]));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,4));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,8));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,16));
        if(group==0) maxima[warp][t*2+r]=a;
    }
    __syncthreads();
    if(!active) return;
    if(e<0) {
        if(lane<16) for(int r=0;r<8;++r) if(entry[1+r]>=0) {
            const size_t ix=static_cast<size_t>(entry[1+r])*n+nt*16+lane;
            h[ix]=__float2bfloat16_rn(NAN);
            if constexpr(DEBUG) {gout[ix]=NAN;uout[ix]=NAN;}
        }
        return;
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int r=t*2+(j&1),slot=entry[1+r];
        if(slot>=0) {
            const float a=prop_max(maxima[warp][r],maxima[warp^1][r]);
            h[static_cast<size_t>(slot)*n+nt*16+group+((j>=2)?8:0)]=__float2bfloat16_rn(quantize(v[j],a));
        }
    }
}
template<bool DEBUG,int SPLIT,bool COMPENSATED=false,bool FINE=false,int OUTPUT_WARPS=4,bool CARRY=false> __global__ __launch_bounds__(32*OUTPUT_WARPS*SPLIT,(OUTPUT_WARPS==2?2:1)) void gate_up_split(
    const __nv_bfloat16* x,const float* rw,const int* schedule,
    const uint4* w,const unsigned char* s,const uint4* up,const unsigned char* us,
    __nv_bfloat16* h,float* gout,float* uout,int n,int k,int slots) {
    __shared__ float maxima[OUTPUT_WARPS][8];
    __shared__ float partial_g[SPLIT][OUTPUT_WARPS][4][32],partial_u[SPLIT][OUTPUT_WARPS][4][32];
    __shared__ float partial_cg[(FINE||CARRY)?SPLIT:1][OUTPUT_WARPS][4][32],partial_cu[(FINE||CARRY)?SPLIT:1][OUTPUT_WARPS][4][32];
    const int lane=threadIdx.x&31,warp=(threadIdx.x>>5)%OUTPUT_WARPS,split=(threadIdx.x>>5)/OUTPUT_WARPS;
    const int work=blockIdx.x*OUTPUT_WARPS+warp,ntotal=n/16;
    if(blockIdx.x*OUTPUT_WARPS>=schedule[0]*ntotal) return; // uniform CTA exit
    const bool active=true; // N%64 makes four output warps a complete CTA
    const int task=work/ntotal,nt=work%ntotal;
    const int* entry=schedule+1+task*9;
    const int e=active?entry[0]:-1,group=lane>>2,t=lane&3;
    float acc[4]={0,0,0,0},ua[4]={0,0,0,0},v[4]={0,0,0,0};
    float sum[4]={0,0,0,0},usum[4]={0,0,0,0},correction[4]={0,0,0,0},ucorrection[4]={0,0,0,0};
    if(active && e>=0) {
        const int input_slot=entry[1+group],blocks=k/64,groups=k/32;
        for(int block=(blocks*split)/SPLIT;block<(blocks*(split+1))/SPLIT;++block) {
            if constexpr(COMPENSATED) {
                #pragma unroll
                for(int j=0;j<4;++j) {acc[j]=0;ua[j]=0;}
            }
            const size_t wi=((static_cast<size_t>(e)*ntotal+nt)*blocks+block)*32+lane;
            const uint4 packed=w[wi],uv=up[wi];
            const u32 words[4]={packed.x,packed.y,packed.z,packed.w};
            const u32 uwords[4]={uv.x,uv.y,uv.z,uv.w};
            #pragma unroll
            for(int j=0;j<4;++j) {
                const size_t si=((static_cast<size_t>(e)*ntotal+nt)*groups+block*2+j/2)*16+group;
                u32 a[4],au[4];decode(words[j],scale_pair(s[si]),scale_pair(s[si+8]),a);
                decode(uwords[j],scale_pair(us[si]),scale_pair(us[si+8]),au);
                u32 b0=0,b1=0;
                if(input_slot>=0) {
                    const auto* ptr=reinterpret_cast<const u32*>(x+static_cast<size_t>(input_slot/3)*k+block*64+j*16+t*2);
                    b0=ptr[0];b1=ptr[4];
                }
                if constexpr(FINE) {
                    #pragma unroll
                    for(int q=0;q<4;++q) {acc[q]=0;ua[q]=0;}
                }
                mma(acc,a,b0,b1);mma(ua,au,b0,b1);
                if constexpr(FINE) {
                    #pragma unroll
                    for(int q=0;q<4;++q) {
                        const float y=acc[q]-correction[q],u=ua[q]-ucorrection[q];
                        const float next=sum[q]+y,unext=usum[q]+u;
                        correction[q]=(next-sum[q])-y;ucorrection[q]=(unext-usum[q])-u;
                        sum[q]=next;usum[q]=unext;
                    }
                }
            }
            if constexpr(COMPENSATED && !FINE) {
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const float y=acc[j]-correction[j],u=ua[j]-ucorrection[j];
                    const float next=sum[j]+y,unext=usum[j]+u;
                    correction[j]=(next-sum[j])-y;ucorrection[j]=(unext-usum[j])-u;
                    sum[j]=next;usum[j]=unext;
                }
            }
        }
        if constexpr(COMPENSATED) {
            #pragma unroll
            for(int j=0;j<4;++j) {acc[j]=sum[j];ua[j]=usum[j];}
        }
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        partial_g[split][warp][j][lane]=acc[j];partial_u[split][warp][j][lane]=ua[j];
        if constexpr(FINE || CARRY) {partial_cg[split][warp][j][lane]=correction[j];partial_cu[split][warp][j][lane]=ucorrection[j];}
    }
    __syncthreads();
    if(split==0) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            if constexpr(COMPENSATED) {correction[j]=0;ucorrection[j]=0;}
            #pragma unroll
            for(int part=1;part<SPLIT;++part) {
                if constexpr(COMPENSATED) {
                    const float y=partial_g[part][warp][j][lane]-correction[j];
                    const float u=partial_u[part][warp][j][lane]-ucorrection[j];
                    const float next=acc[j]+y,unext=ua[j]+u;
                    correction[j]=(next-acc[j])-y;ucorrection[j]=(unext-ua[j])-u;
                    acc[j]=next;ua[j]=unext;
                } else {
                    acc[j]+=partial_g[part][warp][j][lane];ua[j]+=partial_u[part][warp][j][lane];
                }
            }
        }
        if constexpr(FINE || CARRY) {
            #pragma unroll
            for(int j=0;j<4;++j) {
                float sg=0,su=0,cg=0,cu=0;
                #pragma unroll
                for(int part=0;part<SPLIT;++part) {
                    #pragma unroll
                    for(int q=0;q<2;++q) {
                        const float xg=q==0?partial_g[part][warp][j][lane]:-partial_cg[part][warp][j][lane];
                        const float xu=q==0?partial_u[part][warp][j][lane]:-partial_cu[part][warp][j][lane];
                        const float tg=sg+xg,tu=su+xu;
                        cg+=fabsf(sg)>=fabsf(xg)?(sg-tg)+xg:(xg-tg)+sg;
                        cu+=fabsf(su)>=fabsf(xu)?(su-tu)+xu:(xu-tu)+su;
                        sg=tg;su=tu;
                    }
                }
                acc[j]=sg+cg;ua[j]=su+cu;
            }
        }
        if(active && e>=0) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int slot=entry[1+t*2+(j&1)];
            v[j]=silu_route(acc[j],ua[j],slot>=0?rw[slot]:0.0f);
            if constexpr(DEBUG) if(slot>=0) {
                const size_t ix=static_cast<size_t>(slot)*n+nt*16+group+((j>=2)?8:0);
                gout[ix]=acc[j];uout[ix]=ua[j];
            }
        }
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        float a=prop_max(fabsf(v[r]),fabsf(v[r+2]));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,4));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,8));
        a=prop_max(a,__shfl_xor_sync(0xffffffff,a,16));
        if(group==0) maxima[warp][t*2+r]=a;
    }
    }
    __syncthreads();
    if(!active || split!=0) return;
    if(e<0) {
        if(lane<16) for(int r=0;r<8;++r) if(entry[1+r]>=0) {
            const size_t ix=static_cast<size_t>(entry[1+r])*n+nt*16+lane;
            h[ix]=__float2bfloat16_rn(NAN);
            if constexpr(DEBUG) {gout[ix]=NAN;uout[ix]=NAN;}
        }
        return;
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int r=t*2+(j&1),slot=entry[1+r];
        if(slot>=0) {
            const float a=prop_max(maxima[warp][r],maxima[warp^1][r]);
            h[static_cast<size_t>(slot)*n+nt*16+group+((j>=2)?8:0)]=__float2bfloat16_rn(quantize(v[j],a));
        }
    }
}
template<int WARPS> __global__ __launch_bounds__(WARPS*32) void down(
    const __nv_bfloat16* h,const int* schedule,const uint4* w,const unsigned char* s,
    float* out,int n,int k,int slots,int out_stride) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int work=blockIdx.x*WARPS+warp,ntotal=n/16;
    if(work>=schedule[0]*ntotal) return;
    const int task=work/ntotal,nt=work%ntotal;
    const int* entry=schedule+1+task*9;
    const int e=entry[0],group=lane>>2,t=lane&3;
    if(e<0) {
        if(lane<16) for(int j=0;j<8;++j) if(entry[j+1]>=0) {
            const int slot=entry[j+1];
            out[static_cast<size_t>(slot/3)*out_stride+(slot%3)*n+nt*16+lane]=NAN;
        }
        return;
    }
    const int input_slot=entry[1+group],blocks=k/64,groups=k/32;
    float acc[4]={0,0,0,0};
    for(int block=0;block<blocks;++block) {
        const size_t wi=((static_cast<size_t>(e)*ntotal+nt)*blocks+block)*32+lane;
        const uint4 packed=w[wi];
        const u32 words[4]={packed.x,packed.y,packed.z,packed.w};
        #pragma unroll
        for(int j=0;j<4;++j) {
            const size_t si=((static_cast<size_t>(e)*ntotal+nt)*groups+block*2+j/2)*16+group;
            u32 a[4];decode(words[j],scale_pair(s[si]),scale_pair(s[si+8]),a);
            u32 b0=0,b1=0;
            if(input_slot>=0) {
                const auto* ptr=reinterpret_cast<const u32*>(h+static_cast<size_t>(input_slot)*k+block*64+j*16+t*2);
                b0=ptr[0];b1=ptr[4];
            }
            mma(acc,a,b0,b1);
        }
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int slot=entry[1+t*2+(j&1)];
        if(slot>=0) out[static_cast<size_t>(slot/3)*out_stride+(slot%3)*n+nt*16+group+((j>=2)?8:0)]=acc[j];
    }
}
// Two B8 MMA tiles share every decoded weight fragment. A group of up to16
// occurrences replaces the inherited repeated weight passes above eight.
template<bool DEBUG> __global__ __launch_bounds__(256) void gate_up_group16(
    const __nv_bfloat16* x,const float* rw,const int* schedule,
    const uint4* w,const unsigned char* s,const uint4* up,const unsigned char* us,
    __nv_bfloat16* h,float* gout,float* uout,int n,int k,int slots) {
    __shared__ float pg[2][4][2][4][32],pu[2][4][2][4][32],maxima[4][16];
    const int lane=threadIdx.x&31,warp=(threadIdx.x>>5)%4,split=(threadIdx.x>>5)/4;
    const int work=blockIdx.x*4+warp,ntotal=n/16;
    if(blockIdx.x*4>=schedule[0]*ntotal) return;
    const int task=work/ntotal,nt=work%ntotal;
    const int* entry=schedule+1+task*17;
    const int e=entry[0],group=lane>>2,t=lane&3;
    const bool second=entry[9]>=0;
    float acc[2][4]={{0,0,0,0},{0,0,0,0}},ua[2][4]={{0,0,0,0},{0,0,0,0}};
    if(e>=0) {
        const int blocks=k/64,groups=k/32;
        for(int block=(blocks*split)/2;block<(blocks*(split+1))/2;++block) {
            const size_t wi=((static_cast<size_t>(e)*ntotal+nt)*blocks+block)*32+lane;
            const uint4 packed=w[wi],uv=up[wi];
            const u32 words[4]={packed.x,packed.y,packed.z,packed.w};
            const u32 uwords[4]={uv.x,uv.y,uv.z,uv.w};
            #pragma unroll
            for(int j=0;j<4;++j) {
                const size_t si=((static_cast<size_t>(e)*ntotal+nt)*groups+block*2+j/2)*16+group;
                u32 a[4],au[4];decode(words[j],scale_pair(s[si]),scale_pair(s[si+8]),a);
                decode(uwords[j],scale_pair(us[si]),scale_pair(us[si+8]),au);
                #pragma unroll
                for(int b=0;b<2;++b) if(b==0 || second) {
                    const int input_slot=entry[1+b*8+group];
                    u32 b0=0,b1=0;
                    if(input_slot>=0) {
                        const auto* ptr=reinterpret_cast<const u32*>(x+static_cast<size_t>(input_slot/3)*k+block*64+j*16+t*2);
                        b0=ptr[0];b1=ptr[4];
                    }
                    mma(acc[b],a,b0,b1);mma(ua[b],au,b0,b1);
                }
            }
        }
    }
    #pragma unroll
    for(int b=0;b<2;++b) if(b==0 || second) {
        #pragma unroll
        for(int j=0;j<4;++j) {pg[split][warp][b][j][lane]=acc[b][j];pu[split][warp][b][j][lane]=ua[b][j];}
    }
    __syncthreads();
    float v[2][4];
    if(split==0) {
        #pragma unroll
        for(int b=0;b<2;++b) if(b==0 || second) {
            #pragma unroll
            for(int j=0;j<4;++j) {
                acc[b][j]+=pg[1][warp][b][j][lane];ua[b][j]+=pu[1][warp][b][j][lane];
                const int slot=entry[1+b*8+t*2+(j&1)];
                v[b][j]=silu_route(acc[b][j],ua[b][j],slot>=0?rw[slot]:0.0f);
                if constexpr(DEBUG) if(slot>=0) {
                    const size_t ix=static_cast<size_t>(slot)*n+nt*16+group+((j>=2)?8:0);
                    gout[ix]=e>=0?acc[b][j]:NAN;uout[ix]=e>=0?ua[b][j]:NAN;
                }
            }
            #pragma unroll
            for(int r=0;r<2;++r) {
                float a=prop_max(fabsf(v[b][r]),fabsf(v[b][r+2]));
                a=prop_max(a,__shfl_xor_sync(0xffffffff,a,4));
                a=prop_max(a,__shfl_xor_sync(0xffffffff,a,8));
                a=prop_max(a,__shfl_xor_sync(0xffffffff,a,16));
                if(group==0) maxima[warp][b*8+t*2+r]=a;
            }
        }
    }
    __syncthreads();
    if(split!=0) return;
    #pragma unroll
    for(int b=0;b<2;++b) if(b==0 || second) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int r=b*8+t*2+(j&1),slot=entry[1+r];
            if(slot>=0) {
                const float a=prop_max(maxima[warp][r],maxima[warp^1][r]);
                h[static_cast<size_t>(slot)*n+nt*16+group+((j>=2)?8:0)]=
                    __float2bfloat16_rn(e<0?NAN:quantize(v[b][j],a));
            }
        }
    }
}
__global__ __launch_bounds__(128) void down_group16(
    const __nv_bfloat16* h,const int* schedule,const uint4* w,const unsigned char* s,
    float* out,int n,int k,int slots,int out_stride) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int work=blockIdx.x*4+warp,ntotal=n/16;
    if(work>=schedule[0]*ntotal) return;
    const int task=work/ntotal,nt=work%ntotal;
    const int* entry=schedule+1+task*17;
    const int e=entry[0],group=lane>>2,t=lane&3;
    const bool second=entry[9]>=0;
    if(e<0) {
        if(lane<16) for(int j=0;j<16;++j) if(entry[j+1]>=0) {
            const int slot=entry[j+1];out[static_cast<size_t>(slot/3)*out_stride+(slot%3)*n+nt*16+lane]=NAN;
        }
        return;
    }
    const int blocks=k/64,groups=k/32;
    float acc[2][4]={{0,0,0,0},{0,0,0,0}};
    for(int block=0;block<blocks;++block) {
        const size_t wi=((static_cast<size_t>(e)*ntotal+nt)*blocks+block)*32+lane;
        const uint4 packed=w[wi];const u32 words[4]={packed.x,packed.y,packed.z,packed.w};
        #pragma unroll
        for(int j=0;j<4;++j) {
            const size_t si=((static_cast<size_t>(e)*ntotal+nt)*groups+block*2+j/2)*16+group;
            u32 a[4];decode(words[j],scale_pair(s[si]),scale_pair(s[si+8]),a);
            #pragma unroll
            for(int b=0;b<2;++b) if(b==0 || second) {
                const int input_slot=entry[1+b*8+group];u32 b0=0,b1=0;
                if(input_slot>=0) {
                    const auto* ptr=reinterpret_cast<const u32*>(h+static_cast<size_t>(input_slot)*k+block*64+j*16+t*2);
                    b0=ptr[0];b1=ptr[4];
                }
                mma(acc[b],a,b0,b1);
            }
        }
    }
    #pragma unroll
    for(int b=0;b<2;++b) if(b==0 || second) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int slot=entry[1+b*8+t*2+(j&1)];
            if(slot>=0) out[static_cast<size_t>(slot/3)*out_stride+(slot%3)*n+nt*16+group+((j>=2)?8:0)]=acc[b][j];
        }
    }
}

static void schedule(u64 ids,u64 dst,u64 status,int slots,int experts,bool i64,int group,u64 st) {
    if(slots<3 || slots>60 || slots%3 || experts<1) throw std::runtime_error("invalid fusedmoe schedule shape");
    if(group!=8 && group!=16) throw std::runtime_error("group must be 8 or 16");
    if(group==16) {
        if(i64) make_schedule<int64_t,16><<<1,64,0,stream(st)>>>(reinterpret_cast<const int64_t*>(ids),reinterpret_cast<int*>(dst),reinterpret_cast<int*>(status),slots,experts);
        else make_schedule<int,16><<<1,64,0,stream(st)>>>(reinterpret_cast<const int*>(ids),reinterpret_cast<int*>(dst),reinterpret_cast<int*>(status),slots,experts);
    } else {
        if(i64) make_schedule<int64_t,8><<<1,64,0,stream(st)>>>(reinterpret_cast<const int64_t*>(ids),reinterpret_cast<int*>(dst),reinterpret_cast<int*>(status),slots,experts);
        else make_schedule<int,8><<<1,64,0,stream(st)>>>(reinterpret_cast<const int*>(ids),reinterpret_cast<int*>(dst),reinterpret_cast<int*>(status),slots,experts);
    }
    check(cudaGetLastError());
}
static void launch(u64 x,u64 rw,u64 sched,u64 w,u64 s,u64 up,u64 us,u64 w2,u64 s2,u64 h,u64 out,
                   u64 gout,u64 uout,int n,int k,int slots,int out_stride,int warps,u64 st) {
    if(n<=0 || n%64 || k<=0 || k%64 || slots<3 || slots>60 || slots%3 || out_stride<3*k)
        throw std::runtime_error("invalid fusedmoe launch dimensions");
    if(bool(gout)!=bool(uout)) throw std::runtime_error("both projection debug buffers required");
#define ARGS reinterpret_cast<const __nv_bfloat16*>(x),reinterpret_cast<const float*>(rw),reinterpret_cast<const int*>(sched),reinterpret_cast<const uint4*>(w),reinterpret_cast<const unsigned char*>(s),reinterpret_cast<const uint4*>(up),reinterpret_cast<const unsigned char*>(us),reinterpret_cast<__nv_bfloat16*>(h),reinterpret_cast<float*>(gout),reinterpret_cast<float*>(uout),n,k,slots
#define DARGS reinterpret_cast<const __nv_bfloat16*>(h),reinterpret_cast<const int*>(sched),reinterpret_cast<const uint4*>(w2),reinterpret_cast<const unsigned char*>(s2),reinterpret_cast<float*>(out),k,n,slots,out_stride
#define DISPATCH(W) if(gout) gate_up<true,W><<<(slots*(n/32)+W-1)/W,W*32,0,stream(st)>>>(ARGS); else gate_up<false,W><<<(slots*(n/32)+W-1)/W,W*32,0,stream(st)>>>(ARGS); check(cudaGetLastError()); down<W><<<(slots*(k/16)+W-1)/W,W*32,0,stream(st)>>>(DARGS)
    if(warps==131072) {
        if(gout) gate_up_split<true,2,true,false,4,true><<<(slots*(n/16)+3)/4,256,0,stream(st)>>>(ARGS);
        else gate_up_split<false,2,true,false,4,true><<<(slots*(n/16)+3)/4,256,0,stream(st)>>>(ARGS);
        check(cudaGetLastError()); down<4><<<(slots*(k/16)+3)/4,128,0,stream(st)>>>(DARGS);
    } else if(warps==65536) {
        if(gout) gate_up_group16<true><<<(slots*(n/16)+3)/4,256,0,stream(st)>>>(ARGS);
        else gate_up_group16<false><<<(slots*(n/16)+3)/4,256,0,stream(st)>>>(ARGS);
        check(cudaGetLastError()); down_group16<<<(slots*(k/16)+3)/4,128,0,stream(st)>>>(DARGS);
    } else if(warps==16384) {
        if(gout) gate_up_split<true,8,true,true,2><<<(slots*(n/16)+1)/2,512,0,stream(st)>>>(ARGS);
        else gate_up_split<false,8,true,true,2><<<(slots*(n/16)+1)/2,512,0,stream(st)>>>(ARGS);
        check(cudaGetLastError()); down<4><<<(slots*(k/16)+3)/4,128,0,stream(st)>>>(DARGS);
    } else if(warps==4096 || warps==8192) {
#define FINE_LAUNCH(S) if(gout) gate_up_split<true,S,true,true><<<(slots*(n/16)+3)/4,128*S,0,stream(st)>>>(ARGS); else gate_up_split<false,S,true,true><<<(slots*(n/16)+3)/4,128*S,0,stream(st)>>>(ARGS); check(cudaGetLastError()); down<4><<<(slots*(k/16)+3)/4,128,0,stream(st)>>>(DARGS)
        if(warps==4096) {FINE_LAUNCH(2);} else {FINE_LAUNCH(4);}
#undef FINE_LAUNCH
    } else if(warps==256 || warps==512 || warps==1024 || warps==2048) {
#define SPLIT_LAUNCH(S,C) if(gout) gate_up_split<true,S,C><<<(slots*(n/16)+3)/4,128*S,0,stream(st)>>>(ARGS); else gate_up_split<false,S,C><<<(slots*(n/16)+3)/4,128*S,0,stream(st)>>>(ARGS); check(cudaGetLastError()); down<4><<<(slots*(k/16)+3)/4,128,0,stream(st)>>>(DARGS)
        if(warps==256) {SPLIT_LAUNCH(2,false);} else if(warps==512) {SPLIT_LAUNCH(4,false);}
        else if(warps==1024) {SPLIT_LAUNCH(4,true);} else {SPLIT_LAUNCH(2,true);}
#undef SPLIT_LAUNCH
    } else if(warps==64 || warps==128) {
#define PRECISE(W) if(gout) gate_up_shared<true,W,true><<<(slots*(n/16)+W-1)/W,W*32,0,stream(st)>>>(ARGS); else gate_up_shared<false,W,true><<<(slots*(n/16)+W-1)/W,W*32,0,stream(st)>>>(ARGS); check(cudaGetLastError()); down<W><<<(slots*(k/16)+W-1)/W,W*32,0,stream(st)>>>(DARGS)
        if(warps==64) {PRECISE(4);} else {PRECISE(8);}
#undef PRECISE
    } else if(warps==16 || warps==32) {
#define SHARED(W) if(gout) gate_up_shared<true,W><<<(slots*(n/16)+W-1)/W,W*32,0,stream(st)>>>(ARGS); else gate_up_shared<false,W><<<(slots*(n/16)+W-1)/W,W*32,0,stream(st)>>>(ARGS); check(cudaGetLastError()); down<W><<<(slots*(k/16)+W-1)/W,W*32,0,stream(st)>>>(DARGS)
        if(warps==16) {SHARED(4);} else {SHARED(8);}
#undef SHARED
    } else if(warps==4) {DISPATCH(4);} else if(warps==8) {DISPATCH(8);}
    else throw std::runtime_error("unsupported fusedmoe variant");
#undef DISPATCH
#undef DARGS
#undef ARGS
    check(cudaGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("schedule",&schedule);m.def("launch",&launch);}
