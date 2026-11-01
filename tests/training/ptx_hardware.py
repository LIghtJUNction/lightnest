import ctypes as C
import torch
from lightnest.architecture.ptx_internal import PTXIntegerOps


def verify_hardware_ptx():
    ptx = b'''.version 8.0
    .target sm_80
    .address_size 64
    .visible .entry check_ops(.param .u64 a, .param .u64 b, .param .u64 x, .param .u64 y) {
    .reg .b32 %r<6>; .reg .b64 %rd<12>;
    ld.param.u64 %rd1,[a]; ld.param.u64 %rd2,[b]; ld.param.u64 %rd3,[x]; ld.param.u64 %rd4,[y];
    mov.u32 %r1,%tid.x; mul.wide.u32 %rd5,%r1,4;
    add.u64 %rd6,%rd1,%rd5; add.u64 %rd7,%rd2,%rd5;
    add.u64 %rd8,%rd3,%rd5; add.u64 %rd9,%rd4,%rd5;
    ld.global.u32 %r2,[%rd6]; ld.global.u32 %r3,[%rd7];
    add.u32 %r4,%r2,%r3; mul.lo.u32 %r5,%r2,%r3;
    st.global.u32 [%rd8],%r4; st.global.u32 [%rd9],%r5; ret; }'''
    lib=C.CDLL('libcuda.so.1')
    lib.cuInit.argtypes=[C.c_uint];lib.cuInit.restype=C.c_int
    lib.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p];lib.cuModuleLoadData.restype=C.c_int
    lib.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p];lib.cuModuleGetFunction.restype=C.c_int
    lib.cuLaunchKernel.argtypes=[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p];lib.cuLaunchKernel.restype=C.c_int
    lib.cuModuleUnload.argtypes=[C.c_void_p];lib.cuModuleUnload.restype=C.c_int
    assert lib.cuInit(0)==0
    a=torch.tensor([0,1,0xffffffff,0x80000000]+list(range(28)),device='cuda',dtype=torch.int64)
    b=torch.flip(a,[0]);aa=a.to(torch.int32);bb=b.to(torch.int32);out1=torch.empty_like(aa);out2=torch.empty_like(aa)
    mod=C.c_void_p();src=C.create_string_buffer(ptx);status=lib.cuModuleLoadData(C.byref(mod),C.cast(src,C.c_void_p));assert status==0,status
    fn=C.c_void_p();status=lib.cuModuleGetFunction(C.byref(fn),mod,b'check_ops');assert status==0,status
    pointers=[C.c_uint64(v.data_ptr()) for v in (aa,bb,out1,out2)];params=(C.c_void_p*4)(*[C.addressof(p) for p in pointers])
    status=lib.cuLaunchKernel(fn,1,1,1,32,1,1,0,None,params,None);assert status==0,status
    torch.cuda.synchronize();sim=PTXIntegerOps().cuda()
    assert torch.equal(out1.to(torch.int64)&0xffffffff,sim(a,b,operation='add.u32'))
    assert torch.equal(out2.to(torch.int64)&0xffffffff,sim(a,b,operation='mul.lo.u32'))
    assert lib.cuModuleUnload(mod)==0
