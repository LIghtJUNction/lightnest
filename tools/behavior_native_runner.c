/* Bounded native research runner. No security certification.
   Usage: runner function.bin cases.bin
   cases.bin: little-endian uint32 count; repeated uint32 length + raw bytes.
   Input buffers are read-only and end immediately before a PROT_NONE page.
   GCC/Linux/x86-64 only. All I/O is completed before candidate execution. */
#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <stddef.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <grp.h>

static int exact(int fd, void *p, size_t n) {
    size_t off=0;
    while(off<n) { ssize_t r=read(fd,(char*)p+off,n-off); if(r<=0)return -1; off+=(size_t)r; }
    return 0;
}
static int filter(void) {
    struct sock_filter f[]={
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,arch)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AUDIT_ARCH_X86_64,1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_exit,6,0),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_exit_group,5,0),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_rt_sigreturn,4,0),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_write,0,4),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[0])),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,1,1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS)
    };
    struct sock_fprog p={(unsigned short)(sizeof(f)/sizeof(f[0])),f};
    return prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0) || prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&p);
}
int main(int argc,char **argv) {
    if(argc!=3)return 101;
    int fd=open(argv[1],O_RDONLY|O_NOFOLLOW);
    struct stat st;
    if(fd<0 || fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<1 || st.st_size>4096)return 102;
    void *code=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(code==MAP_FAILED || exact(fd,code,(size_t)st.st_size))return 103;
    close(fd);
    if(mprotect(code,4096,PROT_READ|PROT_EXEC))return 104;
    fd=open(argv[2],O_RDONLY|O_NOFOLLOW);
    uint32_t count;
    if(fd<0 || fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size>16*1024*1024 ||
       exact(fd,&count,4) || count<1 || count>512)return 105;
    const uint8_t *inputs[512];
    uint32_t lengths[512];
    uint64_t output[512];
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    if(page!=4096)return 106;
    for(uint32_t i=0;i<count;i++) {
        uint32_t n;
        if(exact(fd,&n,4) || n>131072)return 107;
        size_t pages=((size_t)n+page-1)/page;
        if(pages==0)pages=1;
        size_t mid=pages*page, total=mid+2*page;
        uint8_t *base=mmap(NULL,total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(base==MAP_FAILED || mprotect(base+page,mid,PROT_READ|PROT_WRITE))return 108;
        uint8_t *buf=base+page+mid-n;
        if(exact(fd,buf,n) || mprotect(base+page,mid,PROT_READ))return 109;
        inputs[i]=buf; lengths[i]=n;
    }
    uint8_t extra;
    if(read(fd,&extra,1)!=0)return 110;
    close(fd);
    struct rlimit cpu={1,1},as={256*1024*1024,256*1024*1024},core={0,0},fs={8192,8192};
    if(setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_AS,&as)||
       setrlimit(RLIMIT_CORE,&core)||setrlimit(RLIMIT_FSIZE,&fs))return 111;
    if(geteuid()==0 && (setgroups(0,NULL)||setgid(65534)||setuid(65534)))return 112;
    for(int i=3;i<1024;i++)close(i);
    if(filter())return 113;
    uint64_t (*fn)(const uint8_t*,uint64_t)=code;
    for(uint32_t i=0;i<count;i++)output[i]=fn(inputs[i],lengths[i]);
    long wrote=syscall(SYS_write,1,output,8*count);
    syscall(SYS_exit_group,wrote==8*count?0:114);
    return 0;
}
