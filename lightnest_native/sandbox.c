#define _GNU_SOURCE
#include <errno.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <linux/audit.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Load a bounded code block and uint64 input triples from stdin before filtering.
 * Candidate execution cannot open files, access a network, fork, exec or mmap.
 * Only exit and a single bounded stdout result write remain available.
 */
static int read_all(void *p, size_t n) {
    unsigned char *b=p;
    while(n) { ssize_t r=read(0,b,n); if(r<=0)return -1; b+=r;n-=r; }
    return 0;
}

int main(void) {
#if !defined(__x86_64__)
    return 90;
#endif
    uint32_t sizes[2];
    if(read_all(sizes,sizeof(sizes)) || sizes[0]==0 || sizes[0]>16384 || sizes[1]==0 || sizes[1]>64)return 91;
    size_t page=(size_t)sysconf(_SC_PAGESIZE), n=(sizes[0]+page-1)/page*page;
    unsigned char *mapping=mmap(NULL,n+2*page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(mapping==MAP_FAILED || mprotect(mapping+page,n,PROT_READ|PROT_WRITE))return 92;
    unsigned char *code=mapping+page;
    if(read_all(code,sizes[0]) || mprotect(code,n,PROT_READ|PROT_EXEC))return 93;
    uint64_t inputs[64][3], outputs[64];
    if(read_all(inputs,sizes[1]*3*sizeof(uint64_t)))return 94;
    struct rlimit cpu={1,1},mem={256*1024*1024,256*1024*1024},core={0,0},size={512,512};
    if(setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_AS,&mem)||setrlimit(RLIMIT_CORE,&core)||setrlimit(RLIMIT_FSIZE,&size))return 95;
    struct sock_filter filter[]={
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,arch)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AUDIT_ARCH_X86_64,1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_exit,12,0),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_exit_group,11,0),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_write,1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[0])),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,1,0,6),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[2])+4),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,0,1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[2])),
        BPF_JUMP(BPF_JMP|BPF_JGT|BPF_K,sizeof(outputs),1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program={.len=(unsigned short)(sizeof(filter)/sizeof(filter[0])),.filter=filter};
    if(prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)||prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&program))return 96;
    uint64_t (*function)(uint64_t,uint64_t,uint64_t)=(void *)code;
    for(uint32_t i=0;i<sizes[1];i++)outputs[i]=function(inputs[i][0],inputs[i][1],inputs[i][2]);
    size_t bytes=sizes[1]*sizeof(uint64_t);
    if(write(1,outputs,bytes)!=(ssize_t)bytes)_exit(97);
    _exit(0);
}
