/* Bounded RV32IM interpreter. Candidate instructions are DATA, not native code.
 * Little-endian, explicit mapped memory, no syscalls or guest host access.
 * ISA: RV32I 2.1 + M 2.0. No compressed, privileged, floating or atomic ISA.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#define CODE 0x1000u
#define MAX_CODE 16384u
#define DATA 0x20000u
#define OUT 0x21000u
#define STACK 0x30000u
#define STACK_BYTES 16384u
#define RETURN_PC 0x800u
#define FUEL 30000u

typedef struct {uint32_t a[32],n,key,alias;} Input;
typedef struct {uint32_t value,a[32],out[32],status,steps;} Output;
typedef struct {uint8_t data[128],out[128],stack[STACK_BYTES];} Memory;
static int32_t sx(uint32_t x, unsigned n){
    uint32_t sign=1u<<(n-1);return (int32_t)((x^sign)-sign);
}
static uint32_t asr(uint32_t x,uint32_t y){
    y&=31u;if(!y)return x;return (x>>y)|((0u-(x>>31))<<(32u-y));
}
static uint8_t *ptr(Memory*m,const uint8_t*code,uint32_t size,uint32_t a,unsigned w,int write){
    if((w!=1&&w!=2&&w!=4)||(a&(w-1u)))return NULL;
    if(a>=DATA&&a-DATA<=128u-w)return m->data+(a-DATA);
    if(a>=OUT&&a-OUT<=128u-w)return m->out+(a-OUT);
    if(a>=STACK&&a-STACK<=STACK_BYTES-w)return m->stack+(a-STACK);
    if(!write&&size>=w&&a>=CODE&&a-CODE<=size-w)return (uint8_t*)code+(a-CODE);
    return NULL;
}
static uint32_t readle(const uint8_t*p,unsigned w){
    uint32_t v=0;for(unsigned j=0;j<w;j++)v|=(uint32_t)p[j]<<(8u*j);return v;
}
static void writele(uint8_t*p,unsigned w,uint32_t v){
    for(unsigned j=0;j<w;j++)p[j]=(uint8_t)(v>>(8u*j));
}
int vm_run(const uint8_t*code,uint32_t size,uint32_t entry,const Input*in,Output*out,uint8_t*trace){
    Memory mem;uint32_t r[32]={0},pc=entry,status=0,steps=0;
    memset(&mem,0,sizeof(mem));memset(out,0,sizeof(*out));
    if(!size||size>MAX_CODE||(size&3u)||entry<CODE||entry>=CODE+size||in->n>32u||in->alias>=32u){out->status=5;return 5;}
    for(unsigned j=0;j<32;j++)writele(mem.data+4*j,4,in->a[j]);
    r[1]=RETURN_PC;r[2]=STACK+STACK_BYTES-16u;
    r[10]=DATA;r[11]=in->n;r[12]=in->key;r[13]=DATA+4*in->alias;r[14]=OUT;
    for(steps=0;steps<FUEL;steps++){
        if(pc==RETURN_PC)break;
        if(pc<CODE||pc-CODE>=size||(pc&3u)){status=1;break;}
        unsigned index=(pc-CODE)/4u;if(trace)trace[index]=1;
        uint32_t w=readle(code+pc-CODE,4),op=w&127u,d=(w>>7)&31u,f=(w>>12)&7u;
        uint32_t a=(w>>15)&31u,b=(w>>20)&31u,h=w>>25,x=r[a],y=r[b],v=0,npc=pc+4u;
        int set=1;int32_t imm=sx(w>>20,12);
        if(op==0x37u)v=w&0xfffff000u;
        else if(op==0x17u)v=pc+(w&0xfffff000u);
        else if(op==0x6fu){
            uint32_t j=((w>>31)<<20)|(((w>>12)&255u)<<12)|(((w>>20)&1u)<<11)|(((w>>21)&1023u)<<1);
            v=npc;npc=pc+(uint32_t)sx(j,21);
        }else if(op==0x67u&&f==0){v=npc;npc=(x+(uint32_t)imm)&~1u;}
        else if(op==0x63u){
            uint32_t off=((w>>31)<<12)|(((w>>7)&1u)<<11)|(((w>>25)&63u)<<5)|(((w>>8)&15u)<<1);
            int take=0;set=0;
            if(f==0)take=x==y;else if(f==1)take=x!=y;else if(f==4)take=(int32_t)x<(int32_t)y;
            else if(f==5)take=(int32_t)x>=(int32_t)y;else if(f==6)take=x<y;else if(f==7)take=x>=y;
            else{status=3;break;}if(take)npc=pc+(uint32_t)sx(off,13);
        }else if(op==3u){
            unsigned width=(f==0||f==4)?1:(f==1||f==5)?2:f==2?4:0;
            uint8_t*p=ptr(&mem,code,size,x+(uint32_t)imm,width,0);if(!p){status=2;break;}
            /* A program may read its own instruction bytes as data. Such
             * locations must also survive trace-guided candidate pruning. */
            uint32_t address=x+(uint32_t)imm;
            if(trace&&address>=CODE&&address-CODE<size)
                for(unsigned j=0;j<width;j++)trace[(address-CODE+j)/4u]=1;
            v=readle(p,width);if(f==0)v=(uint32_t)sx(v,8);else if(f==1)v=(uint32_t)sx(v,16);
        }else if(op==0x23u){
            unsigned width=f==0?1:f==1?2:f==2?4:0;
            uint32_t off=((w>>25)<<5)|((w>>7)&31u);
            uint8_t*p=ptr(&mem,code,size,x+(uint32_t)sx(off,12),width,1);
            if(!p){status=2;break;}writele(p,width,y);set=0;
        }else if(op==0x13u){
            if(f==0)v=x+(uint32_t)imm;else if(f==2)v=(int32_t)x<imm;
            else if(f==3)v=x<(uint32_t)imm;else if(f==4)v=x^(uint32_t)imm;
            else if(f==6)v=x|(uint32_t)imm;else if(f==7)v=x&(uint32_t)imm;
            else if(f==1&&h==0)v=x<<b;else if(f==5&&h==0)v=x>>b;
            else if(f==5&&h==32)v=asr(x,b);else{status=3;break;}
        }else if(op==0x33u){
            if(h==0){
                if(f==0)v=x+y;else if(f==1)v=x<<(y&31);else if(f==2)v=(int32_t)x<(int32_t)y;
                else if(f==3)v=x<y;else if(f==4)v=x^y;else if(f==5)v=x>>(y&31);
                else if(f==6)v=x|y;else v=x&y;
            }else if(h==32&&(f==0||f==5))v=f==0?x-y:asr(x,y);
            else if(h==1){
                if(f==0)v=x*y;
                else if(f==1)v=(uint32_t)((uint64_t)((int64_t)(int32_t)x*(int64_t)(int32_t)y)>>32);
                else if(f==2)v=(uint32_t)((uint64_t)((int64_t)(int32_t)x*(int64_t)y)>>32);
                else if(f==3)v=(uint32_t)(((uint64_t)x*(uint64_t)y)>>32);
                else if(f==4)v=!y?UINT32_MAX:(x==0x80000000u&&y==UINT32_MAX)?x:(uint32_t)((int32_t)x/(int32_t)y);
                else if(f==5)v=!y?UINT32_MAX:x/y;
                else if(f==6)v=!y?x:(x==0x80000000u&&y==UINT32_MAX)?0:(uint32_t)((int32_t)x%(int32_t)y);
                else v=!y?x:x%y;
            }else{status=3;break;}
        }else if(op==0x0fu&&f==0){set=0;}
        else{status=3;break;}
        if(set&&d)r[d]=v;r[0]=0;pc=npc;
    }
    if(steps==FUEL)status=4;
    out->value=r[10];out->status=status;out->steps=steps;
    for(unsigned j=0;j<32;j++){out->a[j]=readle(mem.data+4*j,4);out->out[j]=readle(mem.out+4*j,4);}
    return (int)status;
}
int vm_equal(const Output*a,const Output*b){
    return a->status==0&&b->status==0&&a->value==b->value&&
        !memcmp(a->a,b->a,sizeof(a->a))&&!memcmp(a->out,b->out,sizeof(a->out));
}
/* Return first failed test. This bounds work for incorrect candidates. */
int vm_check(const uint8_t*code,uint32_t size,uint32_t entry,const Input*inputs,const Output*expected,
             uint32_t n,uint64_t*work){
    Output out;for(uint32_t i=0;i<n;i++){
        vm_run(code,size,entry,inputs+i,&out,NULL);work[0]++;work[1]+=out.steps;
        if(!vm_equal(&out,expected+i))return (int)i;
    }return -1;
}
void vm_batch(const uint8_t*code,uint32_t size,uint32_t entry,const Input*inputs,Output*outputs,uint32_t n){
    for(uint32_t i=0;i<n;i++)vm_run(code,size,entry,inputs+i,outputs+i,NULL);
}
