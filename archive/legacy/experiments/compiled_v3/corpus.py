"""Original, bounded C kernels resembling embedded/data-processing work.

These are authored benchmark programs, not harvested production bugs. Mutations
are injected into compiled machine instructions. All arithmetic is unsigned.
"""
from __future__ import annotations

FAMILIES = {
    'sum_filter': '''
    u32 s=key;
    for (u32 i=0;i<n;i++) if ((a[i]&3u)!=P%4u) s += a[i]*(P|1u);
    return s;''',
    'prefix_scan': '''
    u32 s=key;
    for(u32 i=0;i<n;i++){s+=a[i]+P; out[i]=s;}
    return s;''',
    'dot_product': '''
    u32 s=0;
    for(u32 i=0;i<n;i++) s+=a[i]*a[(i+P)%32u];
    out[0]=s^key; return s;''',
    'rolling_hash': '''
    u32 h=key^P;
    for(u32 i=0;i<n;i++) h=(h^a[i])*16777619u;
    return h;''',
    'adler_style': '''
    u32 s1=(key&65535u)%65521u,s2=(key>>16)%65521u;
    for(u32 i=0;i<n;i++){s1=(s1+(a[i]&255u)+P)%65521u;s2=(s2+s1)%65521u;}
    return (s2<<16)|s1;''',
    'crc_bitwise': '''
    u32 c=~key;
    for(u32 i=0;i<n;i++){c^=(a[i]+P)&255u;
        for(u32 j=0;j<8;j++) c=(c>>1)^(0xedb88320u&(0u-(c&1u)));}
    return ~c;''',
    'saturating_sum': '''
    u32 s=key;
    for(u32 i=0;i<n;i++){u32 x=a[i]+P,t=s+x;s=t<s?0xffffffffu:t;}
    return s;''',
    'insertion_sort': '''
    for(u32 i=1;i<n;i++){u32 v=a[i],j=i;
        while(j>0 && a[j-1]>v){a[j]=a[j-1];j--;}a[j]=v;}
    return n?a[(key+P)%n]:0;''',
    'lower_bound': '''
    u32 lo=0,hi=n,t=key+P;
    /* The search is defined for every array; sortedness is not assumed in tests. */
    while(lo<hi){u32 mid=lo+(hi-lo)/2u;if(a[mid]<t)lo=mid+1;else hi=mid;}
    return lo;''',
    'overlap_move': '''
    u32 off=key%16u,m=n<16u?n:16u;
    /* b may alias a, and the order of stores matters. */
    for(u32 i=m;i>0;i--) a[off+i-1]=a[i-1]+P;
    *b ^= key;return a[off]^*b;''',
    'histogram': '''
    for(u32 i=0;i<n;i++) out[(a[i]+P)&15u]++;
    return out[key&15u];''',
    'run_lengths': '''
    u32 groups=0,last=key;
    for(u32 i=0;i<n;i++){u32 v=a[i]^P;if(i==0||v!=last){out[groups++]=1;}else out[groups-1]++;last=v;}
    return groups;''',
    'parse_decimal': '''
    u32 value=0,used=0;
    for(u32 i=0;i<n;i++){u32 c=a[i]&255u;if(c<48u||c>57u)break;
        u32 d=c-48u;if(value>(0xffffffffu-d)/10u)break;value=value*10u+d;used++;}
    out[0]=used; return value^(key+P);''',
    'ring_buffer': '''
    u32 head=key&31u;
    for(u32 i=0;i<n;i++){u32 v=a[i]+P;out[head]=v;head=(head+1u)&31u;}
    return head;''',
    'linked_walk': '''
    u32 idx=key&31u,s=P;
    for(u32 i=0;i<n;i++){u32 v=a[idx];s+=v;idx=(v>>5)&31u;}
    out[0]=idx;return s;''',
    'rare_dispatch': '''
    u32 s=helper(a,n,key+P);
    if(key==(0xc0de0000u+P)) s=(s^a[0])+a[1];
    if(n==32u && a[0]==(0x1bad0000u+P)) out[31]=s+P;
    return s;''',
}


def source(family: str, variant: int) -> str:
    """Use compile-time parameters, not copies of historical RV32 generators."""
    if family not in FAMILIES or not 0 <= variant < 100:
        raise ValueError('Unknown family or invalid variant')
    p = 3 + variant * 2
    helper = '''__attribute__((noinline)) static u32 helper(u32 *a,u32 n,u32 k){
        u32 s=k;for(u32 i=0;i<n;i++)s=(s+a[i])^(s>>3);return s;
    }''' if family == 'rare_dispatch' else ''
    return ('/* Original LightNest compiled-v3 benchmark, not production source. */\n'
            'typedef unsigned int u32;\n_Static_assert(sizeof(u32)==4,"u32");\n'
            f'#define P {p}u\n' + helper + '\n'
            '__attribute__((visibility("default")))\n'
            'u32 entry(u32 *a,u32 n,u32 key,u32 *b,u32 *out){\n'
            + FAMILIES[family] + '\n}\n')
