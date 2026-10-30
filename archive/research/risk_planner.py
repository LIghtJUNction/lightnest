"""Two-query finite-catalogue identification. Supplied candidates, not synthesis."""
from __future__ import annotations
import numpy as np


def validate(h, p):
    h=np.asarray(h,np.uint8);p=np.asarray(p,np.float64)
    if h.ndim!=2 or h.shape[0]!=8 or p.shape!=(8,) or not np.isin(h,(0,1)).all():
        raise ValueError('expected eight Boolean candidate rows and eight weights')
    if not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:raise ValueError('invalid weights')
    return h,p/p.sum()


def pair_targets(h,p):
    """[first query, second query, first answer]; exact leaf mass sums."""
    h,p=validate(h,p)
    hx=h.T[:,None,:];hz=h.T[None,:,:]
    v=np.empty((h.shape[1],h.shape[1],2))
    for a in (0,1):
        v[:,:,a]=sum(np.max(np.where((hx==a)&(hz==b),p,0),axis=-1) for b in (0,1))
    return v


def query_values(h,p):
    joint=pair_targets(h,p)
    one=np.array([joint[x,x].sum() for x in range(h.shape[1])])
    two=joint.max(axis=1).sum(axis=1)
    return one,two,joint


def pair_features(h,p):
    h,p=validate(h,p)
    order=np.argsort(-p,kind='stable');h=h[order];p=p[order]
    n=h.shape[1]
    return np.concatenate([np.broadcast_to(p,(n,n,8)),
        np.broadcast_to(h.T[:,None,:],(n,n,8)),np.broadcast_to(h.T[None,:,:],(n,n,8))],axis=-1).astype(np.float32)


def choose(h,p,policy,budget,model=None):
    h,p=validate(h,p)
    if policy=='entropy':
        q=p@h; v=-(q*np.log2(np.clip(q,1e-300,1))+(1-q)*np.log2(np.clip(1-q,1e-300,1)))
    elif policy in ('risk1','risk2'):
        one,two,_=query_values(h,p);v=two if policy=='risk2' and budget==2 else one
    elif policy=='neural':
        import torch
        with torch.inference_mode():
            z=model(torch.from_numpy(pair_features(h,p)).reshape(-1,24)).numpy().reshape(h.shape[1],h.shape[1],2)
        v=z.max(axis=1).sum(axis=1) if budget==2 else np.array([z[x,x].sum() for x in range(h.shape[1])])
    else:raise ValueError(policy)
    # Do not query a bit already constant on nonzero-posterior candidates.
    eligible=np.any(h[p>0]==0,axis=0)&np.any(h[p>0]==1,axis=0)
    return int(np.argmax(np.where(eligible,v,-np.inf))) if eligible.any() else 0


def tree_success(h,p,policy,model=None):
    """Identify each of the eight targets; score under p and under uniform truth."""
    h,p=validate(h,p)
    if policy=='random':
        n=h.shape[1]; w=0.;u=0.
        # Uniform distinct query pairs; the second draw may equivalently be adaptive.
        for x in range(n):
            for z in range(n):
                if x==z:continue
                for a in (0,1):
                    for b in (0,1):
                        ids=np.flatnonzero((h[:,x]==a)&(h[:,z]==b))
                        if len(ids):
                            winner=ids[np.argmax(p[ids])];w+=p[winner];u+=1/8
        return dict(weighted=w/(n*(n-1)),uniform=u/(n*(n-1)),first_query=None,branches=[])
    x=choose(h,p,policy,2,model)
    w=0.;u=0.;branches=[]
    for a in (0,1):
        p1=p*(h[:,x]==a)
        if p1.sum()==0:continue
        z=choose(h,p1,policy,1,model)
        for b in (0,1):
            ids=np.flatnonzero((h[:,x]==a)&(h[:,z]==b))
            if len(ids):
                winner=int(ids[np.argmax(p[ids])]);w+=p[winner];u+=1/8
                branches.append(dict(first_answer=a,second_query=int(z),second_answer=b,winner=winner))
    return dict(weighted=float(w),uniform=float(u),first_query=x,branches=branches)


def entropy2_values(h, p):
    """Exact two-query expected leaf entropy, for horizon-matched comparisons.

    This is distinct from maximizing final correct-identification probability.
    """
    h, p = validate(h, p)
    first = h.T[:, None, :]
    second = h.T[None, :, :]
    scores = np.zeros((h.shape[1], h.shape[1], 2), dtype=float)
    for answer in (0, 1):
        for reply in (0, 1):
            mass = np.sum(np.where((first == answer) & (second == reply), p, 0), axis=-1)
            scores[:, :, answer] -= mass * np.log2(np.maximum(mass, 1e-300))
    return scores.max(axis=1).sum(axis=1)


def entropy2_tree(h, p):
    """Execute optimal two-step entropy root and greedy second query; no labels read."""
    h, p = validate(h, p)
    query = int(np.argmax(entropy2_values(h, p)))
    weighted = 0.0
    uniform = 0.0
    for answer in (0, 1):
        p1 = p * (h[:, query] == answer)
        if p1.sum() == 0:
            continue
        second = choose(h, p1, 'entropy', 1)
        for reply in (0, 1):
            ids = np.flatnonzero((h[:, query] == answer) & (h[:, second] == reply))
            if len(ids):
                weighted += float(p[ids].max())
                uniform += 1 / 8
    return {'weighted': weighted, 'uniform': uniform, 'first_query': query}
