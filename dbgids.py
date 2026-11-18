import torch
from attn_model import AttnDiac
m=AttnDiac(nchar=60,max_c=8,syl=16,dim=32,window=8)
a={id(p) for n,p in m.named_parameters() if n.startswith("out.")}
b={id(p) for n,p in m.named_parameters() if n.startswith("out.")}
print("ids equal across calls:", a==b, len(a), len(b))
na=[n for n,p in m.named_parameters() if id(p) not in a]
print("encoder params kept:", len(na))
print("out params kept:", [n for n,p in m.named_parameters() if id(p) in a])
