import json
import numpy as np
import torch
from metrics import RetrievalMetrics

def generate_graph(data):
    with open(f"record/{data}_head/0/pred_kge.txt", "r") as f:
        lines = f.readlines()
        G = set()
        i = 0
        while i < len(lines):
            h,r,t,m,rank = lines[i].split('\t')
            probs = lines[i+1].split(' ')[:-1]
            if m == "h":
                for prob in probs:
                    rel, score = prob.split(":")
                    if float(score) > 0.5:
                        G.add((rel, r, t, float(score)))
            else:
                for prob in probs:
                    rel, score = prob.split(":")
                    if float(score) > 0.5:
                        G.add((h, r, rel, float(score)))
            i += 2
        return G

def getTailEntity(G):
    h2t = {}
    for g in G:
        h,r,t,s = g
        if (h,r) not in h2t:
            h2t[(h,r)] = []
        h2t[(h,r)].append((t,s))
    return h2t


def getHeadEntity(G):
    t2h = {}
    for g in G:
        h,r,t,s = g
        if (t,r) not in t2h:
            t2h[(t,r)] = []
        t2h[(t,r)].append((h,s))
    return t2h


def getDict(data, typ="entities"):
    with open(f"data/{data}/{typ}.dict", "r") as f:
        ed = {}
        de = {}
        lines = f.readlines()
        for line in lines:
            i, e = line.replace('\n', '').split('\t')
            ed[int(i)] = e
            de[e] = int(i)
    return ed, de

# TODO: remove if conditions
def f(data, h2t, metrics):
    test_data_path = f"data/{data}/test.txt"
    with open(test_data_path, "r") as f:
        lines = f.readlines()
        hits_at_1, hits_at_3, hits_at_5, hits_at_10 = 0,0,0,0
        for line in lines:
            h,r,t = line.split('\t')
            t = t.replace('\n','')
            if (h,r) in h2t:
                preds = torch.tensor([g[1] for g in h2t[(h,r)]]).unsqueeze(0)
                target = torch.tensor([1.0 if g[0] == t else 0.0 for g in h2t[(h,r)]]).unsqueeze(0).bool()
                metrics.update(preds, target)
                scores = metrics.compute()
                hits_at_1 += scores["hits_at_1"].item() / len(lines)
                hits_at_3 += scores["hits_at_3"].item() / len(lines)
                hits_at_5 += scores["hits_at_5"].item() / len(lines)
                hits_at_10 += scores["hits_at_10"].item() / len(lines)
    print("Query Type -- f")
    print(f"Hits@1: {hits_at_1}")
    print(f"Hits@3: {hits_at_3}")
    print(f"Hits@5: {hits_at_5}")
    print(f"Hits@10: {hits_at_10}")
    print("*"*25)


def i(data, t2h, metrics):
    test_data_path = f"data/{data}/test.txt"
    with open(test_data_path, "r") as f:
        lines = f.readlines()
        hits_at_1, hits_at_3, hits_at_5, hits_at_10 = 0,0,0,0
        for line in lines:
            h,r,t = line.split('\t')
            t = t.replace('\n','')
            if (t,r) in t2h:
                preds = torch.tensor([g[1] for g in t2h[(t,r)]]).unsqueeze(0)
                target = torch.tensor([1.0 if g[0] == t else 0.0 for g in t2h[(t,r)]]).unsqueeze(0).bool()
                metrics.update(preds, target)
                scores = metrics.compute()
                hits_at_1 += scores["hits_at_1"].item() / len(lines)
                hits_at_3 += scores["hits_at_3"].item() / len(lines)
                hits_at_5 += scores["hits_at_5"].item() / len(lines)
                hits_at_10 += scores["hits_at_10"].item() / len(lines)
    print("Query Type -- i")
    print(f"Hits@1: {hits_at_1}")
    print(f"Hits@3: {hits_at_3}")
    print(f"Hits@5: {hits_at_5}")
    print(f"Hits@10: {hits_at_10}")
    print("*"*25)


def ff(data, h2t, t2e, t2r):
    test_data_path = f"data/{data}/fol_data/ff/data.jsonl"
    m1 = np.zeros((len(t2e), len(t2e)))
    with open(test_data_path, "r") as f:
        lines = f.readlines()
        p = 0
        count = 0
        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            inputs = [t2e[i[0]] for i in json_line["o"]]
            # out_t = [[t2e[i[0]] for i in s] for s in json_line["t"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            out = []
            # for t,v,c in zip(out_t, out_v, out_c):
            #     out.append(t + v + c)
            for v,c in zip(out_v, out_c):
                out.append(v+c)

            # 
            
            for e,i in enumerate(inputs):
                count += 1
                h,r = i, p1
                if (h,r) in h2t:
                    for tr, sr in h2t[(h,r)]:
                        p_curr = sr / sum(tup[1] for tup in h2t[(h,r)])
                        n_h, n_r = tr, p2
                        if (n_h,n_r) in h2t:
                            p_new = 0
                            for tx, sx in h2t[(n_h,n_r)]:
                                if tx in out[e]:
                                    p_new += sx / sum(tup[1] for tup in h2t[(n_h,n_r)])
                        p += p_curr * p_new
        p /= count
    print(p)


def fi(data, h2t, t2h, t2e, t2r):
    test_data_path = f"data/{data}/fol_data/ff/data.jsonl"
    with open(test_data_path, "r") as f:
        lines = f.readlines()
        p = 0
        count = 0
        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            inputs = [t2e[i[0]] for i in json_line["o"]]
            # out_t = [[t2e[i[0]] for i in s] for s in json_line["t"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            out = []
            # for t,v,c in zip(out_t, out_v, out_c):
            #     out.append(t + v + c)
            for v,c in zip(out_v, out_c):
                out.append(v+c)
            
            for e,i in enumerate(inputs):
                count += 1
                h,r = i, p1
                if (h,r) in h2t:
                    for tr, sr in h2t[(h,r)]:
                        p_curr = sr / sum(tup[1] for tup in h2t[(h,r)])
                        n_t, n_r = tr, p2
                        if (n_t,n_r) in t2h:
                            p_new = 0
                            for hx, sx in t2h[(n_t,n_r)]:
                                if hx in out[e]:
                                    p_new += sx / sum(tup[1] for tup in t2h[(n_t,n_r)])
                        p += p_curr * p_new
        p /= count
    print(p)


def If(data, h2t, t2h, t2e, t2r):
    test_data_path = f"data/{data}/fol_data/ff/data.jsonl"
    with open(test_data_path, "r") as f:
        lines = f.readlines()
        p = 0
        count = 0
        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            inputs = [t2e[i[0]] for i in json_line["o"]]
            # out_t = [[t2e[i[0]] for i in s] for s in json_line["t"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            out = []
            # for t,v,c in zip(out_t, out_v, out_c):
            #     out.append(t + v + c)
            for v,c in zip(out_v, out_c):
                out.append(v+c)
            
            for e,i in enumerate(inputs):
                count += 1
                t,r = i, p1
                if (t,r) in t2h:
                    for hr, sr in t2h[(t,r)]:
                        p_curr = sr / sum(tup[1] for tup in t2h[(t,r)])
                        n_h, n_r = hr, p2
                        if (n_h,n_r) in h2t:
                            p_new = 0
                            for tx, sx in h2t[(n_h,n_r)]:
                                if tx in out[e]:
                                    p_new += sx / sum(tup[1] for tup in h2t[(n_h,n_r)])
                        p += p_curr * p_new
        p /= count
    print(p)


def ii(data, h2t, t2h, t2e, t2r):
    test_data_path = f"data/{data}/fol_data/ff/data.jsonl"
    with open(test_data_path, "r") as f:
        lines = f.readlines()
        p = 0
        count = 0
        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            inputs = [t2e[i[0]] for i in json_line["o"]]
            # out_t = [[t2e[i[0]] for i in s] for s in json_line["t"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            out = []
            # for t,v,c in zip(out_t, out_v, out_c):
            #     out.append(t + v + c)
            for v,c in zip(out_v, out_c):
                out.append(v+c)
            
            for e,i in enumerate(inputs):
                count += 1
                t,r = i, p1
                if (t,r) in t2h:
                    for hr, sr in t2h[(t,r)]:
                        p_curr = sr / sum(tup[1] for tup in t2h[(t,r)])
                        n_t, n_r = hr, p2
                        if (n_t,n_r) in t2h:
                            p_new = 0
                            for hx, sx in t2h[(n_t,n_r)]:
                                if hx in out[e]:
                                    p_new += sx / sum(tup[1] for tup in t2h[(n_t,n_r)])
                        p += p_curr * p_new
        p /= count
    print(p)

if __name__ == "__main__":
    data = "umls"
    G = generate_graph(data)
    metrics = RetrievalMetrics(ks_for_hits=[1,3,5,10])
    t2e, e2t = getDict(data, "entities")
    t2r, r2t = getDict(data, "relations")
    h2t = getTailEntity(G)
    t2h = getHeadEntity(G)

    # --------

    f(data, h2t, metrics)
    i(data, t2h, metrics)

    # --------

    ff(data, h2t, t2e, t2r)
    fi(data, h2t, t2h, t2e, t2r)
    If(data, h2t, t2h, t2e, t2r)
    ii(data, h2t, t2h, t2e, t2r)