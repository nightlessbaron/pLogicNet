import json
import numpy as np
import torch
from metrics import RetrievalMetrics

def generate_graph(data, t2e):
    # with open(f"record/{data}_head_v2/0/pred_kge.txt", "r") as f:
    with open("record/2024-05-23_11:30:32.881053/0/pred_kge.txt", "r") as f:
        lines = f.readlines()
        G = dict()
        i = 0
        while i < len(lines):
            h,r,t,m,rank = lines[i].replace('\n', '').split('\t')
            probs = lines[i+1].replace(' \n', '').split(' ')
            assert len(probs) == len(t2e), f"Length mismatch {len(probs)} != {len(t2e)}"
            
            if m == "h":
                if (m, t, r) not in G and r not in ["sep", "pad"] and t not in ["sep", "pad"]:
                    G[(m, t, r)] = []
                    for prob in probs:
                        rel, score = prob.split(":")
                        if rel in ["sep", "pad"] or h in ["sep", "pad"]:
                            continue
                        G[(m, t, r)].append((rel, float(score)))
                    assert len(G[(m, t, r)]) == len(t2e)-2, f"Length mismatch: {len(G[(m, t, r)])} != {len(t2e)-2}, t: {t}, r: {r}"
            elif m == "t":
                if (m, h, r) not in G and r not in ["sep", "pad"] and h not in ["sep", "pad"]:
                    G[(m, h, r)] = []
                    for prob in probs:
                        rel, score = prob.split(":")
                        if rel in ["sep", "pad"] or h in ["sep", "pad"]:
                            continue
                        G[(m, h, r)].append((rel, float(score)))
                    assert len(G[(m, h, r)]) == len(t2e)-2, f"Length mismatch: {len(G[(m, h, r)])} != {len(t2e)-2}, h: {h}, r: {r}"
            else:
                raise ValueError(f"Invalid mode: {m}")
            i += 2
        return G


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


def f(data, G, t2e, t2r, metrics):
    test_data_path = f"data/{data}/fol_data/f/data.jsonl"
    with open(test_data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            json_line = json.loads(line)
            r = t2r[json_line["p"][0]]
            inputs = [i[0] for i in json_line["o"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            outputs = []
            for v,c in zip(out_v, out_c):
                outputs.append(v+c)
            preds = []
            target = []
            for e, h in enumerate(inputs):
                h = t2e[h]
                preds.append(torch.tensor([g[1] for g in G[("t",h,r)]]))
                target.append(torch.tensor([1.0 if g[0] in outputs[e] else 0.0 for g in G[("t",h,r)]]).bool())
            stacked_preds = torch.stack(preds)
            stacked_targets = torch.stack(target)
            assert stacked_preds.shape == stacked_targets.shape
            assert stacked_preds.shape == (len(inputs), len(t2e)-2), f"Shape mismatch: {stacked_preds.shape} != {(len(inputs), len(t2e)-2)}"
            metrics.update(stacked_preds, stacked_targets)
            scores = metrics.compute()

    print("Query Type -- f (All metrics)")
    for key, value in scores.items():
        print(f"{key}: {value}")
    print("*"*25)


def i(data, G, t2e, t2r, metrics):
    test_data_path = f"data/{data}/fol_data/i/data.jsonl"
    with open(test_data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            json_line = json.loads(line)
            r = t2r[json_line["p"][0]]
            inputs = [i[0] for i in json_line["o"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            outputs = []
            for v,c in zip(out_v, out_c):
                outputs.append(v+c)
            preds = []
            target = []
            for e, t in enumerate(inputs):
                t = t2e[t]
                preds.append(torch.tensor([g[1] for g in G[("h",t,r)]]))
                target.append(torch.tensor([1.0 if g[0] in outputs[e] else 0.0 for g in G[("h",t,r)]]).bool())
            stacked_preds = torch.stack(preds)
            stacked_targets = torch.stack(target)
            assert stacked_preds.shape == stacked_targets.shape
            assert stacked_preds.shape == (len(inputs), len(t2e)-2), f"Shape mismatch: {stacked_preds.shape} != {(len(inputs), len(t2e)-2)}"
            metrics.update(stacked_preds, stacked_targets)
            scores = metrics.compute()

    print("Query Type -- i")
    print(f"All metrics --->")
    for key, value in scores.items():
        print(f"{key}: {value}")
    print("*"*25)


def ff(data, G, t2e, e2t, t2r, metrics):
    test_data_path = f"data/{data}/fol_data/ff/data.jsonl"

    with open(test_data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            inputs = [t2e[i[0]] for i in json_line["o"]]
            # out_t = [[t2e[i[0]] for i in s] for s in json_line["t"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            outputs = []
            for v,c in zip(out_v, out_c):
                outputs.append(v+c)
            matrix_r1 = []
            target = []
            for j, h in enumerate(inputs):
                matrix_r1.append(torch.tensor([g[1] for g in G[("t",h,p1)]]))
                target.append(torch.tensor([1.0 if g[0] in outputs[j] else 0.0 for g in G[("t",h,p1)]]).bool())
            matrix_r1 = torch.stack(matrix_r1)
            stacked_targets = torch.stack(target)

            matrix_r2 = []
            for e in e2t:
                if e not in ["pad", "sep"]:
                    matrix_r2.append(torch.tensor([g[1] for g in G[("t",e,p2)]]))
            matrix_r2 = torch.stack(matrix_r2)

            preds = torch.matmul(matrix_r1, matrix_r2.T)
            assert preds.shape == stacked_targets.shape
            assert preds.shape == (len(inputs), len(t2e)-2), f"Shape mismatch: {preds.shape} != {(len(inputs), len(t2e)-2)}"
            metrics.update(preds, stacked_targets)
            scores = metrics.compute()
    
    print("Query Type -- ff")
    print(f"All metrics --->")
    for key, value in scores.items():
        print(f"{key}: {value}")
    print("*"*25)


def fi(data, G, t2e, e2t, t2r, metrics):
    test_data_path = f"data/{data}/fol_data/fi/data.jsonl"

    with open(test_data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            inputs = [t2e[i[0]] for i in json_line["o"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            outputs = []
            for v,c in zip(out_v, out_c):
                outputs.append(v+c)
            matrix_r1 = []
            target = []
            for j, h in enumerate(inputs):
                matrix_r1.append(torch.tensor([g[1] for g in G[("t",h,p1)]]))
                target.append(torch.tensor([1.0 if g[0] in outputs[j] else 0.0 for g in G[("t",h,p1)]]).bool())
            matrix_r1 = torch.stack(matrix_r1)
            stacked_targets = torch.stack(target)

            matrix_r2 = []
            for e in e2t:
                if e not in ["pad", "sep"]:
                    matrix_r2.append(torch.tensor([g[1] for g in G[("h",e,p2)]]))
            matrix_r2 = torch.stack(matrix_r2)

            preds = torch.matmul(matrix_r1, matrix_r2.T)
            assert preds.shape == stacked_targets.shape
            assert preds.shape == (len(inputs), len(t2e)-2), f"Shape mismatch: {preds.shape} != {(len(inputs), len(t2e)-2)}"
            metrics.update(preds, stacked_targets)
            scores = metrics.compute()
    
    print("Query Type -- fi")
    print(f"All metrics --->")
    for key, value in scores.items():
        print(f"{key}: {value}")
    print("*"*25)


def If(data, G, t2e, e2t, t2r, metrics):
    test_data_path = f"data/{data}/fol_data/if/data.jsonl"

    with open(test_data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            inputs = [t2e[i[0]] for i in json_line["o"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            outputs = []
            for v,c in zip(out_v, out_c):
                outputs.append(v+c)
            matrix_r1 = []
            target = []
            for j, t in enumerate(inputs):
                matrix_r1.append(torch.tensor([g[1] for g in G[("h",t,p1)]]))
                target.append(torch.tensor([1.0 if g[0] in outputs[j] else 0.0 for g in G[("h",t,p1)]]).bool())
            matrix_r1 = torch.stack(matrix_r1)
            stacked_targets = torch.stack(target)

            matrix_r2 = []
            for e in e2t:
                if e not in ["pad", "sep"]:
                    matrix_r2.append(torch.tensor([g[1] for g in G[("t",e,p2)]]))
            matrix_r2 = torch.stack(matrix_r2)

            preds = torch.matmul(matrix_r1, matrix_r2.T)
            assert preds.shape == stacked_targets.shape
            assert preds.shape == (len(inputs), len(t2e)-2), f"Shape mismatch: {preds.shape} != {(len(inputs), len(t2e)-2)}"
            metrics.update(preds, stacked_targets)
            scores = metrics.compute()
    
    print("Query Type -- if")
    print(f"All metrics --->")
    for key, value in scores.items():
        print(f"{key}: {value}")
    print("*"*25)

def ii(data, G, t2e, e2t, t2r, metrics):
    test_data_path = f"data/{data}/fol_data/ii/data.jsonl"
    with open(test_data_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            json_line = json.loads(line)
            p1, p2 = json_line["p"]
            p1, p2 = t2r[p1], t2r[p2]
            
            inputs = [t2e[i[0]] for i in json_line["o"]]
            out_v = [[t2e[i[0]] for i in s] for s in json_line["v"]]
            out_c = [[t2e[i[0]] for i in s] for s in json_line["c"]]
            outputs = []
            for v,c in zip(out_v, out_c):
                outputs.append(v+c)
            matrix_r1 = []
            target = []
            for j, t in enumerate(inputs):
                matrix_r1.append(torch.tensor([g[1] for g in G[("h",t,p1)]]))
                target.append(torch.tensor([1.0 if g[0] in outputs[j] else 0.0 for g in G[("h",t,p1)]]).bool())
            matrix_r1 = torch.stack(matrix_r1)
            stacked_targets = torch.stack(target)

            matrix_r2 = []
            for e in e2t:
                if e not in ["pad", "sep"]:
                    matrix_r2.append(torch.tensor([g[1] for g in G[("h",e,p2)]]))
            matrix_r2 = torch.stack(matrix_r2)

            preds = torch.matmul(matrix_r1, matrix_r2.T)
            assert preds.shape == stacked_targets.shape
            assert preds.shape == (len(inputs), len(t2e)-2), f"Shape mismatch: {preds.shape} != {(len(inputs), len(t2e)-2)}"
            metrics.update(preds, stacked_targets)
            scores = metrics.compute()
    
    print("Query Type -- ii")
    print(f"All metrics --->")
    for key, value in scores.items():
        print(f"{key}: {value}")
    print("*"*25)

if __name__ == "__main__":
    data = "kinship"

    t2e, e2t = getDict(data, "entities")
    t2r, r2t = getDict(data, "relations")
    metrics = RetrievalMetrics(ks_for_hits=[1,3,5,10])

    G = generate_graph(data, t2e)
    graph_size = sum([len(G[i]) for i in G])
    expected_size = (len(t2e)-2)**2 * len(t2r) * 2
    assert graph_size == expected_size, f"Something is wrong with the graph generation {graph_size} != {expected_size}."

    # --------

    f(data, G, t2e, t2r, metrics)
    i(data, G, t2e, t2r, metrics)

    # --------

    ff(data, G, t2e, e2t, t2r, metrics)
    fi(data, G, t2e, e2t, t2r, metrics)
    If(data, G, t2e, e2t, t2r, metrics)
    ii(data, G, t2e, e2t, t2r, metrics)