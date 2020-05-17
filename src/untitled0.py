l = ["AREA", "BALL", "DEAR", "LADY", "LEAD", "YARD", "ROME"]
dic = {}
for m in l:
    ini = m[0]
    if dic.get(ini, False):
        dic[ini].append(m)
    else:
        dic[ini] = [m]

res = []
for m in l:
    seq = []
    for k in m:
        words = dic.get(k, [])
        appended = False
        for w in words:
            if w not in seq:
                seq.append(w)
                appended = True
                break
        if not appended:
            break
    if len(seq) == len(m):
        res.append((m, seq))
        
print(res)
    