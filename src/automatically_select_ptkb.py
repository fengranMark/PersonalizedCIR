import json



def extent_qrel(one_ptkb_path, qrels_path, qrels_extent_path):  # In order to facilitate direct retrieval, we have expanded the qrel file to correspond with the output file of add_ptkb_one_by_one.py
    with open(one_ptkb_path) as f1:
        lines  = f1.readlines()
    with open(qrels_path, "r") as f2:
        qrel = f2.readlines()

    dic1 = dict()
    dic2 = dict()
    l = []   
    for line in lines:
        line = json.loads(line.strip())
        id2 = line["sample_id"]
        id2 = id2.rsplit("-", 1)[0]
        if id2 not in dic1:
            dic1[id2]=0
        dic1[id2]+=1

    for line in qrel:
        line = line.replace("_", "-")
        id1 = line.split(" ")[0].replace("_", "-")
        if id1 not in dic2:
            l = []
            dic2[id1] = l
        dic2[id1].append(line)
    for k,v in dic1.items():
        for i in range(1,v+1):
            lis = dic2[k]
            lis = [x.split()[0] + "-" + str(i) +" " + x.split(maxsplit=1)[1] for x in lis]
            with open(qrels_extent_path, 'a+') as output_file:
                output_file.writelines(lis)



def choose_ptkb(input_path, no_ptkb_scores, one_ptkb_scores, threshold=0):  # Select beneficial PTKBs (multiple can be selected in one round) by comparing the ndcg3 of without ptkb and one ptkb.
    id = []
    ptkb_amount = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            sample_id = data.get('sample_id','')
            ptkb = data.get('ptkb', '')
            ptkb_amount.append(len(ptkb))
            id.append(sample_id)

    # Split by different IDs   [1818] -> [[]*176]
    one_ptkb_scores_splited = []
    for amount in ptkb_amount:
        end = start + amount
        one_ptkb_scores_splited.append(one_ptkb_scores[start:end])
        start = end

    ptkb_pro = {}
    for i in range(0,176):
        ptkb_num = 0
        for j in one_ptkb_scores_splited[i]:
            ptkb_num += 1
            if id[i] not in ptkb_pro:
                ptkb_pro[id[i]] = []
            if j-no_ptkb_scores[i] > threshold:
                ptkb_pro[id[i]].append(ptkb_num)
    return ptkb_pro
