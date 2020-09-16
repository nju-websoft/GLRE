import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default=str)
parser.add_argument("--output_path", type=str)
inp = parser.parse_args()

input_file = "./data/DocRED/test.json"
ori_data = json.load(open(input_file))
docid = 0
docid2title = {}
for doc in ori_data:
    docid2title[docid] = doc['title']
    docid += 1

index = 0
result = []
# "./docred-dev/MLRGNN_lstm2/test.errors
with open(inp.input_path, "r", encoding="utf-8") as f:
    title = ""
    h_idx = ""
    t_idx = ""
    r = ""
    evidence = []
    for line in f.readlines():
        line = line.strip()
        index += 1
        if index % 7 == 1:
            r = str(line.split("\t")[0].split(":")[1]).strip()
        elif index % 7 == 2:
            title = docid2title[int(line)]
        elif index % 7 ==3:
            continue
        elif index % 7 == 4:
            h_idx = line.split("|")[0].split(":")[1].strip()
        elif index % 7 == 5:
            t_idx = line.split("|")[0].split(":")[1].strip()
        elif index % 7 == 6:
            continue
        else:
            temp = {"title": title, "h_idx": int(h_idx), "t_idx": int(t_idx), "r": r, "evidence": evidence}
            result.append(temp)

json.dump(result, open(inp.output_path, "w", encoding="utf-8"))