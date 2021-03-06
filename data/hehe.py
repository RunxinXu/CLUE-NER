import json

with open('pred.json') as f:
    pred = json.load(f)
gold = []
with open('test.json') as f:
    for line in f:
        gold.append(json.loads(line))

KIND = ['name', 'book', 'organization', 'company', 'game', 'address', 'scene', 'government', 'position', 'movie']
KIND2ID = dict()
ID2KIND = dict()
for i in range(len(KIND)):
    KIND2ID[KIND[i]] = i
    ID2KIND[i] = KIND[i]

tp = [0 for _ in range(10)]
fp = [0 for _ in range(10)]
fn = [0 for _ in range(10)]
precision = [0 for _ in range(10)]
recall = [0 for _ in range(10)]
f1 = [0 for _ in range(10)]

occur_times = [[0, 0] for _ in range(10)] # total, times
occur_times_avg = [0 for _ in range(10)]
span_length = [[0, 0] for _ in range(10)]
span_length_avg = [0 for _ in range(10)]

badcases = []

for p, g in zip(pred, gold):
    text = g['text']
    fp_entity = []
    fn_entity = []

    g = g['label']
    for idx, kind in enumerate(KIND):
        pred_entity = p.get(kind, {}).keys()
        gold_entity = g.get(kind, {}).keys()
        for i in pred_entity:
            if i in gold_entity:
                tp[idx] += 1
            else:
                fp[idx] += 1
                fp_entity.append(i)
        for i in gold_entity:
            if i not in pred_entity:
                fn[idx] += 1
                fn_entity.append(i)
                
#         # occur time
#         if len(gold_entity) > 0:
#             occur_times[idx][0] += len(gold_entity)
#             occur_times[idx][1] += 1

        # span length
        for span_list in g.get(kind, {}).values():
            for span in span_list:
                span_length[idx][0] += span[1] - span[0] + 1
                span_length[idx][1] += 1
                
    # bad cases
    if len(fp_entity) > 2 or len(fn_entity) > 2:
        badcases.append((text,g, p))
            
for i in range(10):
    precision[i] = tp[i] / (tp[i] + fp[i])
    recall[i] = tp[i] / (tp[i] + fn[i])
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

#     occur_times_avg[i] = occur_times[i][0] / occur_times[i][1]
    span_length_avg[i] = span_length[i][0] / span_length[i][1]

for i in range(10):
    print('Kind: {}'.format(KIND[i]))
    print('Precision: {}'.format(round(precision[i]*100, 1)))
    print('Recall: {}'.format(round(recall[i]*100, 1)))
    print('f1: {}'.format(round(f1[i]*100, 1)))
#     print('occur_times_avg: {}'.format(round(occur_times_avg[i], 1)))
    print('span_length_avg: {}'.format(round(span_length_avg[i], 1)))
    
with open('badcases.json', 'w') as f:
    for text, p, g in badcases:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
        f.write(json.dumps(g, ensure_ascii=False) + '\n')
        f.write('\n'*3)
