def draft_split(n_num,thresholdVal,dataset):
    left,right = list(),list()
    for row in dataset:
        if row[n_num] < thresholdVal:
            left.append(row)
        else:
            right.append(row)
    return left,right

def GiniCostFunc(groups,classes):
    m = float(sum(len(group) for group in groups)) # m is the size of all training examples in the node
    #porportion = count(class_value) / count(rows)
    #gini_index = (1.0 - sum(proportion * proportion)) * (group_size/total_samples)
    gini = 0
    for group in groups:
        m_group = len(group)
        if m_group == 0:
            continue
        score = 0
        for val in classes:
            porportion = [row[-1] for row in group].count(val) / m_group
            score += porportion * porportion
        gini += (1.0 - score) * (m_group / m)
    return gini
def get_split(dataset):
    y = list(set(row[-1] for row in dataset))
    n_ind,ThresholdVal,gini_score, split_groups = 999,999,999,None
    for n in range(len(dataset[0]-1)):
        for row in dataset:
            groups = draft_split(n,row[n],dataset)
            gini = GiniCostFunc(groups,dataset)
            print('')
            if gini < ThresholdVal:
                n_ind,ThresholdVal,gini_score, split_groups = n, row[n],gini,groups
        return {'feature_index':n_ind,''}
    split,gini_score,
dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]

print(GiniCostFunc([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
