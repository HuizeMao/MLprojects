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
    for n in range(len(dataset[0])-1):
        for row in dataset:
            groups = draft_split(n,row[n],dataset)
            gini = round(GiniCostFunc(groups,y),3)
            print("x{} < {} Gini Score = {} ".format(n+1,row[n],gini))
            if gini < gini_score:
                n_ind,ThresholdVal,gini_score, split_groups = n,row[n],gini,groups
    return {'feature_index':n_ind,'ThresholdVal':ThresholdVal,'groups':split_groups}
    split,gini_score

def get_terminal(group):
    y = [row[-1] for row in group]
    return max(set(y),key = y.count)

def split(node,max_depth,min_size,depth): #node is a dictionary contains group split info
    left,right = node['groups']
    del node['groups']
    #check if the next split only have one child
    if not left or not right:
        node['left'] = node['right'] = get_terminal(left + right)
        return
    #check if the recursion reach the max number 
    if depth >= max_depth:
        node['left'],node['right'] = get_terminal(left),get_terminal(right)
        return
    #process left child 
    if len(left) <= min_size:
        node['left'] = get_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'],max_depth,min_size,depth+1)
    #process right child 
    if len(right) <= min_size:
        node['right'] = get_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'],max_depth,min_size,depth+1)
    
def build_tree(dataset,max_depth,min_size):
    root = get_split(dataset)
    split(root,max_depth,min_size,1)
    print(root)
    return root  #root is a dictionary which has been modified during spliting 
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['feature_index']+1), node['ThresholdVal'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
def predict(x,modal):#single row x as input, modal is the dictionary created before
    if x[modal['feature_index']] < modal['ThresholdVal']:
        if isinstance(modal['left'],dict):
            return predict(x,modal['left'])
        else:
            return modal['left']
    else:
        if isinstance(modal['right'],dict):
            return predict[x,modal['right'],]
        else:
            return modal['right']

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
predict_modal = build_tree(dataset,1,1)
print_tree(predict_modal)
for row in dataset:
    prediction = predict(row,predict_modal)
    print(prediction)


