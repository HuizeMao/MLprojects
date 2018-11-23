def draft_split(n_num,thresholdVal,dataset): # split training examples
    left,right = list(),list()
    for row in dataset: # iterate through every single training example
        if row[n_num] < thresholdVal:  #if the feature index of one example is smaller put it in the left
            left.append(row)
        else:
            right.append(row)
    return left,right

def GiniCostFunc(groups,classes): #calculates how mixed the splitting groups are
    m = float(sum(len(group) for group in groups)) # m is the total training examples in the groups
    gini = 0
    for group in groups: # iterate through each group
        m_group = len(group)
        if m_group == 0: # check if there is an empty group
            continue
        score = 0
        for val in classes: # iterate through total y values
            porportion = [row[-1] for row in group].count(val) / m_group #calculate proportion
            score += porportion * porportion
        gini += (1.0 - score) * (m_group / m) # calculates a score of how mixed the groups are
    return gini
def get_split(dataset):
    y = list(set(row[-1] for row in dataset))
    n_ind,ThresholdVal,gini_score, split_groups = 999,999,999,None
    for n in range(len(dataset[0])-1): #iterate through all feature index
        for row in dataset:
            groups = draft_split(n,row[n],dataset) # create randomly split groups using a features of a group
            gini = round(GiniCostFunc(groups,y),3) #calculates gini score using the feature above
            print("x{} < {} Gini Score = {} ".format(n+1,row[n],gini))
            if gini < gini_score: # if the gini score is lower than the previous gini score, store it as the ideal split
                n_ind,ThresholdVal,gini_score, split_groups = n,row[n],gini,groups
    return {'feature_index':n_ind,'ThresholdVal':ThresholdVal,'groups':split_groups} #return a dict with all info

def get_terminal(group): #get the final value in the terminal node
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

def build_tree(dataset,max_depth,min_size): #builds a dicision tree
    root = get_split(dataset) #create the first node
    split(root,max_depth,min_size,1) #add the rest nodes to the dict root
    return root
def print_tree(node, depth=0):#prints the dicision tree
	if isinstance(node, dict):
		print('{}[X{} < {}]' .format((depth*' ', (node['feature_index']+1), node['ThresholdVal'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('{}[{}]' .format((depth*' ', node)))
def predict(x,modal):#single row x as input, modal is the dictionary created before
    if x[modal['feature_index']] < modal['ThresholdVal']: # check if feature of x is smaller than thresholdVal
        if isinstance(modal['left'],dict): #check if the testing example could go into left node
            return predict(x,modal['left']) # goes to the left node
        else: # if not return the value of the current node prediction
            return modal['left']
    else: #if it is greater or equal to the thresholdVal
        if isinstance(modal['right'],dict): # check if there is a right node
            return predict[x,modal['right'],] # goes to the right node
        else: # if not then return the value of the current right node
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
