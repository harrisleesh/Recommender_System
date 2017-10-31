import random
import numpy as np
import pandas as pd
from collections import defaultdict

def loadData (fileName, d):#fileName
    userIdToUserIndex = {}
    basicUserIndex = 0
    itemIdToItemIndex = {}
    basicItemIndex = 0
    trainData = []
    testData = []
    trainFile = fileName
    testFile = "./ML100k/" + d + ".test"
    print(trainFile)
    print(testFile)

    for line in open(trainFile):
        userId, itemId, rating = line.strip().split(' ')
        trainData.append([userId, itemId, rating])

    for line in open(testFile):
        userId, itemId, rating, stamp = line.strip().split('\t')
        testData.append([userId, itemId, rating])

    random.seed(123456789)
    random.shuffle(trainData)
    random.shuffle(testData)

    trainSet = []
    for i in range(int(len(trainData))):
        userId, itemId, rating = trainData[i]
        if userId not in userIdToUserIndex:
            userIdToUserIndex[userId] = basicUserIndex
            basicUserIndex += 1
        if itemId not in itemIdToItemIndex:
            itemIdToItemIndex[itemId] = basicItemIndex
            basicItemIndex += 1
        userIndex = userIdToUserIndex[userId]
        itemIndex = itemIdToItemIndex[itemId]
        trainSet.append([userIndex, itemIndex, float(rating)])

    testSet = []
    for i in range(int(len(testData))):
        userId, itemId, rating = testData[i]
        if userId in userIdToUserIndex and itemId in itemIdToItemIndex:
            userIndex = userIdToUserIndex[userId]
            itemIndex = itemIdToItemIndex[itemId]
            testSet.append([userIndex, itemIndex, float(rating)])
            
    return len(userIdToUserIndex), len(itemIdToItemIndex), np.array(trainSet), np.array(testSet)


def processData(itemCount, trainSet, testSet):
    trainData = defaultdict(lambda: [0] * itemCount)    # trainData = [0,0,4,3,0,5,....]
    trainData_i = defaultdict(lambda: [0] * itemCount)  # trainData_i = [0,0,1,1,0,1,....]
    trainMask = defaultdict(lambda: [0] * itemCount)    # trainMask = [0,0,1,1,0,1,....]
    trainMask_i = defaultdict(lambda: [0] * itemCount)  # trainMask_i = [0,0,1,1,0,1,....]

    unratedItemsMask = defaultdict(lambda: [1] * itemCount) # unratedItemsMask = [1,1,0,0,1,0,....]
    
    positiveMask = defaultdict(lambda: [0] * itemCount) # positiveMask = [0,0,1,1,0,1,....]
    negativeMask = defaultdict(lambda: [1] * itemCount) # negativeMask = [1,1,0,0,1,0,....]
    
    #training data
    for t in trainSet:
        userId = int(t[0])
        itemId = int(t[1])
        rating = int(t[2])
        
        trainData[userId][itemId] = rating
        trainData_i[userId][itemId] = 1
        trainMask[userId][itemId] = 1
        trainMask_i[userId][itemId] = 1

        unratedItemsMask[userId][itemId] = 0

        positiveMask[userId][itemId] = 1
        negativeMask[userId][itemId] = 0

    #test data
    testData = defaultdict(lambda: [0] * itemCount)
    testData_i = defaultdict(lambda: [0] * itemCount)

    for t in testSet:
        userId = int(t[0])
        itemId = int(t[1])
        rating = int(t[2])
        if userId in trainData:
            if rating == 5:
                testData[userId][itemId] = 1
            testData_i[userId][itemId] = 1

    # unrated items
    unrated_items = []
    numOfRatings = []   # numOfRatings = [23, 52, 27, 64, ....]

    for i in range(len(trainData)):
        unrated_items.append(np.nonzero(unratedItemsMask[i])[0])
        numOfRatings.append(np.count_nonzero(trainData[i]))
    
    # ground truth
    GroundTruth = []
    GroundTruth_i = []
    
    for i in list(testData.keys()):
        tmp = []
        for j in range(len(testData[i])):
            if testData[i][j] == 1:
                tmp.append(j)
        GroundTruth.append(tmp)

    for i in list(testData_i.keys()):
        tmp = []
        for j in range(len(testData_i[i])):
            if testData_i[i][j] == 1:
                tmp.append(j)
        GroundTruth_i.append(tmp)
        
    return testData, testData_i, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask, positiveMask, negativeMask, numOfRatings, unrated_items, GroundTruth, GroundTruth_i

def prepareTrainAndTest (trainData, unratedItemsMask, testData):
    allTrainData = []
    allTestData = []
    unratedTrainMask = []
    ulist = list(trainData.keys())

    for userId in testData:
        allTrainData.append(trainData[userId])
        allTestData.append(testData[userId])
        unratedTrainMask.append(unratedItemsMask[userId])

    allTrainData = np.array(allTrainData)
    allTestData = np.array(allTestData)
    unratedTrainMask = np.array(unratedTrainMask)

    return ulist, allTrainData, allTestData, unratedTrainMask
    
    
# To be used.....
# 1.
def remove_users_under_k(d, k=12):

    # read data
    data_path = "C:/Users/qweqa/Desktop/AutoRecProject/dataset/"+d+"-all"
    # tag_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
    tag_headers = ['user_id', 'movie_id', 'rating']

    ratings = pd.read_table(data_path, sep=' ', header=None, names=tag_headers, index_col=False)
    #ratings.sort_values(by=['user_id', 'movie_id'], ascending=[True, True], inplace=True)
    #ratings.reset_index(drop=True, inplace=True)
    #ratings.to_csv("C:/Users/qweqa/Desktop/AutoRecProject/dataset/"+d+"--sort", header=None, index=None, sep=' ', mode='w')
    print(ratings)
    # count ratings
    tmpUser = defaultdict(set)


    for index, row in ratings.iterrows():
        tmpUser[row['user_id']].add(row['movie_id'])


    # remove users having under k ratings
    del_users = pd.DataFrame()
    del_index = []
    del_key = []
    #print(tmpUser.items())
    for key, value in tmpUser.items():
        if len(value) < k:
            del_key.append(key)



    del_index.append([index for index, row in ratings.iterrows() if row['user_id'] in del_key])
    del_index= del_index[0]


    del_users = del_users.append(ratings.iloc[del_index], ignore_index=True)
    #print(del_users)

    ratings.drop(ratings.index[del_index], inplace=True)

    #ratings = ratings.query('@del_users not in user_id')
    ratings.reset_index(drop=True, inplace=True)

    print(len(del_users), 'users are deleted...')

    return ratings, del_users, tmpUser

    # save data
    #new_data_path = "dataset/"+d+"-all-"+str(k)
    #ratings.to_csv(new_data_path, header=None, index=None, sep=' ', mode='w')

# Generate Train and Test based on the order of Time Stamp
# Make the last k items per user as test set and the rest are the training set
# 2.
def leave_k_out(d, k=1):
    print(d)
    print("Separating.....")
    data_path = "dataset/"+d+"-all-10"
    test_path = "dataset/"+d+"--test"
    train_path = "dataset/"+d+"--train"
    
    tag_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
    

def load_bprdata(data_path):
    '''
    As for bpr experiment, all ratings are removed.
    '''
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f.readlines():
            u, i, _ = line.split(" ")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)

    #print("max_u_id:", max_u_id)
    #print("max_i_id:", max_i_id)

    return max_u_id, max_i_id, user_ratings
def load_bprdata_2(traindata):
    '''
    As for bpr experiment, all ratings are removed.
    '''
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    for u in range(len(traindata)):
        for i in range(len(traindata[u])):
            if traindata[u][i] is not 0:
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)

    print("max_u_id:", max_u_id+1)
    print("max_i_id:", max_i_id+1)

    return max_u_id+1, max_i_id+1, user_ratings



# Generate Train and Test Randomly:
# 3.
def Separate_training_and_test(d, k=20):
    print(d)
    print("Separating.....")
    data_path = "C:/Users/qweqa/Desktop/AutoRecProject/dataset/"+d+"-all"
    test_path = "C:/Users/qweqa/Desktop/AutoRecProject/dataset/"+d+"--test"
    train_path = "C:/Users/qweqa/Desktop/AutoRecProject/dataset/"+d+"--train"
    
    #tag_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
    tag_headers = ['user_id', 'movie_id', 'rating']

    ratings, del_users, tmpUser = remove_users_under_k(d,k*2)

    testui = {}
    testindices = []
    for key, value in tmpUser.items():
        if len(value) > k*2:
            tmp = []
            while(len(tmp) is not k):                  #k개 test set
                n = random.randrange(0, len(value))     #random
                if list(value)[n] not in tmp:
                    tmp.append(list(value)[n])
            testui[key]=tmp

    testindices.append([index for index, row in ratings.iterrows() if row['user_id'] in list(testui.keys()) and row['movie_id'] in testui[row['user_id']]])
    testindices=testindices[0]

    test_ratings = pd.DataFrame()
    # generate training and test dataframe
    test_ratings = test_ratings.append(ratings.iloc[testindices], ignore_index=True)
    #print(test_ratings)
    #print(ratings)
    ratings.drop(ratings.index[testindices], inplace=True)
    ratings.reset_index(drop=True, inplace=True)


    ratings = ratings.append(del_users)
    # print(ratings)
    # save training file
    ratings.sort_values(by=['user_id', 'movie_id'], ascending=[True, True], inplace=True)
    ratings.reset_index(drop=True, inplace=True)
    ratings.to_csv(train_path, header=None, index=None, sep=' ', mode='w')

    # save test file
    test_ratings.sort_values(by=['user_id','movie_id'], ascending=[True, True], inplace=True)
    test_ratings.reset_index(drop=True, inplace=True)
    test_ratings.to_csv(test_path, header=None, index=None, sep=' ', mode='w')

    print("Data Separation done....")


"""
    total = len(ratings.index)
    us_total = ratings.user_id.unique()
    it_total = ratings.movie_id.unique()

    
    test_index = random.sample(range(1, total), int(total*k*0.1))


            
    #us_train = ratings.user_id.unique()
    #it_train = ratings.movie_id.unique()
    #print(len(us_total), len(us_train), len(it_train))
    #us_test = test_ratings.user_id.unique()
    #it_test = test_ratings.movie_id.unique()
    #print(len(it_total), len(us_test), len(it_test))
            
    # save training file
    ratings.sort_values( by=['user_id', 'movie_id'], ascending=[True, True], inplace=True )
    ratings.reset_index(drop=True, inplace=True )
    ratings.to_csv(train_path, header=None, index=None, sep=' ', mode='w')
    
    # save test file
    test_ratings.sort_values( by=['movie_id', 'user_id'], ascending=[True, True], inplace=True )
    test_ratings.reset_index(drop=True, inplace=True )
    test_ratings.to_csv(test_path, header=None, index=None, sep=' ', mode='w')
    
    print("Data Separation done....")
    
    return len(us_total)+1, len(it_total)+1
    """

#remove_users_under_k("ML1M")
#Separate_training_and_test("Ciao",6)


