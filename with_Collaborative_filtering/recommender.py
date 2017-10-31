import math
import sys


# find neighbour set by user
def neighbour_set(user):
    neighbor = []

    #evalueate cosine similarity from neighbor
    for i in range(1,upperbound):
        neig_cos = []
        if i != user:
            neig_cos.append(cossim(user,i))
            neig_cos.append(i)
        neighbor.append(neig_cos)

    neighbor.sort()
    neighbor.reverse()
    real_neig = []

    #select only cosine similarity value is over 0.95
    for i in neighbor:
        if len(i)>0 and i[0]>0.95:
            real_neig.append(i[1])
        else:
            break
    return real_neig


# this function evaluate cosine simmilarity value
def cossim(user1,user2):
    both = []
    simm = 0
    sqrt_user1 = 0
    sqrt_user2 = 0
    for i in watched[user1]:
        if i in watched[user2]:
            both.append(int(i))
    if len(both)==0:
        return 0
    else:
        for item in both:
            simm += int(rate_dict[user1][item])*int(rate_dict[user2][item])
            sqrt_user1 += math.pow(int(rate_dict[user1][item]), 2)
            sqrt_user2 += math.pow(int(rate_dict[user2][item]), 2)
        sqrt_user1 = math.sqrt(sqrt_user1)
        sqrt_user2 = math.sqrt(sqrt_user2)
    return simm/(sqrt_user1*sqrt_user2)


# this function evaluate each test item's rating from neighbor's ratings
def predict_rate(item_id, neighbor, data):
    rate = 0
    num = 0
    for i in neighbor:
        if item_id in data[i].keys():
            rate += int(data[i][item_id])
            num +=1
    if num!=0:
        return round(rate/num,4)
    else:
        return 3


#input and output file format
inputfilename=sys.argv[1]
training = open(sys.argv[1],'r')
test = open(sys.argv[2],'r')
outputfilename = 'u'+inputfilename[1]+'.base_prediction.txt'
predict = open(outputfilename,'w')

readline_test = test.readlines()
readline_train = training.readlines()

#user_rated_itemset is temp array for find rating matrix
user_rated_itemset = []
#watched array is a array which contain watched items from each user
watched=[]

for i in readline_train:
    rating = i.split()
    rate = []
    rate.append(rating[0])
    rate.append(rating[1])
    user_rated_itemset.append(rate)

#upperbound is the highest number of user id
upperbound = int(user_rated_itemset[-1][0])

for i in range(0,upperbound+1):
    watched.append([])

for i in user_rated_itemset:
    watched[int(i[0])].append(i[1])


#rate_dict is a dictionary which contains all of training data
#each user has item and ratings
rate_dict = {}
for i in range(0,upperbound+1):
    rate_dict[i]={}

for i in readline_train:
    rating = i.split()
    rate= {}
    rate[int(rating[1])]=rating[2]
    rate_dict[int(rating[0])].update(rate)

#neighbor array show the neighbor of each user
neighbor = []
for i in range(0,upperbound+1):
    neighbor.append([])

for i in range(1,upperbound+1):
    neighbor[i]=neighbour_set(i)


#predictset contain which user item pair should be predicted
predictset = []
test_set = []
for i in range(0,upperbound+1):
    predictset.append([])


for i in readline_test:
    rating = i.split()
    rate = []
    rate.append(rating[0])
    rate.append(rating[1])
    test_set.append(rate)

for i in test_set:
    predictset[int(i[0])].append(int(i[1]))


#write predict ratings to output file
for i in range(1,upperbound+1):
    if len(predictset[i])!= 0:
        for j in predictset[i]:
            id = str(i)
            predict.write(id)
            predict.write('\t')
            predict.write(str(j))
            predict.write('\t')
            if(len(neighbor[i])==0):
                predict.write('3')
            else:
                predict.write(str(predict_rate(j,neighbor[i],rate_dict)))
            predict.write('\n')



training.close()
test.close()
predict.close()

