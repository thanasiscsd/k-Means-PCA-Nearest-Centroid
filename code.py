from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def mahalanobis(x,centroid,inv_cov):
    distance = np.sqrt((x - centroid).T @ inv_cov @ (x - centroid)) #square root ((x-y).T * cov^-1 *(x-y))
    return distance

def PCA(train_set,numDim):

    #center data, by z=χ-μ
    mean = np.mean(train_set,axis=0) # axis=0 to take each column (pixels) (samples X pixels)
    z=train_set-mean

    #find covariance matrix
    cov=np.cov(z.T) #np.cov expects (features x samples), z is (samples x features), so take transpose

    #find eigenvectors (directions) and eigenvalues (variance in each direction)
    eigenvalues , eigenvectors=np.linalg.eigh(cov) #eigh returns in ascending order but for PCA descending is required
    eigenvalues=np.flip(eigenvalues) #flip for descending order
    eigenvectors=np.flip(eigenvectors,axis=1) #flip for descending order,axis=1 for each sample

    #lower dimensional space
    U=eigenvectors[:,:numDim] #find projection matrix
    zReduced=np.dot(z,U)
    return zReduced,U,mean

def kMeans(train_set,k):

        #first find k random indices
        kRandom=np.random.choice(train_set.shape[0],size=k,replace=False) #replace=false so we don't take the same index 2 times
        centroids=train_set[kRandom]  #use these indices as centroids
        for i in range(100):  # 100 operations of k-means
            #trainset is (samples x features), centroids is (k x features), incompatible for operations
            addedDimTrainSet=train_set[:,np.newaxis,:] #(samples x 1 x features)
            addedDimCentroids=centroids[np.newaxis,:,:] #(1 x k x features)
            #now their difference is a compatible operation (for the third axis (features))
            diff=addedDimTrainSet-addedDimCentroids
            distances=np.linalg.norm(diff,axis=2) #calculate distances using Euclidean distance,axis=2 for each sample
            labelsFound=np.argmin(distances,axis=1) #find the smallest distance,axis=1 for each sample
            #for new centroids calculation, find mean of each k
            newCentroids=np.array([np.mean(train_set[labelsFound==j],axis=0) for j in range(k)]) #take only labels equal to j, find mean (axis=0 for each feature, pixel)

            if np.allclose(centroids,newCentroids): #if centroids don't change, it converges
                break

            centroids=newCentroids
        return centroids,labelsFound

def nearestCentroid(train_set,test_set,labels):

    numClasses=len(np.unique(labels)) #number of unique classes

    mask=labels[:,np.newaxis]==np.arange(numClasses)  #boolean array mask[i,c], if i has label c, true
    # e.g. (labels=[0,1] ,numClasses=2, mask=[True, False], [False, True])

    centroids=mask.T @ train_set #(numClasses,num_sample) * (num_samples,num_features)

    counts=mask.sum(axis=0)[:,np.newaxis] #axis=0 for numClasses

    centroids=centroids/counts

    centroids=np.array(centroids)


    inv_covs=[]
    for i in range(numClasses): #for each class calculate covariance matrix
        current_train_set=train_set[labels==i] #take only labels of the current class
        cov=np.cov(current_train_set.T) + np.eye(train_set.shape[1]) * 1e-6 #find covariance matrix,add small value diagonally so matrix is inversive
        inv_cov=np.linalg.inv(cov) #inverse for distance calculation
        inv_covs.append(inv_cov)

    distances = []
    for x in test_set: #for each test sample
         distance=[mahalanobis(x,centroids[i],inv_covs[i]) for i in range(numClasses)] #calculate distance for each class for each sample in test set
         distances.append(distance)
    predictions=np.argmin(distances,axis=1) #append class with the smallest distance

    return predictions

#loading the dataset
dataset=load_dataset('ylecun/mnist')
dataset.set_format(type='numpy')

train_set=dataset['train']
train_set_images=np.array(train_set['image'])
train_set_images= train_set_images.reshape(-1,784).astype(np.float32) #reshape from 28 x 28 to 784 (flatten)
labels=np.array(train_set['label'])


test_set=dataset['test']
test_set_images = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32) #reshape from 28 x 28 to 784 (flatten)
test_labels=np.array(test_set['label'])

#for classes 4,6,7 select 1000 samples and apply PCA
digit4=train_set_images[labels==4][:1000]
digit6=train_set_images[labels==6][:1000]
digit7=train_set_images[labels==7][:1000]

allDigits=np.concatenate((digit4,digit6,digit7),axis=0)
zReduced,U,mean=PCA(allDigits,2)
zReduced4=zReduced[:1000]
zReduced6=zReduced[1000:2000]
zReduced7=zReduced[2000:]

centroids,labelsFound=kMeans(zReduced,3)

#plot
plt.figure(figsize=(8,6))
plt.scatter(zReduced4[:,0], zReduced4[:,1], label='4', alpha=0.6)
plt.scatter(zReduced6[:,0], zReduced6[:,1], label='6', alpha=0.6)
plt.scatter(zReduced7[:,0], zReduced7[:,1], label='7', alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of MNIST Digits')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(zReduced[:,0],zReduced[:,1],c=labelsFound,alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title("k-Means Clustering Results")
plt.legend()
plt.show()

#in order to return to original dimensions, multiply centroids with projection matrix (U)
centroidsOriginal=np.dot(centroids,U.T) #use transpose (3 x 2) * (2 x 784)
#then add back mean (removed in PCA)
centroidsFinal=centroidsOriginal+mean

plt.figure(figsize=(10, 3))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(centroidsFinal[i].reshape(28, 28), cmap='gray')
    plt.title(f"Centroid {i}")
    plt.axis('off')
plt.show()

#make of 10 classes 10 dimensional by applying PCA
zReducedTrain,U,mean=PCA(train_set_images,10)

#transform test set from 784 dimensions to 10 dimensions using (X-μ)*U
#we avoid using PCA for the test set, so U is the same for both set (test and train)
zReducedTest=np.dot((test_set_images-mean),U)
#zReducedTest,U,mean=PCA(test_set_images,10)

predictions=nearestCentroid(zReducedTrain,zReducedTest,labels)
accuracy=np.mean(predictions==test_labels)
print(f"Accuracy with 10 classes: {accuracy * 100:.2f}%")

#confusion matrix
cm = confusion_matrix(test_labels, predictions)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - 10 Classes")
plt.show()



zSubTrainList=[]
subLabelsList=[]
subToOriginalMap=[] #for ID of subclass(0-19) to ID (0-9) (original)
for i in range(10): #for each class

    current_train_set=zReducedTrain[labels==i]
    # break each class into 2, using k-means
    centroids,labelsFound=kMeans(current_train_set,2)
    new_sub_labels=labelsFound+(i*2) #(0,1 for class 0, 2,3 for class 1...)

    zSubTrainList.append(current_train_set)
    subLabelsList.append(new_sub_labels)

    #append to times because 2 labels transform to 1 (0,1 for class 0, 2,3 for class 1...)
    subToOriginalMap.append(i)
    subToOriginalMap.append(i)#

zSubTrain = np.concatenate(zSubTrainList, axis=0)
subLabels = np.concatenate(subLabelsList, axis=0)
#for 20 classes run again nearest centroid
predictions=nearestCentroid(zSubTrain,zReducedTest,subLabels)
#convert to original labels (from 0-19 to 0-9)
predictions_final = np.array([subToOriginalMap[p] for p in predictions])
accuracy=np.mean(predictions_final==test_labels)
print(f"Accuracy with 20 sub-classes: {accuracy * 100:.2f}%")

#confusion matrix
cm = confusion_matrix(test_labels, predictions_final)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - 20 Subclasses")
plt.show()

