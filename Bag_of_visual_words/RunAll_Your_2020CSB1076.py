import helper
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


k = 31
trainHistogramSet,testHistogramSet = helper.CreateVisualDictionary(k)



countImagesTrain = helper.image_freq_for_words(trainHistogramSet)
countImagesTest = helper.image_freq_for_words(testHistogramSet)

print("Normalizing histograms of training set...")
trainHistogramSet =  helper.normalizeHistogramSet(trainHistogramSet,countImagesTrain)
print("Normalizing histograms of testing set...")
testHistogramSet =  helper.normalizeHistogramSet(testHistogramSet,countImagesTest)

print("Calculating Model Accuracy..")

predictions =  helper.getPredictions( helper.train_labels,  helper.test_labels,trainHistogramSet,testHistogramSet)



test_for = len( helper.test_labels)

y_true =  helper.test_labels[:test_for]
y_predict = predictions[:test_for]
print( classification_report(y_true, y_predict, target_names =  helper.class_name))
print("Overall accuracy => ")
print(accuracy_score(y_true, y_predict))


