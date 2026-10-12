import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters: 
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            # the 2D array of each image is flattened because in machine learning
            # a classifier requires that each sample is a 1d array
            # for example a 10*20 array will become 1*400
            # in machine learning terms that's 400 features with each pixel

            rows = np.any(binary_image, axis=1)
            cols = np.any(binary_image, axis=0)
            if not rows.any():
                continue

            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            cropped = binary_image[r0:r1+1, c0:c1+1]

            resized = resize(cropped.astype(float), (20, 20)) > 0.5
            image_data.append(resized.reshape(-1))
            target_data.append(each_letter)

            # flat_bin_image = binary_image.reshape(-1)
            # image_data.append(flat_bin_image)
            # target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold will determine the type of validation 
    # for example the num_of_fold = 4 then we are performing a 4-fold cross validation
    # this will divide the dataset into 4 and use 1/4 of it for testing
    # and then there is 3/4 remaining for the training
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv = num_of_fold)
    print("Cross Validation Result For", str(num_of_fold), " -fold")

    print(accuracy_result * 100)

current_dir = os.path.dirname(os.path.realpath(__file__))

training_dataset_dir = os.path.join(current_dir, 'train')

image_data, target_data = read_training_data(training_dataset_dir)

# the kernel can be 'linear', 'poly', or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
svc_model = SVC(kernel='linear', probability=False)

cross_validation(svc_model, 4, image_data, target_data)

#  training the model with all the input data
svc_model.fit(image_data, target_data)

# using joblib module will persist the model
# into files. this will save time because the next
# prediction won't require training another model
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory + '/svc.pkl')