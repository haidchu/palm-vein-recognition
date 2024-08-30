# Dorsal hand vein recognition

Last modified: 28.08.24

## Image acquisition

The method of capturing the image of a person's dorsal hand vein is described below:
* A NIR light source is used to illuminate the person's dorsal hand
* An IR camera is used to capture the illuminated dorsal hand

Since oxygen-poor blood can absorb NIR light in the range 760nm-970nm, the result of the above method  
will contain dark regions of veins under the skin. These regions are unique to a person's dorsal hand,  
therefore can be used to identify the person.

Currently, the images are acquired from a public database. In this database, each dorsal hand images  
will be taken at two different time, with 4 images (db1) taken together and 3 images(db2) taken 2 weeks  
after that.


## Image preprocessing
In the source code, two different image preprocessing methods are used. 
* `preprocess_dorsal_v1.py`creates a well-highlighted vein system of the dorsal hand vein.  
However, this method takes too long to process a single image.
* `preprocess_dorsal_v2.py` creates a worse results comparing to the method above, but this  
method is faster.


## Feature extraction
The main method applied is using Resnet50 with an additional layer of Arcface loss.  
Different versions seen in the folder works on different datasets produced by the two  
preprocessing methods above


## Evaluation
After extracting features, the feature vector of each images is run through a KNN. The result is positive,  
with 95-100% of the test set correctly classified to their true labels. This proves that the images are  
well-presented by the feature vectors.


## Further work
Change to palm vein in the future: after an effective method of taking the palm vein picture is used,  
one could change the database to a palm vein one. In that case, ROI extraction method can be found in  
`preprocess_palm_v1.py`

## Dataset
the dataset used in the training process is from https://github.com/wilchesf/dorsalhandveins/tree/main

## Code for rearranging image data
This code will process each image and write them to the folder with its id
```
imdir = './dorsal_git/'
tardir = './dorsal_db_v2/'

import shutil

for impath in os.listdir(imdir):
    if img_path.find('png') == -1: continue
    k = 7
    name = int(impath[k:k+3]) + (138 if impath.find('_R') != -1 else 0)
    path = tardir + str(name) + '/'
    os.makedirs(path, exist_ok=True)

    src = imdir + impath
    dst = path + impath

    img = cv2.imread(src)
    processed_img = np.bitwise_not(preprocessing(img))
    cv2.imwrite(dst, processed_img)
    # shutil.copy(imdir + impath, path + impath)
```
