
# EE-666: Assignment 3 - Panaroma Stitching

## Given a set of images, Find Homography matrices and stitch images to make a Panaroma.
This assignment is intended to provide practical experience with GitHub, along with the concepts discussed in class.


### Pre-requisites
 - Github Account.  Don't have one still? Create one. 
 - Install git.
 - Install Python locally in your system. Recommended : open-source distribution of the Python by Anaconda.
 - Install opencv ```pip install opencv-python```
 - Clone the repository @ `https://github.com/shash29-dev/ES666-Assignment3.git`

```
# cd /path/to/folder/where/to/clone
git clone https://github.com/shash29-dev/ES666-Assignment3.git
```

## Check if boilerplate code works

```
cd ES666-Assignment3
python main.py
```

Running `main.py` should create `./results` folder and exit without Error. 


## Inside the repo
 - `Images : ` This folder contains images to be stitched to create panaroma.
 - `src` : Your Code goes here, Inside a folder. Check a Dummy Submissions by `JohnDoe`.
    - `JohnDoe/stitcher.py :` contains class `PanaromaStitcher`. Go through the class method named `make_panaroma_for_images_in` which should return two outputs: Final stitched Image and Homography matrices.

    Note:  You can organise your code however you want but yout folder must have `stitcher.py` file containing class `PanaromaStitcher` with atleast one method named `make_panaroma_for_images_in` returning Final stitched Image and Homography matrices.

    - `main.py :` Main file to run all Implementations inside `src`. Edit `Line-11` with folder of images to be stitched.


## Create Your Stitcher

 - Check output related to`DartVader's` submission. The `try` block in `main.py` should not fail. The returned outputs from `stitcher.py` should be in required order: stitched_image and a list of matrices. The `stitched_image` will get saved in `./results` folder.
 - Don't delete/make any changes in dummy submssions included in Repo.
 - Create a folder inside src with `stitcher.py` and complete the class method `make_panaroma_for_images_in` as discussed above.
 - Check `./results` for generated results.

