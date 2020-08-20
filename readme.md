# General info on the project files.

To run the project:

pull the project files from git.

go to capstone and then open the deployement folder, use the command "pip install requirements.txt" to install all the relevant libraries and dependencies.

download the "POSTMAN" API and set it up in your system.

Open the postman application and make the HTML request to test the model.
You can use the images from predict_img inside the dataset folder in capstone.

NOTE: The versions of libraries and the dependencies mentioned in the requirements document have to match or the test request may throw error.

Actual Projects files are in the Capstone folder.


Raw Images - contains all the images directly downloaded from the web, it has many irreleveant images and icons in each folder.

Original Images - contains all the correctly labeled images, but still no processing has been done on them yet.

Aligned Images - using create_dataset.py all the original images have been aligned and cropped and stored.