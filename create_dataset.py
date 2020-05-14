import os
import dlib
import cv2
import imutils
from uuid import uuid4
from tqdm import tqdm

ORIGINAL_DATASET_PATH='original_images'
CROPPED_ALIGNED_FACES_DATASET_PATH='aligned_images'
SHAPE_DETECTOR_MODEL_PATH='shape_predictor_68_face_landmarks.dat'
MAX_FACES_TO_COMPENSATE=4
IMAGE_RESIZE_HEIGHT=650
FACE_DETECTOR=dlib.get_frontal_face_detector()
SHAPE_DETECTOR=dlib.shape_predictor(SHAPE_DETECTOR_MODEL_PATH)
ALLOWED_FILE_EXTENSIONS=['jpg','jpeg','png']

if not os.path.exists(CROPPED_ALIGNED_FACES_DATASET_PATH):

    os.mkdir(CROPPED_ALIGNED_FACES_DATASET_PATH)

if __name__=='__main__':

    for folder_name in tqdm(os.listdir(ORIGINAL_DATASET_PATH),desc='First Loop'):

        original_folder_path=os.path.join(ORIGINAL_DATASET_PATH,folder_name)

        if os.path.isdir(original_folder_path):

            new_folder_path=os.path.join(CROPPED_ALIGNED_FACES_DATASET_PATH,folder_name)

            if not os.path.exists(new_folder_path):

                os.mkdir(new_folder_path)

            for file_name in os.listdir(original_folder_path):

                if file_name.lower().split('.')[-1].strip() in ALLOWED_FILE_EXTENSIONS:

                    original_image_path=os.path.join(original_folder_path,file_name)
                    image_data_bgr=cv2.imread(original_image_path,cv2.IMREAD_COLOR)
                    image_data_bgr=imutils.resize(image_data_bgr,height=IMAGE_RESIZE_HEIGHT)
                    image_data_rgb=cv2.cvtColor(image_data_bgr,cv2.COLOR_BGR2RGB)
                    face_detections=FACE_DETECTOR(image_data_rgb,1)

                    if not 0<=len(face_detections)<=MAX_FACES_TO_COMPENSATE:

                        continue

                    for detected_face in face_detections:

                        faces=dlib.full_object_detections()
                        faces.append(SHAPE_DETECTOR(image_data_rgb,detected_face))
                        detected_face_image_bgr=cv2.cvtColor(dlib.get_face_chips(image_data_rgb,faces,size=256)[0],
                                                             cv2.COLOR_RGB2BGR)
                        cropped_image_path=os.path.join(new_folder_path,str(uuid4())+'.jpg')
                        cv2.imwrite(cropped_image_path,detected_face_image_bgr)
                        





