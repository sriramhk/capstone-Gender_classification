import os
import dlib
from flask import Flask,request,jsonify
from werkzeug.utils import secure_filename
import logging
import numpy as np
import cv2
import time
from flask_cors import CORS
#from keras.models import load_model
#import keras
import tensorflow as tf

log=logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)
handler=logging.FileHandler('model_test_api.log')
handler.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

UPLOAD_FOLDER='uploads'

if not os.path.exists(UPLOAD_FOLDER):

    os.mkdir(UPLOAD_FOLDER)

app=Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

FACE_DETECTOR=dlib.get_frontal_face_detector()
SHAPE_PREDICTOR=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
GENDER_DETECTION_MODEL=tf.keras.models.load_model('trained_model.hdf5',{'tf':tf})
GENDER_DETECTION_THRESHOLD=0.50


def return_response(data,message,error):

    try:

        _response_json={
            'data':None if error else data,
            'message':message,
            'status':not error
        }

        return jsonify(_response_json)

    except:

        log.error('Traceback of return response of model_test_api.py',exc_info=True)

@app.route('/recognition/api/v1.0/gender_detection',methods=['POST'])
def gender_detection_main():

    try:
        print(request.form)
        log.info('Request files: {}'.request.files)
        log.info('Request files: {}'.format(request.files))

        if request.method=='POST':

            if 'image_file' not in request.files:

                return return_response(data=None,message='key image_file not found!!',error=True)

            query_file=request.files['image_file']

            if query_file.filename =='':

                return return_response(data=None,message='No file selected!!',error=True)

            if query_file:

                session_time=time.time()
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],str(session_time)))
                query_img_filename=secure_filename('{}_{}'.format(session_time,query_file.filename))
                query_img_full_path=os.path.join(app.config['UPLOAD_FOLDER'],str(session_time),query_img_filename)
                query_file.save(query_img_full_path)

            else:

                return return_response(data=None,message='No file selected!!!',error=True)

            query_image_data_bgr=cv2.imread(query_img_full_path,cv2.IMREAD_COLOR)
            query_image_data_rgb=cv2.cvtColor(query_image_data_bgr,cv2.COLOR_BGR2RGB)
            face_detections=FACE_DETECTOR(query_image_data_rgb,1)

            if len(face_detections)==0:

                return return_response(data=None,message='No face present!!',error=True)

            elif len(face_detections)>1:

                return return_response(data=None,message='Multiple faces detected!!',error=True)

            else:

                detected_face=face_detections[0]

            faces=dlib.full_object_detections()
            faces.append(SHAPE_PREDICTOR(query_image_data_rgb,detected_face))
            detected_face_image_aligned_rgb=dlib.get_face_chips(query_image_data_rgb,faces,size=64)[0]

            # cv2.imshow('Aligned Face Image',detected_face_image_aligned_rgb[:,:,::-1])
            # cv2.waitKey(0)

            detected_face_image_aligned_rgb=detected_face_image_aligned_rgb.astype(dtype=np.float64)
            detected_face_image_aligned_rgb/=255.0
            detected_face_image_aligned_rgb=detected_face_image_aligned_rgb[None,:,:,:]
            person_gender=GENDER_DETECTION_MODEL.predict(detected_face_image_aligned_rgb)[0]

            log.info('Gender model predicted values: {}'.format(person_gender))

            if person_gender[0]<GENDER_DETECTION_THRESHOLD:

                return return_response(data={'gender':'female'},message='Gender detected successfully!!',error=False)

            else:

                return return_response(data={'gender':'male'},message='Gender detected successfully!!',error=False)


        else:

            return return_response(data=None,message='Invalid Request Type!!!',error=True)

    except:

        log.error('Traceback of gender_detection_main of model_test_api.py',exc_info=True)

        return return_response(data=None,message='Unknown error occurred!!!',error=True)

if __name__=='__main__':

    app.run(host='0.0.0.0',port=9004,threaded=True)

