from flask import Flask, render_template, request, Response, redirect, url_for
import base64
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model, load_model, model_from_json
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate
from keras.layers import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from facerecog_utils import *
from inception_blocks import *
from database_utils import *
from align import AlignDlib

app = Flask(__name__)

FRmodel = faceRecoModel(input_shape=(3,96,96))
FRmodel.load_weights('Libraries/nn4.small2.v1.h5')
face_dict = np.load('friends.npy', allow_pickle=True).item()


def gen_frames():  
    cap = cv2.VideoCapture(0)
    # time.sleep(15)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)##
        img_show = face_recognition(frame, FRmodel, database=face_dict, plot=False)
        frame = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_video(path):   
    cap = cv2.VideoCapture(path)
    frame_skip_interval = 25
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)##
            img_show = face_recognition(frame, FRmodel, database=face_dict, plot=False)
            frame = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('base.html')

@app.route('/picture_recognition', methods=['GET', 'POST'])
def picture_recognition():
    if request.method == 'POST':
        # Handle picture recognition logic here
        if 'image' in request.files:
            image_file = request.files['image']
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Apply face recognition
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_show = face_recognition(frame, FRmodel, database=face_dict, plot=False)
            img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
             
            _, img_encoded = cv2.imencode('.jpg', img_show)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            return render_template('picture_display.html', img_data=img_base64)  # Pass image data to template
    return render_template('picture.html')


@app.route('/live_stream_recognition')
def live_stream_recognition():
    # Open webcam and return live stream
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close_camera', methods=['POST'])
def close_camera():
    cap.release()  # Release camera object
    cv2.destroyAllWindows()  # Destroy OpenCV window
    return 'Camera closed and OpenCV window destroyed'


@app.route('/video_recognition', methods=['GET', 'POST'])
def video_recognition():
    if request.method == 'POST':
        # Handle picture recognition logic here
        if 'video' in request.files:
            video_file = request.files['video']
            video_path = "uploads/video.mp4"  
            # Save the uploaded video to a file
            video_file.save(video_path)
           
            return render_template('video2.html')  # Pass image data to template
    return render_template('video.html')

@app.route('/video_2_feed')
def video_2_feed():
    video_path = "uploads/video.mp4"
    return Response(gen_frames_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/add_to_database')
# def add_to_database():
#     # Handle add to database logic here 
#     return "Add to Database"

userImages=[];
@app.route('/upload_photos', methods=['POST'])
def upload_photos():
    username = request.form.get('username')
    data_folder = os.path.join('Database_01', 'Data', username)
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    try:
        for i in range(1, 7):
            image_file = request.files.get(f'image_{i}')
            if image_file:
                image_path = os.path.join(data_folder, f'{username}_{i:03}.jpg')
                image_file.save(image_path)
        
        return jsonify(success=True)
    except Exception as e:
        print(f'Error saving images: {e}')
        return jsonify(success=False, error=str(e))
@app.route('/add_to_database', methods=['POST','GET'])
def add_to_database():
    if request.method == 'POST':
        # Get the username from the form data
        username = request.form['username']

        # Create a folder for the user in the 'Database' folder
        user_folder = os.path.join('Database', username)
        os.makedirs(user_folder, exist_ok=True)

        # Save the images in the user's folder
        for idx, img_data in enumerate(userImages):
            img_path = os.path.join(user_folder, f'{username}_{idx + 1}.jpg')
            with open(img_path, 'wb') as img_file:
                img_file.write(base64.b64decode(img_data.split(',')[1]))

        # Clear the userImages list after saving the images
        userImages.clear()

        # Return a success message or redirect to another page
        return redirect(url_for('index'))

    # Handle GET request or other cases
    return render_template('add_to_database.html')

if __name__ == '__main__':
    app.run(debug=True)
