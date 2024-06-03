from flask import Flask, render_template, request, Response, redirect, url_for
import base64
import cv2
from flask import jsonify
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

def save_images(username, image_files):
    data_folder = os.path.join('Database', 'Data', username)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    try:
        for idx, image_file in enumerate(image_files, start=1):
            image_path = os.path.join(data_folder, f'{username}_{idx:03}.jpg')
            image_file.save(image_path)
        
        return True, None  # Success, no error message
    except Exception as e:
        error_msg = f'Error saving images: {e}'
        return False, error_msg
app.config['UPLOAD_FOLDER'] = 'Database/Data'
def allowed_file(filename):
    allowed_extensions = {'jpeg', 'jpg', 'tiff', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/create_folder', methods=['POST'])
def create_folder():
    data = request.get_json()
    folder_name = data.get('folderName')
    if folder_name:
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
        try:
            os.makedirs(folder_path, exist_ok=True)
            return jsonify({'message': f'Folder "{folder_name}" created successfully!'})
        except OSError as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Folder name is required'}), 400
@app.route('/upload_images', methods=['POST'])
def upload_images():
    if 'username' not in request.form:
        return jsonify({'error': 'No username provided'}), 400

    if 'images' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    username = request.form.get('username')
    files = request.files.getlist('images')
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], username)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    for idx, file in enumerate(files, start=1):
        if file and allowed_file(file.filename):
            filename = f'{username}_{idx:03}.{file.filename.rsplit(".", 1)[1].lower()}'
            file_path = os.path.join(folder_path, filename)
            file.save(file_path)
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    return jsonify({'success': 'Images successfully uploaded'})
@app.route('/upload', methods=['POST'])
def upload():
    username = request.form['username']

    # Create a folder for the user if it doesn't exist
    user_folder = os.path.join('Database', 'Data', username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Save uploaded files to the user's folder with username_001, username_002, etc. naming convention
    for i in range(1, 4):  # Assuming you have 3 image uploads
        file_key = f'image{i}'
        if file_key in request.files:
            file = request.files[file_key]
            if file.filename != '':
                # Generate the filename with the desired format
                filename = f"{username}_{str(i).zfill(3)}{os.path.splitext(file.filename)[1]}"
                file.save(os.path.join(user_folder, filename))

    # Return a response with the uploaded file paths
    uploaded_images = [os.path.join(user_folder, f'{username}_{str(i).zfill(3)}') for i in range(1, 4)]
    return jsonify({'images': uploaded_images})

@app.route('/upload_photos', methods=['POST'])
def upload_photos():
    username = request.form.get('username')
    data_folder = os.path.join('Database', 'Data', username)
    
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
    

userImages=[];
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

# New route for adding images to the database
@app.route('/add_images_to_database', methods=['GET', 'POST'])
def add_images_to_database():
    return render_template('add_images_to_database.html')

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    global face_dict
    generate_database('Database/Data', FRmodel, augmentations=3, output_name='friends.npy')   
    face_dict = np.load('friends.npy', allow_pickle=True).item()
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
