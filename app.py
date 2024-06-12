from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import os
from PIL import Image

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO(r'static/model/weights/best.pt')
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG','.jpeg', '.JPEG', '.png', '.PNG', '.heif', '.HEIF', '.heic', '.HEIC', '.webp', '.WEBP']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/imgpred', methods=['GET', 'POST'])
def readURL():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename != '':

            image_ext = os.path.splitext(file.filename)[1]

            if image_ext in app.config['UPLOAD_EXTENSIONS']:
                # Save the uploaded image to a temporary location
                image_path = "static/uploaded_image.jpg"
                file.save(image_path)

                # Run inference on the uploaded image
                results = model(image_path)  # results list

                # Visualize the results
                for i, r in enumerate(results):
                    # Plot results image
                    im_bgr = r.plot()  # BGR-order numpy array
                    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

                    # Save the result image
                    result_image_path = "static/result_image.jpg"
                    im_rgb.save(result_image_path)
                
                os.remove(image_path)

            else:
                result_image_path="static/error_image.jpeg"

            # Render the HTML template with the result image path
            return render_template('index.html', image_path=result_image_path)

    # If no file is uploaded or GET request, render the form
    return render_template('index.html', image_path=None)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=False, port=5000, host='0.0.0.0', threaded = True)
