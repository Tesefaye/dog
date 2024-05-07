import asyncio
from flask import Flask, render_template, request, jsonify
from inference import predict_breed

app = Flask(__name__)

async def run_async(predict_func, *args, **kwargs):
    result = await predict_func(*args, **kwargs)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_file' in request.files and request.files['image_file'].filename:
        print("Received image file: ", request.files['image_file'].filename)
        image_file = request.files['image_file']
        try:
            # Run predict_breed asynchronously
            result = asyncio.run(run_async(predict_breed, image_file))
            # Perform breed prediction on the uploaded image
            predicted_breed, _ = result
            print("Prediction completed.")
            return jsonify(predicted_breed=predicted_breed)
        except Exception as e:
            return jsonify(error=str(e))
    elif 'image_url' in request.form and request.form['image_url']:
        image_url = request.form['image_url']
        print("Received image URL: ", image_url)
        try:
            # Run predict_breed asynchronously
            result = asyncio.run(run_async(predict_breed, image_url=image_url))
            # Perform breed prediction on the image URL
            predicted_breed, _ = result
            print("Prediction completed.")
            return jsonify(predicted_breed=predicted_breed)
        except Exception as e:
            return jsonify(error=str(e))
    else:
        return jsonify(error='No image file or URL provided.')

if __name__ == '__main__':
    app.run(debug=True)

