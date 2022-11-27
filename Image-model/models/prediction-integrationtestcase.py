from prediction-integrationtestcase import image_reshaping, predict_birdImage

# Example - AFRICAN CROWNED CRANE - class 3
image_path = '1.jpg'
img, img_file = image_reshaping(image_path)
predict_birdImage(img, img_file)
