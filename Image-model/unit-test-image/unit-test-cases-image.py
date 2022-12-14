
from unittest_imageclassification import testImageExtension, testImageShape

# Example correct case
from PIL import Image
path = '1.jpg'
img_file = Image.open(path)
image = img_file.resize((224,224))
testImageExtension(path)
testImageShape(image)

# Example incorrect case - first unit test
from PIL import Image
path = 'incorrect_unittestcase_exampleimage.png'
img_file = Image.open(path)
image = img_file.resize((200,200))
testImageExtension(path)

# Example incorrect case - second unit test
testImageShape(image)
