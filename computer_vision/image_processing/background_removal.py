from rembg import remove
from PIL import Image

img = Image.open('adam.jpg')
result = remove(img)
result.save('result.png')


