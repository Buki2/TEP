from main import tep_method

image_path = 'images/demo.jpg'
text = 'apple on the right'
box = tep_method(image_path, text)
print(box)