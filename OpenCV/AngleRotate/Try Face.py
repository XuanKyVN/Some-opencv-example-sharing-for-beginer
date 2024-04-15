from faceorienter import FaceOrienter

fo = FaceOrienter('path/to/image.jpg')
fo.predict_orientation()  # Detect the image's orientation (returns either `down`, `up`, `left`, or `right`)
fo.fix_orientation('path/to/new/image.jpg')  # Corrects the orientation and writes the new image to the specified path