import tensorflow as tf

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg"])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
tensor = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
	print(sess.run(tensor))
	print(sess.run(tensor))
	
dataset = dataset.map(_parse_function)

