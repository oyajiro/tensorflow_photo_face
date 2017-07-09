import tensorflow as tf
import sys
import shutil
import os


readdir = "./photos/output/"
outputdir = './photos/face_detect/'
graphname = 'face_graph.pb'
labelsname = 'face_labels.txt'
scorelimit = 0.9

_, _, filenames = next(os.walk(readdir), (None, None, []))

# Unpersists graph from file
with tf.gfile.FastGFile("/tensor/" + graphname, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("/tensor/" + labelsname)]
filesInPercent = len(filenames) / 100
percentcounter = 0
counter = 0
with tf.Session() as sess:
    for filename in filenames:
        if counter % filesInPercent == 0:
            percentcounter += 1
            print str(percentcounter) + "%\n"
        counter += 1
        # Read in the image_data
        image_data = tf.gfile.FastGFile(readdir + filename, 'rb').read()

        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        try:
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        except Exception:
            print 'brokenFile ' + filename
            continue

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        winner = top_k[0]
        score = predictions[0][winner]
        name = label_lines[winner]
        if (score > scorelimit):
            if name != "face" and name != "background":
                print('%s (score = %.5f) %s' % (name, score, filename))
                shutil.move(readdir + filename, outputdir + name.replace(" ", "-") + '/' + filename)
            if name == "background":
                print 'trash'
                shutil.move(readdir + filename, "./photos/trash/" + filename)
        # for node_id in top_k:
        #     human_string = label_lines[node_id]
        #     score = predictions[0][node_id]
        #     print score
        #     print('%s (score = %.5f)' % (human_string, score))