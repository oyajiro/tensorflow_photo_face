import tensorflow as tf
import sys
import shutil
import os


readdir = "./photos/output/"
outputdir = './output/'
graphname = 'ss_graph.pb'
labelsname = 'ss_labels.txt'
scorelimit = 0.5

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
            print('%s (score = %.5f)' % (name, score))
            outputname = name.replace(" ", "-")
            if (score > 0.9):
                if not os.path.exists(outputdir + 'sure/' + outputname):
                    os.makedirs(outputdir + 'sure/' + outputname)
                shutil.move(readdir + filename, outputdir + 'sure/' + outputname + '/' + str(int(score * 100)) + '_' + filename)
            else:
                if not os.path.exists(outputdir + 'notsure/' + outputname):
                    os.makedirs(outputdir + 'notsure/' + outputname)
                shutil.move(readdir + filename, outputdir + 'notsure/' + outputname + '/' + str(int(score * 100)) + '_' + filename)
        # for node_id in top_k:
        #     human_string = label_lines[node_id]
        #     score = predictions[0][node_id]
        #     print score
        #     print('%s (score = %.5f)' % (human_string, score))