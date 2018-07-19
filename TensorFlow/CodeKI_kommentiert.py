
Manjas-MacBook-Pro:~ manjangoc$ git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
# Einbinden des Tensorflow for poets 2 Reposetoris über github 

Cloning into 'tensorflow-for-poets-2'...
remote: Counting objects: 405, done.
remote: Total 405 (delta 0), reused 0 (delta 0), pack-reused 405
Receiving objects: 100% (405/405), 33.96 MiB | 7.08 MiB/s, done.
Resolving deltas: 100% (149/149), done.
Checking out files: 100% (142/142), done.

Manjas-MacBook-Pro:~ manjangoc$ cd tensorflow-for-poets-2
# öffnen des Ordners tensorflow for poets 2 

Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ ls tf_files/KI
# Das Verzeichnis KI wird ausgegeben

Flecken   Menschen  Text    Tiere
# Die Ordner Flecken, Menschen, Text und Tiere werden erstellt

Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ IMAGE_SIZE=224
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

# Die Trainingsphase des neuronalen Netzwerkes beginnt 
    
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps= \ # Wir arbeiten mit unendlich vielen Trainingssteps
  --model_dir=tf_files/models/ \ #der Ordner "models" wird als Unterordner in tf_files erstellt
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \ #erstellt den Output graphen namens "retrained_graph.pb" mit den Trainingsdaten
  --output_labels=tf_files/retrained_labels.txt \ #erstellt eine txt mit den labels (Kategorien)
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/KI #Pfad aus welchen die Trainingsdaten geholt werden sollen

# bottlenecks werden erstellt: bottelnecks sind die letzte Ebende vor der tatsächlichen Klassifizierung 
# Zu dem Ordner KI wurden in ca. 4000 Training steps verschiedene Inhalte erstellt:
# tf_files, summaries, graph, labels 
# Das Training läuft weiterhin


/Users/manjangoc/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
usage: retrain.py [-h] [--image_dir IMAGE_DIR] [--output_graph OUTPUT_GRAPH]
                  [--intermediate_output_graphs_dir INTERMEDIATE_OUTPUT_GRAPHS_DIR]
                  [--intermediate_store_frequency INTERMEDIATE_STORE_FREQUENCY]
                  [--output_labels OUTPUT_LABELS]
                  [--summaries_dir SUMMARIES_DIR]
                  [--how_many_training_steps HOW_MANY_TRAINING_STEPS]
                  [--learning_rate LEARNING_RATE]
                  [--testing_percentage TESTING_PERCENTAGE]
                  [--validation_percentage VALIDATION_PERCENTAGE]
                  [--eval_step_interval EVAL_STEP_INTERVAL]
                  [--train_batch_size TRAIN_BATCH_SIZE]
                  [--test_batch_size TEST_BATCH_SIZE]
                  [--validation_batch_size VALIDATION_BATCH_SIZE]
                  [--print_misclassified_test_images] [--model_dir MODEL_DIR]
                  [--bottleneck_dir BOTTLENECK_DIR]
                  [--final_tensor_name FINAL_TENSOR_NAME] [--flip_left_right]
                  [--random_crop RANDOM_CROP] [--random_scale RANDOM_SCALE]
                  [--random_brightness RANDOM_BRIGHTNESS]
                  [--architecture ARCHITECTURE]
retrain.py: error: argument --how_many_training_steps: invalid int value: ''
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ 
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ python -m scripts.retrain \
>   --bottleneck_dir=tf_files/bottlenecks \
>   --model_dir=tf_files/models/"${ARCHITECTURE}" \
>   --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
>   --output_graph=tf_files/retrained_graph.pb \
>   --output_labels=tf_files/retrained_labels.txt \
>   --architecture="${ARCHITECTURE}" \
>   --image_dir=tf_files/KI
/Users/manjangoc/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
>> Downloading mobilenet_v1_0.50_224_frozen.tgz 100.1%
Traceback (most recent call last):
  File "/Users/manjangoc/anaconda2/lib/python2.7/logging/__init__.py", line 861, in emit
    msg = self.format(record)
  File "/Users/manjangoc/anaconda2/lib/python2.7/logging/__init__.py", line 734, in format
    return fmt.format(record)
  File "/Users/manjangoc/anaconda2/lib/python2.7/logging/__init__.py", line 465, in format
    record.message = record.getMessage()
  File "/Users/manjangoc/anaconda2/lib/python2.7/logging/__init__.py", line 329, in getMessage
    msg = msg % self.args
TypeError: not all arguments converted during string formatting
Logged from file tf_logging.py, line 116
INFO:tensorflow:Looking for images in 'Flecken'
INFO:tensorflow:Looking for images in 'Menschen'
INFO:tensorflow:Looking for images in 'Text'
WARNING:tensorflow:WARNING: Folder has less than 20 images, which may cause issues.
INFO:tensorflow:Looking for images in 'Tiere'
INFO:tensorflow:100 bottleneck files created.
INFO:tensorflow:200 bottleneck files created.
WARNING:tensorflow:From /Users/manjangoc/tensorflow-for-poets-2/scripts/retrain.py:790: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.

INFO:tensorflow:2018-06-30 18:53:20.679989: Step 0: Train accuracy = 88.0%
INFO:tensorflow:2018-06-30 18:53:20.680239: Step 0: Cross entropy = 0.382416
INFO:tensorflow:2018-06-30 18:53:20.972077: Step 0: Validation accuracy = 90.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:21.581930: Step 10: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:21.582133: Step 10: Cross entropy = 0.028116
INFO:tensorflow:2018-06-30 18:53:21.644503: Step 10: Validation accuracy = 96.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:22.256181: Step 20: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:22.256386: Step 20: Cross entropy = 0.049478
INFO:tensorflow:2018-06-30 18:53:22.318417: Step 20: Validation accuracy = 93.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:22.922666: Step 30: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:22.922895: Step 30: Cross entropy = 0.028582
INFO:tensorflow:2018-06-30 18:53:22.985818: Step 30: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:23.615153: Step 40: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:23.615358: Step 40: Cross entropy = 0.027384
INFO:tensorflow:2018-06-30 18:53:23.673060: Step 40: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:24.293913: Step 50: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:24.294107: Step 50: Cross entropy = 0.014835
INFO:tensorflow:2018-06-30 18:53:24.357020: Step 50: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:24.946887: Step 60: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:24.947082: Step 60: Cross entropy = 0.012794
INFO:tensorflow:2018-06-30 18:53:25.010098: Step 60: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:25.629089: Step 70: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:25.629384: Step 70: Cross entropy = 0.012292
INFO:tensorflow:2018-06-30 18:53:25.690407: Step 70: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:26.303971: Step 80: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:26.304159: Step 80: Cross entropy = 0.008936
INFO:tensorflow:2018-06-30 18:53:26.368000: Step 80: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:26.971780: Step 90: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:26.972003: Step 90: Cross entropy = 0.009221
INFO:tensorflow:2018-06-30 18:53:27.038378: Step 90: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:27.641037: Step 100: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:27.641233: Step 100: Cross entropy = 0.008015
INFO:tensorflow:2018-06-30 18:53:27.699787: Step 100: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:28.314497: Step 110: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:28.314701: Step 110: Cross entropy = 0.007257
INFO:tensorflow:2018-06-30 18:53:28.379542: Step 110: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:28.981496: Step 120: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:28.981696: Step 120: Cross entropy = 0.009711
INFO:tensorflow:2018-06-30 18:53:29.043793: Step 120: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:29.658731: Step 130: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:29.658926: Step 130: Cross entropy = 0.008726
INFO:tensorflow:2018-06-30 18:53:29.723692: Step 130: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:30.326877: Step 140: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:30.327088: Step 140: Cross entropy = 0.007702
INFO:tensorflow:2018-06-30 18:53:30.388798: Step 140: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:30.989343: Step 150: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:30.989538: Step 150: Cross entropy = 0.008411
INFO:tensorflow:2018-06-30 18:53:31.054117: Step 150: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:31.667443: Step 160: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:31.667636: Step 160: Cross entropy = 0.007296
INFO:tensorflow:2018-06-30 18:53:31.725959: Step 160: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:32.339781: Step 170: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:32.339991: Step 170: Cross entropy = 0.005886
INFO:tensorflow:2018-06-30 18:53:32.400918: Step 170: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:33.003909: Step 180: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:33.004101: Step 180: Cross entropy = 0.007738
INFO:tensorflow:2018-06-30 18:53:33.065391: Step 180: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:33.680657: Step 190: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:33.680846: Step 190: Cross entropy = 0.006301
INFO:tensorflow:2018-06-30 18:53:33.744234: Step 190: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:34.436206: Step 200: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:34.436584: Step 200: Cross entropy = 0.007692
INFO:tensorflow:2018-06-30 18:53:34.524010: Step 200: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:35.225983: Step 210: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:35.226183: Step 210: Cross entropy = 0.007252
INFO:tensorflow:2018-06-30 18:53:35.287378: Step 210: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:35.894812: Step 220: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:35.895016: Step 220: Cross entropy = 0.004468
INFO:tensorflow:2018-06-30 18:53:35.955055: Step 220: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:36.554309: Step 230: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:36.554538: Step 230: Cross entropy = 0.007991
INFO:tensorflow:2018-06-30 18:53:36.616545: Step 230: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:37.221648: Step 240: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:37.221854: Step 240: Cross entropy = 0.004458
INFO:tensorflow:2018-06-30 18:53:37.282731: Step 240: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:37.895255: Step 250: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:37.895454: Step 250: Cross entropy = 0.004849
INFO:tensorflow:2018-06-30 18:53:37.956220: Step 250: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:38.572090: Step 260: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:38.572293: Step 260: Cross entropy = 0.004871
INFO:tensorflow:2018-06-30 18:53:38.630462: Step 260: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:39.250864: Step 270: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:39.251056: Step 270: Cross entropy = 0.005380
INFO:tensorflow:2018-06-30 18:53:39.314706: Step 270: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:39.929141: Step 280: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:39.929361: Step 280: Cross entropy = 0.007960
INFO:tensorflow:2018-06-30 18:53:39.990853: Step 280: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:40.597880: Step 290: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:40.598081: Step 290: Cross entropy = 0.005042
INFO:tensorflow:2018-06-30 18:53:40.659222: Step 290: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:41.269183: Step 300: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:41.269376: Step 300: Cross entropy = 0.005164
INFO:tensorflow:2018-06-30 18:53:41.332273: Step 300: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:41.935837: Step 310: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:41.936044: Step 310: Cross entropy = 0.004249
INFO:tensorflow:2018-06-30 18:53:42.003779: Step 310: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:42.603194: Step 320: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:42.603394: Step 320: Cross entropy = 0.004387
INFO:tensorflow:2018-06-30 18:53:42.661094: Step 320: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:43.497284: Step 330: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:43.497676: Step 330: Cross entropy = 0.003639
INFO:tensorflow:2018-06-30 18:53:43.566200: Step 330: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:44.183879: Step 340: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:44.184080: Step 340: Cross entropy = 0.003862
INFO:tensorflow:2018-06-30 18:53:44.246564: Step 340: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:44.898534: Step 350: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:44.898743: Step 350: Cross entropy = 0.003489
INFO:tensorflow:2018-06-30 18:53:44.961900: Step 350: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:45.607103: Step 360: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:45.607321: Step 360: Cross entropy = 0.003816
INFO:tensorflow:2018-06-30 18:53:45.675169: Step 360: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:46.471588: Step 370: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:46.472285: Step 370: Cross entropy = 0.004535
INFO:tensorflow:2018-06-30 18:53:46.580258: Step 370: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:47.318633: Step 380: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:47.318863: Step 380: Cross entropy = 0.003409
INFO:tensorflow:2018-06-30 18:53:47.390111: Step 380: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:48.246156: Step 390: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:48.246369: Step 390: Cross entropy = 0.003293
INFO:tensorflow:2018-06-30 18:53:48.373073: Step 390: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:49.032238: Step 400: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:49.032468: Step 400: Cross entropy = 0.003974
INFO:tensorflow:2018-06-30 18:53:49.099444: Step 400: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:49.768566: Step 410: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:49.768758: Step 410: Cross entropy = 0.004037
INFO:tensorflow:2018-06-30 18:53:49.838418: Step 410: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:50.586763: Step 420: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:50.587033: Step 420: Cross entropy = 0.003016
INFO:tensorflow:2018-06-30 18:53:50.657939: Step 420: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:51.354690: Step 430: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:51.354882: Step 430: Cross entropy = 0.003690
INFO:tensorflow:2018-06-30 18:53:51.425720: Step 430: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:52.107872: Step 440: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:52.108370: Step 440: Cross entropy = 0.003153
INFO:tensorflow:2018-06-30 18:53:52.171230: Step 440: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:52.850986: Step 450: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:52.851238: Step 450: Cross entropy = 0.002750
INFO:tensorflow:2018-06-30 18:53:52.918683: Step 450: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:53.572817: Step 460: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:53.573011: Step 460: Cross entropy = 0.003846
INFO:tensorflow:2018-06-30 18:53:53.637948: Step 460: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:54.294926: Step 470: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:54.295145: Step 470: Cross entropy = 0.004272
INFO:tensorflow:2018-06-30 18:53:54.362511: Step 470: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:55.174690: Step 480: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:55.174916: Step 480: Cross entropy = 0.004356
INFO:tensorflow:2018-06-30 18:53:55.274131: Step 480: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:55.985877: Step 490: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:55.986091: Step 490: Cross entropy = 0.003005
INFO:tensorflow:2018-06-30 18:53:56.075048: Step 490: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:56.724119: Step 500: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:56.724328: Step 500: Cross entropy = 0.002784
INFO:tensorflow:2018-06-30 18:53:56.783937: Step 500: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:57.431619: Step 510: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:57.431921: Step 510: Cross entropy = 0.002013
INFO:tensorflow:2018-06-30 18:53:57.497308: Step 510: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:58.210542: Step 520: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:58.210823: Step 520: Cross entropy = 0.002215
INFO:tensorflow:2018-06-30 18:53:58.285792: Step 520: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:58.957940: Step 530: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:58.958131: Step 530: Cross entropy = 0.002838
INFO:tensorflow:2018-06-30 18:53:59.034818: Step 530: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:53:59.643663: Step 540: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:53:59.643854: Step 540: Cross entropy = 0.003092
INFO:tensorflow:2018-06-30 18:53:59.715519: Step 540: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:00.461836: Step 550: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:00.462031: Step 550: Cross entropy = 0.001951
INFO:tensorflow:2018-06-30 18:54:00.532416: Step 550: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:01.185217: Step 560: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:01.185422: Step 560: Cross entropy = 0.001864
INFO:tensorflow:2018-06-30 18:54:01.257989: Step 560: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:01.979561: Step 570: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:01.979801: Step 570: Cross entropy = 0.001583
INFO:tensorflow:2018-06-30 18:54:02.060240: Step 570: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:02.720947: Step 580: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:02.721142: Step 580: Cross entropy = 0.002456
INFO:tensorflow:2018-06-30 18:54:02.795407: Step 580: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:03.600607: Step 590: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:03.600797: Step 590: Cross entropy = 0.002684
INFO:tensorflow:2018-06-30 18:54:03.679171: Step 590: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:04.400011: Step 600: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:04.400326: Step 600: Cross entropy = 0.002231
INFO:tensorflow:2018-06-30 18:54:04.498221: Step 600: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:05.455572: Step 610: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:05.455782: Step 610: Cross entropy = 0.002058
INFO:tensorflow:2018-06-30 18:54:05.523184: Step 610: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:06.214047: Step 620: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:06.214346: Step 620: Cross entropy = 0.002258
INFO:tensorflow:2018-06-30 18:54:06.284658: Step 620: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:06.959206: Step 630: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:06.959518: Step 630: Cross entropy = 0.002708
INFO:tensorflow:2018-06-30 18:54:07.027543: Step 630: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:07.731636: Step 640: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:07.731828: Step 640: Cross entropy = 0.002553
INFO:tensorflow:2018-06-30 18:54:07.800212: Step 640: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:08.480064: Step 650: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:08.480257: Step 650: Cross entropy = 0.001629
INFO:tensorflow:2018-06-30 18:54:08.545153: Step 650: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:09.310755: Step 660: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:09.311490: Step 660: Cross entropy = 0.001919
INFO:tensorflow:2018-06-30 18:54:09.392770: Step 660: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:10.088758: Step 670: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:10.088964: Step 670: Cross entropy = 0.002039
INFO:tensorflow:2018-06-30 18:54:10.158362: Step 670: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:10.827779: Step 680: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:10.828000: Step 680: Cross entropy = 0.002499
INFO:tensorflow:2018-06-30 18:54:10.898004: Step 680: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:11.775602: Step 690: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:11.775800: Step 690: Cross entropy = 0.001807
INFO:tensorflow:2018-06-30 18:54:11.858832: Step 690: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:12.544219: Step 700: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:12.544493: Step 700: Cross entropy = 0.002553
INFO:tensorflow:2018-06-30 18:54:12.651699: Step 700: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:13.295567: Step 710: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:13.295761: Step 710: Cross entropy = 0.001532
INFO:tensorflow:2018-06-30 18:54:13.366546: Step 710: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:14.526747: Step 720: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:14.526953: Step 720: Cross entropy = 0.001767
INFO:tensorflow:2018-06-30 18:54:14.603279: Step 720: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:15.330361: Step 730: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:15.330561: Step 730: Cross entropy = 0.002068
INFO:tensorflow:2018-06-30 18:54:15.408758: Step 730: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:16.041114: Step 740: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:16.041303: Step 740: Cross entropy = 0.002077
INFO:tensorflow:2018-06-30 18:54:16.098913: Step 740: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:16.791675: Step 750: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:16.791889: Step 750: Cross entropy = 0.002593
INFO:tensorflow:2018-06-30 18:54:16.859142: Step 750: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:17.535232: Step 760: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:17.535420: Step 760: Cross entropy = 0.001581
INFO:tensorflow:2018-06-30 18:54:17.600579: Step 760: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:18.312544: Step 770: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:18.312747: Step 770: Cross entropy = 0.001809
INFO:tensorflow:2018-06-30 18:54:18.380529: Step 770: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:19.032734: Step 780: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:19.032942: Step 780: Cross entropy = 0.001906
INFO:tensorflow:2018-06-30 18:54:19.096958: Step 780: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:19.809808: Step 790: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:19.810022: Step 790: Cross entropy = 0.001792
INFO:tensorflow:2018-06-30 18:54:19.872697: Step 790: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:20.838285: Step 800: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:20.838471: Step 800: Cross entropy = 0.001590
INFO:tensorflow:2018-06-30 18:54:20.899753: Step 800: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:21.731666: Step 810: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:21.731879: Step 810: Cross entropy = 0.001214
INFO:tensorflow:2018-06-30 18:54:21.794195: Step 810: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:22.430202: Step 820: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:22.430399: Step 820: Cross entropy = 0.002005
INFO:tensorflow:2018-06-30 18:54:22.489097: Step 820: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:23.112231: Step 830: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:23.112435: Step 830: Cross entropy = 0.001855
INFO:tensorflow:2018-06-30 18:54:23.172613: Step 830: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:23.809428: Step 840: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:23.809617: Step 840: Cross entropy = 0.001685
INFO:tensorflow:2018-06-30 18:54:23.876072: Step 840: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:24.498611: Step 850: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:24.498827: Step 850: Cross entropy = 0.001202
INFO:tensorflow:2018-06-30 18:54:24.572459: Step 850: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:25.330566: Step 860: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:25.330778: Step 860: Cross entropy = 0.001798
INFO:tensorflow:2018-06-30 18:54:25.419009: Step 860: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:26.262147: Step 870: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:26.262551: Step 870: Cross entropy = 0.001698
INFO:tensorflow:2018-06-30 18:54:26.328670: Step 870: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:27.092484: Step 880: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:27.092728: Step 880: Cross entropy = 0.001444
INFO:tensorflow:2018-06-30 18:54:27.182252: Step 880: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:27.952884: Step 890: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:27.953095: Step 890: Cross entropy = 0.001636
INFO:tensorflow:2018-06-30 18:54:28.026930: Step 890: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:28.763084: Step 900: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:28.763278: Step 900: Cross entropy = 0.001667
INFO:tensorflow:2018-06-30 18:54:28.828255: Step 900: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:29.458164: Step 910: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:29.458360: Step 910: Cross entropy = 0.001707
INFO:tensorflow:2018-06-30 18:54:29.523180: Step 910: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:30.145200: Step 920: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:30.145403: Step 920: Cross entropy = 0.001385
INFO:tensorflow:2018-06-30 18:54:30.206081: Step 920: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:30.808640: Step 930: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:30.808836: Step 930: Cross entropy = 0.001388
INFO:tensorflow:2018-06-30 18:54:30.869538: Step 930: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:31.489822: Step 940: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:31.490012: Step 940: Cross entropy = 0.001704
INFO:tensorflow:2018-06-30 18:54:31.551906: Step 940: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:32.165846: Step 950: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:32.166037: Step 950: Cross entropy = 0.001209
INFO:tensorflow:2018-06-30 18:54:32.228641: Step 950: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:32.844564: Step 960: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:32.844765: Step 960: Cross entropy = 0.001482
INFO:tensorflow:2018-06-30 18:54:32.904973: Step 960: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:33.603941: Step 970: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:33.604136: Step 970: Cross entropy = 0.001307
INFO:tensorflow:2018-06-30 18:54:33.667387: Step 970: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:34.294292: Step 980: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:34.294493: Step 980: Cross entropy = 0.001015
INFO:tensorflow:2018-06-30 18:54:34.357052: Step 980: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:34.982602: Step 990: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:34.982807: Step 990: Cross entropy = 0.001572
INFO:tensorflow:2018-06-30 18:54:35.046647: Step 990: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:35.665253: Step 1000: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:35.665475: Step 1000: Cross entropy = 0.001890
INFO:tensorflow:2018-06-30 18:54:35.729142: Step 1000: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:36.337824: Step 1010: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:36.338019: Step 1010: Cross entropy = 0.001299
INFO:tensorflow:2018-06-30 18:54:36.403467: Step 1010: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:37.008653: Step 1020: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:37.008880: Step 1020: Cross entropy = 0.001348
INFO:tensorflow:2018-06-30 18:54:37.067745: Step 1020: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:37.672024: Step 1030: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:37.672225: Step 1030: Cross entropy = 0.002016
INFO:tensorflow:2018-06-30 18:54:37.729849: Step 1030: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:38.349330: Step 1040: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:38.349534: Step 1040: Cross entropy = 0.001240
INFO:tensorflow:2018-06-30 18:54:38.411937: Step 1040: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:39.017070: Step 1050: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:39.017262: Step 1050: Cross entropy = 0.000999
INFO:tensorflow:2018-06-30 18:54:39.078160: Step 1050: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:39.743315: Step 1060: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:39.743532: Step 1060: Cross entropy = 0.000931
INFO:tensorflow:2018-06-30 18:54:39.811941: Step 1060: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:40.546074: Step 1070: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:40.546289: Step 1070: Cross entropy = 0.001237
INFO:tensorflow:2018-06-30 18:54:40.651897: Step 1070: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:41.318491: Step 1080: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:41.318687: Step 1080: Cross entropy = 0.001637
INFO:tensorflow:2018-06-30 18:54:41.385647: Step 1080: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:42.005375: Step 1090: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:42.005586: Step 1090: Cross entropy = 0.001734
INFO:tensorflow:2018-06-30 18:54:42.064013: Step 1090: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:42.689711: Step 1100: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:42.689908: Step 1100: Cross entropy = 0.001358
INFO:tensorflow:2018-06-30 18:54:42.749529: Step 1100: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:43.371256: Step 1110: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:43.371461: Step 1110: Cross entropy = 0.001216
INFO:tensorflow:2018-06-30 18:54:43.436061: Step 1110: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:44.274927: Step 1120: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:44.275124: Step 1120: Cross entropy = 0.000952
INFO:tensorflow:2018-06-30 18:54:44.333741: Step 1120: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:45.110849: Step 1130: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:45.111072: Step 1130: Cross entropy = 0.001790
INFO:tensorflow:2018-06-30 18:54:45.171122: Step 1130: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:45.787457: Step 1140: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:45.787650: Step 1140: Cross entropy = 0.000841
INFO:tensorflow:2018-06-30 18:54:45.847690: Step 1140: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:46.467015: Step 1150: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:46.467218: Step 1150: Cross entropy = 0.000913
INFO:tensorflow:2018-06-30 18:54:46.527701: Step 1150: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:47.156235: Step 1160: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:47.156429: Step 1160: Cross entropy = 0.001479
INFO:tensorflow:2018-06-30 18:54:47.213814: Step 1160: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:47.835801: Step 1170: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:47.836009: Step 1170: Cross entropy = 0.000779
INFO:tensorflow:2018-06-30 18:54:47.896537: Step 1170: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:48.512140: Step 1180: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:48.512331: Step 1180: Cross entropy = 0.001011
INFO:tensorflow:2018-06-30 18:54:48.572490: Step 1180: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:49.170629: Step 1190: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:49.170834: Step 1190: Cross entropy = 0.000756
INFO:tensorflow:2018-06-30 18:54:49.232813: Step 1190: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:49.853996: Step 1200: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:49.854219: Step 1200: Cross entropy = 0.001917
INFO:tensorflow:2018-06-30 18:54:49.916217: Step 1200: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:50.532357: Step 1210: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:50.532560: Step 1210: Cross entropy = 0.001161
INFO:tensorflow:2018-06-30 18:54:50.594117: Step 1210: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:51.287788: Step 1220: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:51.288005: Step 1220: Cross entropy = 0.001682
INFO:tensorflow:2018-06-30 18:54:51.378717: Step 1220: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:52.139931: Step 1230: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:52.140132: Step 1230: Cross entropy = 0.001109
INFO:tensorflow:2018-06-30 18:54:52.200907: Step 1230: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:52.813771: Step 1240: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:52.813992: Step 1240: Cross entropy = 0.001117
INFO:tensorflow:2018-06-30 18:54:52.879003: Step 1240: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:53.497847: Step 1250: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:53.498047: Step 1250: Cross entropy = 0.000659
INFO:tensorflow:2018-06-30 18:54:53.557918: Step 1250: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:54.161701: Step 1260: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:54.161897: Step 1260: Cross entropy = 0.001309
INFO:tensorflow:2018-06-30 18:54:54.223046: Step 1260: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:54.832067: Step 1270: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:54.832263: Step 1270: Cross entropy = 0.001286
INFO:tensorflow:2018-06-30 18:54:54.892321: Step 1270: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:55.505691: Step 1280: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:55.505914: Step 1280: Cross entropy = 0.001306
INFO:tensorflow:2018-06-30 18:54:55.567526: Step 1280: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:56.178020: Step 1290: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:56.178241: Step 1290: Cross entropy = 0.000793
INFO:tensorflow:2018-06-30 18:54:56.239161: Step 1290: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:56.886742: Step 1300: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:56.886936: Step 1300: Cross entropy = 0.000857
INFO:tensorflow:2018-06-30 18:54:56.947934: Step 1300: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:57.562014: Step 1310: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:57.562220: Step 1310: Cross entropy = 0.001556
INFO:tensorflow:2018-06-30 18:54:57.628074: Step 1310: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:58.245368: Step 1320: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:58.245564: Step 1320: Cross entropy = 0.001160
INFO:tensorflow:2018-06-30 18:54:58.311409: Step 1320: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:58.924429: Step 1330: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:58.924636: Step 1330: Cross entropy = 0.001023
INFO:tensorflow:2018-06-30 18:54:58.982709: Step 1330: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:54:59.602193: Step 1340: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:54:59.602397: Step 1340: Cross entropy = 0.000749
INFO:tensorflow:2018-06-30 18:54:59.667390: Step 1340: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:00.279858: Step 1350: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:00.280118: Step 1350: Cross entropy = 0.001209
INFO:tensorflow:2018-06-30 18:55:00.343390: Step 1350: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:00.955933: Step 1360: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:00.956139: Step 1360: Cross entropy = 0.001367
INFO:tensorflow:2018-06-30 18:55:01.016070: Step 1360: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:01.644387: Step 1370: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:01.644598: Step 1370: Cross entropy = 0.000929
INFO:tensorflow:2018-06-30 18:55:01.705080: Step 1370: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:02.325653: Step 1380: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:02.325851: Step 1380: Cross entropy = 0.001308
INFO:tensorflow:2018-06-30 18:55:02.391925: Step 1380: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:03.008284: Step 1390: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:03.008479: Step 1390: Cross entropy = 0.001096
INFO:tensorflow:2018-06-30 18:55:03.070628: Step 1390: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:03.693452: Step 1400: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:03.693660: Step 1400: Cross entropy = 0.000872
INFO:tensorflow:2018-06-30 18:55:03.752981: Step 1400: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:04.360535: Step 1410: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:04.360738: Step 1410: Cross entropy = 0.000946
INFO:tensorflow:2018-06-30 18:55:04.420556: Step 1410: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:05.035444: Step 1420: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:05.035639: Step 1420: Cross entropy = 0.000896
INFO:tensorflow:2018-06-30 18:55:05.099560: Step 1420: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:05.791822: Step 1430: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:05.792046: Step 1430: Cross entropy = 0.001140
INFO:tensorflow:2018-06-30 18:55:05.856048: Step 1430: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:06.472686: Step 1440: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:06.472881: Step 1440: Cross entropy = 0.000889
INFO:tensorflow:2018-06-30 18:55:06.553205: Step 1440: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:07.353835: Step 1450: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:07.354049: Step 1450: Cross entropy = 0.001176
INFO:tensorflow:2018-06-30 18:55:07.416195: Step 1450: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:08.039324: Step 1460: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:08.039526: Step 1460: Cross entropy = 0.001428
INFO:tensorflow:2018-06-30 18:55:08.100941: Step 1460: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:08.708078: Step 1470: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:08.708284: Step 1470: Cross entropy = 0.000681
INFO:tensorflow:2018-06-30 18:55:08.775168: Step 1470: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:09.384384: Step 1480: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:09.384576: Step 1480: Cross entropy = 0.000859
INFO:tensorflow:2018-06-30 18:55:09.447201: Step 1480: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:10.052773: Step 1490: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:10.052971: Step 1490: Cross entropy = 0.000686
INFO:tensorflow:2018-06-30 18:55:10.114852: Step 1490: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:10.724284: Step 1500: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:10.724477: Step 1500: Cross entropy = 0.001162
INFO:tensorflow:2018-06-30 18:55:10.784410: Step 1500: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:11.403292: Step 1510: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:11.403494: Step 1510: Cross entropy = 0.000869
INFO:tensorflow:2018-06-30 18:55:11.465259: Step 1510: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:12.081353: Step 1520: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:12.081557: Step 1520: Cross entropy = 0.000610
INFO:tensorflow:2018-06-30 18:55:12.140562: Step 1520: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:12.757189: Step 1530: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:12.757389: Step 1530: Cross entropy = 0.001022
INFO:tensorflow:2018-06-30 18:55:12.819372: Step 1530: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:13.454316: Step 1540: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:13.454507: Step 1540: Cross entropy = 0.000880
INFO:tensorflow:2018-06-30 18:55:13.513349: Step 1540: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:14.124661: Step 1550: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:14.124865: Step 1550: Cross entropy = 0.000684
INFO:tensorflow:2018-06-30 18:55:14.184204: Step 1550: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:14.798736: Step 1560: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:14.798939: Step 1560: Cross entropy = 0.000984
INFO:tensorflow:2018-06-30 18:55:14.854478: Step 1560: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:15.463834: Step 1570: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:15.464071: Step 1570: Cross entropy = 0.000826
INFO:tensorflow:2018-06-30 18:55:15.529980: Step 1570: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:16.169615: Step 1580: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:16.169816: Step 1580: Cross entropy = 0.000831
INFO:tensorflow:2018-06-30 18:55:16.229346: Step 1580: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:16.843457: Step 1590: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:16.843651: Step 1590: Cross entropy = 0.000664
INFO:tensorflow:2018-06-30 18:55:16.904486: Step 1590: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:17.519866: Step 1600: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:17.520061: Step 1600: Cross entropy = 0.001105
INFO:tensorflow:2018-06-30 18:55:17.578977: Step 1600: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:18.237888: Step 1610: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:18.238083: Step 1610: Cross entropy = 0.001463
INFO:tensorflow:2018-06-30 18:55:18.310299: Step 1610: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:18.966975: Step 1620: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:18.967200: Step 1620: Cross entropy = 0.000799
INFO:tensorflow:2018-06-30 18:55:19.036443: Step 1620: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:19.701129: Step 1630: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:19.701373: Step 1630: Cross entropy = 0.000835
INFO:tensorflow:2018-06-30 18:55:19.764469: Step 1630: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:20.414323: Step 1640: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:20.414534: Step 1640: Cross entropy = 0.000852
INFO:tensorflow:2018-06-30 18:55:20.476266: Step 1640: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:21.092464: Step 1650: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:21.092657: Step 1650: Cross entropy = 0.000925
INFO:tensorflow:2018-06-30 18:55:21.152962: Step 1650: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:21.810745: Step 1660: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:21.810953: Step 1660: Cross entropy = 0.000835
INFO:tensorflow:2018-06-30 18:55:21.879103: Step 1660: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:22.499017: Step 1670: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:22.499222: Step 1670: Cross entropy = 0.001152
INFO:tensorflow:2018-06-30 18:55:22.561695: Step 1670: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:23.178884: Step 1680: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:23.179090: Step 1680: Cross entropy = 0.000598
INFO:tensorflow:2018-06-30 18:55:23.239772: Step 1680: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:23.863166: Step 1690: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:23.863369: Step 1690: Cross entropy = 0.000376
INFO:tensorflow:2018-06-30 18:55:23.919817: Step 1690: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:24.552097: Step 1700: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:24.552333: Step 1700: Cross entropy = 0.000932
INFO:tensorflow:2018-06-30 18:55:24.619188: Step 1700: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:25.343733: Step 1710: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:25.343928: Step 1710: Cross entropy = 0.000864
INFO:tensorflow:2018-06-30 18:55:25.405089: Step 1710: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:26.033509: Step 1720: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:26.033705: Step 1720: Cross entropy = 0.000897
INFO:tensorflow:2018-06-30 18:55:26.094438: Step 1720: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:26.747301: Step 1730: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:26.747508: Step 1730: Cross entropy = 0.000995
INFO:tensorflow:2018-06-30 18:55:26.812495: Step 1730: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:27.432817: Step 1740: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:27.433010: Step 1740: Cross entropy = 0.001098
INFO:tensorflow:2018-06-30 18:55:27.493723: Step 1740: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:28.107723: Step 1750: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:28.107927: Step 1750: Cross entropy = 0.000675
INFO:tensorflow:2018-06-30 18:55:28.168502: Step 1750: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:28.810677: Step 1760: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:28.810874: Step 1760: Cross entropy = 0.000862
INFO:tensorflow:2018-06-30 18:55:28.886762: Step 1760: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:29.575692: Step 1770: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:29.575881: Step 1770: Cross entropy = 0.000626
INFO:tensorflow:2018-06-30 18:55:29.643153: Step 1770: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:30.359837: Step 1780: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:30.360045: Step 1780: Cross entropy = 0.000612
INFO:tensorflow:2018-06-30 18:55:30.426611: Step 1780: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:31.103388: Step 1790: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:31.103594: Step 1790: Cross entropy = 0.000662
INFO:tensorflow:2018-06-30 18:55:31.168820: Step 1790: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:31.783771: Step 1800: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:31.783992: Step 1800: Cross entropy = 0.000746
INFO:tensorflow:2018-06-30 18:55:31.848070: Step 1800: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:32.462772: Step 1810: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:32.462965: Step 1810: Cross entropy = 0.000697
INFO:tensorflow:2018-06-30 18:55:32.524603: Step 1810: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:33.207268: Step 1820: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:33.207498: Step 1820: Cross entropy = 0.000887
INFO:tensorflow:2018-06-30 18:55:33.273863: Step 1820: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:34.005976: Step 1830: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:34.008374: Step 1830: Cross entropy = 0.000620
INFO:tensorflow:2018-06-30 18:55:34.105713: Step 1830: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:34.751506: Step 1840: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:34.751715: Step 1840: Cross entropy = 0.000849
INFO:tensorflow:2018-06-30 18:55:34.818524: Step 1840: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:35.461201: Step 1850: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:35.461404: Step 1850: Cross entropy = 0.000503
INFO:tensorflow:2018-06-30 18:55:35.528024: Step 1850: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:36.166676: Step 1860: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:36.166870: Step 1860: Cross entropy = 0.000766
INFO:tensorflow:2018-06-30 18:55:36.228641: Step 1860: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:36.865535: Step 1870: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:36.865735: Step 1870: Cross entropy = 0.000615
INFO:tensorflow:2018-06-30 18:55:36.930284: Step 1870: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:37.563265: Step 1880: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:37.563466: Step 1880: Cross entropy = 0.000841
INFO:tensorflow:2018-06-30 18:55:37.627125: Step 1880: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:38.267286: Step 1890: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:38.267478: Step 1890: Cross entropy = 0.001021
INFO:tensorflow:2018-06-30 18:55:38.334104: Step 1890: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:39.020262: Step 1900: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:39.020456: Step 1900: Cross entropy = 0.000666
INFO:tensorflow:2018-06-30 18:55:39.086091: Step 1900: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:39.720917: Step 1910: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:39.721123: Step 1910: Cross entropy = 0.000663
INFO:tensorflow:2018-06-30 18:55:39.781461: Step 1910: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:40.410410: Step 1920: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:40.410610: Step 1920: Cross entropy = 0.000588
INFO:tensorflow:2018-06-30 18:55:40.474118: Step 1920: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:41.093578: Step 1930: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:41.093771: Step 1930: Cross entropy = 0.000740
INFO:tensorflow:2018-06-30 18:55:41.156220: Step 1930: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:41.774140: Step 1940: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:41.774336: Step 1940: Cross entropy = 0.000876
INFO:tensorflow:2018-06-30 18:55:41.831371: Step 1940: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:42.451104: Step 1950: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:42.451324: Step 1950: Cross entropy = 0.000650
INFO:tensorflow:2018-06-30 18:55:42.512825: Step 1950: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:43.140985: Step 1960: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:43.141211: Step 1960: Cross entropy = 0.000933
INFO:tensorflow:2018-06-30 18:55:43.206665: Step 1960: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:44.560699: Step 1970: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:44.560894: Step 1970: Cross entropy = 0.000902
INFO:tensorflow:2018-06-30 18:55:44.622251: Step 1970: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:45.230836: Step 1980: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:45.231033: Step 1980: Cross entropy = 0.000619
INFO:tensorflow:2018-06-30 18:55:45.299670: Step 1980: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:45.911100: Step 1990: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:45.911287: Step 1990: Cross entropy = 0.000611
INFO:tensorflow:2018-06-30 18:55:45.977344: Step 1990: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:46.590329: Step 2000: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:46.590534: Step 2000: Cross entropy = 0.000717
INFO:tensorflow:2018-06-30 18:55:46.650610: Step 2000: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:47.333739: Step 2010: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:47.333933: Step 2010: Cross entropy = 0.000583
INFO:tensorflow:2018-06-30 18:55:47.395989: Step 2010: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:48.006149: Step 2020: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:48.006370: Step 2020: Cross entropy = 0.000967
INFO:tensorflow:2018-06-30 18:55:48.067016: Step 2020: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:48.701178: Step 2030: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:48.701385: Step 2030: Cross entropy = 0.001147
INFO:tensorflow:2018-06-30 18:55:48.766095: Step 2030: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:49.377611: Step 2040: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:49.377808: Step 2040: Cross entropy = 0.000640
INFO:tensorflow:2018-06-30 18:55:49.439127: Step 2040: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:50.057545: Step 2050: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:50.057753: Step 2050: Cross entropy = 0.000713
INFO:tensorflow:2018-06-30 18:55:50.122882: Step 2050: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:50.743576: Step 2060: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:50.743767: Step 2060: Cross entropy = 0.000830
INFO:tensorflow:2018-06-30 18:55:50.807204: Step 2060: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:51.435073: Step 2070: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:51.435270: Step 2070: Cross entropy = 0.000675
INFO:tensorflow:2018-06-30 18:55:51.495681: Step 2070: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:52.103578: Step 2080: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:52.103782: Step 2080: Cross entropy = 0.000698
INFO:tensorflow:2018-06-30 18:55:52.166394: Step 2080: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:52.793214: Step 2090: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:52.793415: Step 2090: Cross entropy = 0.000675
INFO:tensorflow:2018-06-30 18:55:52.857564: Step 2090: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:53.477860: Step 2100: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:53.478049: Step 2100: Cross entropy = 0.000744
INFO:tensorflow:2018-06-30 18:55:53.536358: Step 2100: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:54.150742: Step 2110: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:54.150949: Step 2110: Cross entropy = 0.000615
INFO:tensorflow:2018-06-30 18:55:54.213652: Step 2110: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:54.805382: Step 2120: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:54.805573: Step 2120: Cross entropy = 0.000978
INFO:tensorflow:2018-06-30 18:55:54.871409: Step 2120: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:55.493766: Step 2130: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:55.493958: Step 2130: Cross entropy = 0.000649
INFO:tensorflow:2018-06-30 18:55:55.554015: Step 2130: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:56.182163: Step 2140: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:56.182367: Step 2140: Cross entropy = 0.000896
INFO:tensorflow:2018-06-30 18:55:56.245492: Step 2140: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:56.857483: Step 2150: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:56.857684: Step 2150: Cross entropy = 0.001033
INFO:tensorflow:2018-06-30 18:55:56.922555: Step 2150: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:57.541535: Step 2160: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:57.541738: Step 2160: Cross entropy = 0.000650
INFO:tensorflow:2018-06-30 18:55:57.602366: Step 2160: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:58.209542: Step 2170: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:58.209747: Step 2170: Cross entropy = 0.000811
INFO:tensorflow:2018-06-30 18:55:58.273746: Step 2170: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:58.896472: Step 2180: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:58.896667: Step 2180: Cross entropy = 0.000788
INFO:tensorflow:2018-06-30 18:55:58.958949: Step 2180: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:55:59.586306: Step 2190: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:55:59.586499: Step 2190: Cross entropy = 0.000616
INFO:tensorflow:2018-06-30 18:55:59.646614: Step 2190: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:00.293664: Step 2200: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:00.293935: Step 2200: Cross entropy = 0.000769
INFO:tensorflow:2018-06-30 18:56:00.386494: Step 2200: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:01.054727: Step 2210: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:01.054932: Step 2210: Cross entropy = 0.000397
INFO:tensorflow:2018-06-30 18:56:01.117900: Step 2210: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:01.746048: Step 2220: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:01.746269: Step 2220: Cross entropy = 0.000504
INFO:tensorflow:2018-06-30 18:56:01.808942: Step 2220: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:02.428459: Step 2230: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:02.428654: Step 2230: Cross entropy = 0.000473
INFO:tensorflow:2018-06-30 18:56:02.492965: Step 2230: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:03.100415: Step 2240: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:03.100624: Step 2240: Cross entropy = 0.000825
INFO:tensorflow:2018-06-30 18:56:03.165840: Step 2240: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:03.782598: Step 2250: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:03.782789: Step 2250: Cross entropy = 0.000505
INFO:tensorflow:2018-06-30 18:56:03.845735: Step 2250: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:04.535851: Step 2260: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:04.536054: Step 2260: Cross entropy = 0.000595
INFO:tensorflow:2018-06-30 18:56:04.599561: Step 2260: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:05.224605: Step 2270: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:05.224824: Step 2270: Cross entropy = 0.000676
INFO:tensorflow:2018-06-30 18:56:05.298990: Step 2270: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:05.951923: Step 2280: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:05.952116: Step 2280: Cross entropy = 0.000674
INFO:tensorflow:2018-06-30 18:56:06.015503: Step 2280: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:06.629230: Step 2290: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:06.629437: Step 2290: Cross entropy = 0.000501
INFO:tensorflow:2018-06-30 18:56:06.691488: Step 2290: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:07.418946: Step 2300: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:07.419130: Step 2300: Cross entropy = 0.000799
INFO:tensorflow:2018-06-30 18:56:07.483704: Step 2300: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:08.401134: Step 2310: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:08.401364: Step 2310: Cross entropy = 0.000879
INFO:tensorflow:2018-06-30 18:56:08.476442: Step 2310: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:09.403309: Step 2320: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:09.403703: Step 2320: Cross entropy = 0.000706
INFO:tensorflow:2018-06-30 18:56:09.529761: Step 2320: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:10.370529: Step 2330: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:10.370747: Step 2330: Cross entropy = 0.000667
INFO:tensorflow:2018-06-30 18:56:10.452157: Step 2330: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:11.409949: Step 2340: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:11.410165: Step 2340: Cross entropy = 0.000904
INFO:tensorflow:2018-06-30 18:56:11.530701: Step 2340: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:12.454697: Step 2350: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:12.454896: Step 2350: Cross entropy = 0.000770
INFO:tensorflow:2018-06-30 18:56:12.518100: Step 2350: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:13.155075: Step 2360: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:13.155380: Step 2360: Cross entropy = 0.000561
INFO:tensorflow:2018-06-30 18:56:13.217214: Step 2360: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:15.815532: Step 2370: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:15.815741: Step 2370: Cross entropy = 0.000612
INFO:tensorflow:2018-06-30 18:56:15.884711: Step 2370: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:16.582018: Step 2380: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:16.582222: Step 2380: Cross entropy = 0.000539
INFO:tensorflow:2018-06-30 18:56:16.649706: Step 2380: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:17.740149: Step 2390: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:17.740351: Step 2390: Cross entropy = 0.000842
INFO:tensorflow:2018-06-30 18:56:17.818075: Step 2390: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:18.597197: Step 2400: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:18.597410: Step 2400: Cross entropy = 0.000538
INFO:tensorflow:2018-06-30 18:56:18.689706: Step 2400: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:19.423854: Step 2410: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:19.424054: Step 2410: Cross entropy = 0.000696
INFO:tensorflow:2018-06-30 18:56:19.490295: Step 2410: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:20.173623: Step 2420: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:20.173835: Step 2420: Cross entropy = 0.000307
INFO:tensorflow:2018-06-30 18:56:20.235845: Step 2420: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:20.905178: Step 2430: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:20.905388: Step 2430: Cross entropy = 0.000497
INFO:tensorflow:2018-06-30 18:56:20.979904: Step 2430: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:21.699389: Step 2440: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:21.699609: Step 2440: Cross entropy = 0.000834
INFO:tensorflow:2018-06-30 18:56:21.766214: Step 2440: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:22.402234: Step 2450: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:22.402457: Step 2450: Cross entropy = 0.000563
INFO:tensorflow:2018-06-30 18:56:22.465189: Step 2450: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:23.171939: Step 2460: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:23.172135: Step 2460: Cross entropy = 0.000700
INFO:tensorflow:2018-06-30 18:56:23.235222: Step 2460: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:23.918722: Step 2470: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:23.918912: Step 2470: Cross entropy = 0.000735
INFO:tensorflow:2018-06-30 18:56:23.982685: Step 2470: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:24.680609: Step 2480: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:24.680810: Step 2480: Cross entropy = 0.000730
INFO:tensorflow:2018-06-30 18:56:24.744425: Step 2480: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:25.393331: Step 2490: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:25.393523: Step 2490: Cross entropy = 0.000597
INFO:tensorflow:2018-06-30 18:56:25.458297: Step 2490: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:26.115282: Step 2500: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:26.115482: Step 2500: Cross entropy = 0.000586
INFO:tensorflow:2018-06-30 18:56:26.184283: Step 2500: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:26.868722: Step 2510: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:26.868949: Step 2510: Cross entropy = 0.000524
INFO:tensorflow:2018-06-30 18:56:26.938476: Step 2510: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:27.581376: Step 2520: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:27.581577: Step 2520: Cross entropy = 0.000560
INFO:tensorflow:2018-06-30 18:56:27.650242: Step 2520: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:28.286409: Step 2530: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:28.286612: Step 2530: Cross entropy = 0.000843
INFO:tensorflow:2018-06-30 18:56:28.347300: Step 2530: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:29.004278: Step 2540: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:29.004488: Step 2540: Cross entropy = 0.000567
INFO:tensorflow:2018-06-30 18:56:29.065594: Step 2540: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:29.709530: Step 2550: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:29.709724: Step 2550: Cross entropy = 0.000429
INFO:tensorflow:2018-06-30 18:56:29.773586: Step 2550: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:30.428781: Step 2560: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:30.428985: Step 2560: Cross entropy = 0.000452
INFO:tensorflow:2018-06-30 18:56:30.494410: Step 2560: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:31.157641: Step 2570: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:31.157835: Step 2570: Cross entropy = 0.000638
INFO:tensorflow:2018-06-30 18:56:31.223631: Step 2570: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:31.903168: Step 2580: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:31.903360: Step 2580: Cross entropy = 0.000667
INFO:tensorflow:2018-06-30 18:56:31.970816: Step 2580: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:32.637505: Step 2590: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:32.637698: Step 2590: Cross entropy = 0.000505
INFO:tensorflow:2018-06-30 18:56:32.704879: Step 2590: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:33.346721: Step 2600: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:33.347058: Step 2600: Cross entropy = 0.000680
INFO:tensorflow:2018-06-30 18:56:33.412509: Step 2600: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:34.067913: Step 2610: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:34.068116: Step 2610: Cross entropy = 0.000557
INFO:tensorflow:2018-06-30 18:56:34.129283: Step 2610: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:34.790830: Step 2620: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:34.791098: Step 2620: Cross entropy = 0.000521
INFO:tensorflow:2018-06-30 18:56:34.860614: Step 2620: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:35.506986: Step 2630: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:35.507183: Step 2630: Cross entropy = 0.000471
INFO:tensorflow:2018-06-30 18:56:35.569538: Step 2630: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:36.222740: Step 2640: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:36.222941: Step 2640: Cross entropy = 0.000725
INFO:tensorflow:2018-06-30 18:56:36.290182: Step 2640: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:36.971186: Step 2650: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:36.971436: Step 2650: Cross entropy = 0.000472
INFO:tensorflow:2018-06-30 18:56:37.033622: Step 2650: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:37.691215: Step 2660: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:37.691415: Step 2660: Cross entropy = 0.000726
INFO:tensorflow:2018-06-30 18:56:37.757358: Step 2660: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:38.435746: Step 2670: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:38.435934: Step 2670: Cross entropy = 0.000610
INFO:tensorflow:2018-06-30 18:56:38.501614: Step 2670: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:39.161506: Step 2680: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:39.161701: Step 2680: Cross entropy = 0.000393
INFO:tensorflow:2018-06-30 18:56:39.227641: Step 2680: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:39.889171: Step 2690: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:39.889395: Step 2690: Cross entropy = 0.000567
INFO:tensorflow:2018-06-30 18:56:39.968420: Step 2690: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:40.629202: Step 2700: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:40.629522: Step 2700: Cross entropy = 0.000669
INFO:tensorflow:2018-06-30 18:56:40.691650: Step 2700: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:41.362487: Step 2710: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:41.362691: Step 2710: Cross entropy = 0.000621
INFO:tensorflow:2018-06-30 18:56:41.430247: Step 2710: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:42.095351: Step 2720: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:42.095545: Step 2720: Cross entropy = 0.000674
INFO:tensorflow:2018-06-30 18:56:42.158655: Step 2720: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:42.862333: Step 2730: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:42.862546: Step 2730: Cross entropy = 0.000443
INFO:tensorflow:2018-06-30 18:56:42.937958: Step 2730: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:44.861817: Step 2740: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:44.862048: Step 2740: Cross entropy = 0.000714
INFO:tensorflow:2018-06-30 18:56:44.932577: Step 2740: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:45.881564: Step 2750: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:45.881781: Step 2750: Cross entropy = 0.000789
INFO:tensorflow:2018-06-30 18:56:45.948422: Step 2750: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:46.596220: Step 2760: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:46.596427: Step 2760: Cross entropy = 0.000378
INFO:tensorflow:2018-06-30 18:56:46.659419: Step 2760: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:48.532847: Step 2770: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:48.533054: Step 2770: Cross entropy = 0.000663
INFO:tensorflow:2018-06-30 18:56:48.598203: Step 2770: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:49.326252: Step 2780: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:49.326453: Step 2780: Cross entropy = 0.000589
INFO:tensorflow:2018-06-30 18:56:49.390191: Step 2780: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:50.032300: Step 2790: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:50.032498: Step 2790: Cross entropy = 0.000466
INFO:tensorflow:2018-06-30 18:56:50.101771: Step 2790: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:50.762177: Step 2800: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:50.762381: Step 2800: Cross entropy = 0.000637
INFO:tensorflow:2018-06-30 18:56:50.828497: Step 2800: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:51.603636: Step 2810: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:51.603994: Step 2810: Cross entropy = 0.000493
INFO:tensorflow:2018-06-30 18:56:51.666226: Step 2810: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:52.312783: Step 2820: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:52.312975: Step 2820: Cross entropy = 0.000570
INFO:tensorflow:2018-06-30 18:56:52.376247: Step 2820: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:53.028816: Step 2830: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:53.029006: Step 2830: Cross entropy = 0.000700
INFO:tensorflow:2018-06-30 18:56:53.097261: Step 2830: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:53.895411: Step 2840: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:53.895618: Step 2840: Cross entropy = 0.000512
INFO:tensorflow:2018-06-30 18:56:53.966514: Step 2840: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:54.646951: Step 2850: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:54.647195: Step 2850: Cross entropy = 0.000507
INFO:tensorflow:2018-06-30 18:56:54.714686: Step 2850: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:55.440536: Step 2860: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:55.440734: Step 2860: Cross entropy = 0.000462
INFO:tensorflow:2018-06-30 18:56:55.509054: Step 2860: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:56.162649: Step 2870: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:56.162859: Step 2870: Cross entropy = 0.000819
INFO:tensorflow:2018-06-30 18:56:56.229978: Step 2870: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:56.895480: Step 2880: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:56.895682: Step 2880: Cross entropy = 0.000602
INFO:tensorflow:2018-06-30 18:56:56.963979: Step 2880: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:57.607939: Step 2890: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:57.608140: Step 2890: Cross entropy = 0.000534
INFO:tensorflow:2018-06-30 18:56:57.672164: Step 2890: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:58.322301: Step 2900: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:58.322500: Step 2900: Cross entropy = 0.000597
INFO:tensorflow:2018-06-30 18:56:58.385788: Step 2900: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:59.045359: Step 2910: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:59.045552: Step 2910: Cross entropy = 0.000573
INFO:tensorflow:2018-06-30 18:56:59.122175: Step 2910: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:56:59.786525: Step 2920: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:56:59.786730: Step 2920: Cross entropy = 0.000517
INFO:tensorflow:2018-06-30 18:56:59.851233: Step 2920: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:00.520888: Step 2930: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:00.521092: Step 2930: Cross entropy = 0.000460
INFO:tensorflow:2018-06-30 18:57:00.589324: Step 2930: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:01.240416: Step 2940: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:01.240640: Step 2940: Cross entropy = 0.000483
INFO:tensorflow:2018-06-30 18:57:01.307713: Step 2940: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:01.947919: Step 2950: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:01.948113: Step 2950: Cross entropy = 0.000510
INFO:tensorflow:2018-06-30 18:57:02.015877: Step 2950: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:02.676107: Step 2960: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:02.676301: Step 2960: Cross entropy = 0.000408
INFO:tensorflow:2018-06-30 18:57:02.747447: Step 2960: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:03.409515: Step 2970: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:03.409853: Step 2970: Cross entropy = 0.000513
INFO:tensorflow:2018-06-30 18:57:03.485226: Step 2970: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:04.294069: Step 2980: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:04.294279: Step 2980: Cross entropy = 0.000524
INFO:tensorflow:2018-06-30 18:57:04.365611: Step 2980: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:05.010360: Step 2990: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:05.010557: Step 2990: Cross entropy = 0.000592
INFO:tensorflow:2018-06-30 18:57:05.075045: Step 2990: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:05.755983: Step 3000: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:05.756168: Step 3000: Cross entropy = 0.000689
INFO:tensorflow:2018-06-30 18:57:05.819733: Step 3000: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:06.480556: Step 3010: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:06.480760: Step 3010: Cross entropy = 0.000534
INFO:tensorflow:2018-06-30 18:57:06.546045: Step 3010: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:07.215382: Step 3020: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:07.215582: Step 3020: Cross entropy = 0.000531
INFO:tensorflow:2018-06-30 18:57:07.283118: Step 3020: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:07.947948: Step 3030: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:07.948187: Step 3030: Cross entropy = 0.000385
INFO:tensorflow:2018-06-30 18:57:08.017608: Step 3030: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:08.680673: Step 3040: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:08.680905: Step 3040: Cross entropy = 0.000576
INFO:tensorflow:2018-06-30 18:57:08.751799: Step 3040: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:09.423676: Step 3050: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:09.423895: Step 3050: Cross entropy = 0.000329
INFO:tensorflow:2018-06-30 18:57:09.500632: Step 3050: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:10.332154: Step 3060: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:10.332348: Step 3060: Cross entropy = 0.000449
INFO:tensorflow:2018-06-30 18:57:10.399504: Step 3060: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:11.402356: Step 3070: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:11.402548: Step 3070: Cross entropy = 0.000474
INFO:tensorflow:2018-06-30 18:57:11.462481: Step 3070: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:12.152463: Step 3080: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:12.152654: Step 3080: Cross entropy = 0.000608
INFO:tensorflow:2018-06-30 18:57:12.219748: Step 3080: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:12.880322: Step 3090: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:12.880525: Step 3090: Cross entropy = 0.000627
INFO:tensorflow:2018-06-30 18:57:12.980192: Step 3090: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:13.656281: Step 3100: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:13.656476: Step 3100: Cross entropy = 0.000540
INFO:tensorflow:2018-06-30 18:57:13.720599: Step 3100: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:16.946122: Step 3110: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:16.946318: Step 3110: Cross entropy = 0.000555
INFO:tensorflow:2018-06-30 18:57:17.025770: Step 3110: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:19.693128: Step 3120: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:19.693322: Step 3120: Cross entropy = 0.000560
INFO:tensorflow:2018-06-30 18:57:19.757739: Step 3120: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:20.477478: Step 3130: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:20.477684: Step 3130: Cross entropy = 0.000495
INFO:tensorflow:2018-06-30 18:57:20.548321: Step 3130: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:23.000254: Step 3140: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:23.000650: Step 3140: Cross entropy = 0.000806
INFO:tensorflow:2018-06-30 18:57:23.066374: Step 3140: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:23.741907: Step 3150: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:23.742114: Step 3150: Cross entropy = 0.000515
INFO:tensorflow:2018-06-30 18:57:23.808135: Step 3150: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:24.481264: Step 3160: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:24.481457: Step 3160: Cross entropy = 0.000381
INFO:tensorflow:2018-06-30 18:57:24.549189: Step 3160: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:25.214333: Step 3170: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:25.214538: Step 3170: Cross entropy = 0.000629
INFO:tensorflow:2018-06-30 18:57:25.284702: Step 3170: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:26.002826: Step 3180: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:26.003033: Step 3180: Cross entropy = 0.000311
INFO:tensorflow:2018-06-30 18:57:26.068875: Step 3180: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:26.869676: Step 3190: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:26.869897: Step 3190: Cross entropy = 0.000440
INFO:tensorflow:2018-06-30 18:57:26.934288: Step 3190: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:27.621629: Step 3200: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:27.621891: Step 3200: Cross entropy = 0.000311
INFO:tensorflow:2018-06-30 18:57:27.709076: Step 3200: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:28.370800: Step 3210: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:28.370990: Step 3210: Cross entropy = 0.000550
INFO:tensorflow:2018-06-30 18:57:28.437489: Step 3210: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:29.110924: Step 3220: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:29.111115: Step 3220: Cross entropy = 0.000609
INFO:tensorflow:2018-06-30 18:57:29.175806: Step 3220: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:29.843423: Step 3230: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:29.843615: Step 3230: Cross entropy = 0.000631
INFO:tensorflow:2018-06-30 18:57:29.910970: Step 3230: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:30.645026: Step 3240: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:30.645354: Step 3240: Cross entropy = 0.000447
INFO:tensorflow:2018-06-30 18:57:30.713560: Step 3240: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:31.410864: Step 3250: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:31.411071: Step 3250: Cross entropy = 0.000596
INFO:tensorflow:2018-06-30 18:57:31.477643: Step 3250: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:32.164453: Step 3260: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:32.164725: Step 3260: Cross entropy = 0.000410
INFO:tensorflow:2018-06-30 18:57:32.250344: Step 3260: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:32.969673: Step 3270: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:32.969861: Step 3270: Cross entropy = 0.000534
INFO:tensorflow:2018-06-30 18:57:33.051398: Step 3270: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:33.734773: Step 3280: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:33.734974: Step 3280: Cross entropy = 0.000470
INFO:tensorflow:2018-06-30 18:57:33.818033: Step 3280: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:34.514374: Step 3290: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:34.514573: Step 3290: Cross entropy = 0.000367
INFO:tensorflow:2018-06-30 18:57:34.581381: Step 3290: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:35.237031: Step 3300: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:35.237224: Step 3300: Cross entropy = 0.000467
INFO:tensorflow:2018-06-30 18:57:35.327704: Step 3300: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:36.005256: Step 3310: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:36.005459: Step 3310: Cross entropy = 0.000546
INFO:tensorflow:2018-06-30 18:57:36.084610: Step 3310: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:36.777169: Step 3320: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:36.777365: Step 3320: Cross entropy = 0.000697
INFO:tensorflow:2018-06-30 18:57:36.850287: Step 3320: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:37.529725: Step 3330: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:37.529937: Step 3330: Cross entropy = 0.000486
INFO:tensorflow:2018-06-30 18:57:37.596653: Step 3330: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:38.278344: Step 3340: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:38.278549: Step 3340: Cross entropy = 0.000589
INFO:tensorflow:2018-06-30 18:57:38.351548: Step 3340: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:39.030214: Step 3350: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:39.030409: Step 3350: Cross entropy = 0.000488
INFO:tensorflow:2018-06-30 18:57:39.099829: Step 3350: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:39.772339: Step 3360: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:39.772529: Step 3360: Cross entropy = 0.000445
INFO:tensorflow:2018-06-30 18:57:39.838911: Step 3360: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:40.506571: Step 3370: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:40.506766: Step 3370: Cross entropy = 0.000431
INFO:tensorflow:2018-06-30 18:57:40.599971: Step 3370: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:41.355714: Step 3380: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:41.355908: Step 3380: Cross entropy = 0.000445
INFO:tensorflow:2018-06-30 18:57:41.419266: Step 3380: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:42.121773: Step 3390: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:42.121979: Step 3390: Cross entropy = 0.000287
INFO:tensorflow:2018-06-30 18:57:42.186401: Step 3390: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:42.854949: Step 3400: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:42.855142: Step 3400: Cross entropy = 0.000421
INFO:tensorflow:2018-06-30 18:57:42.918703: Step 3400: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:46.459506: Step 3410: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:46.459710: Step 3410: Cross entropy = 0.000446
INFO:tensorflow:2018-06-30 18:57:46.531483: Step 3410: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:51.123727: Step 3420: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:51.123933: Step 3420: Cross entropy = 0.000627
INFO:tensorflow:2018-06-30 18:57:51.198249: Step 3420: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:51.917484: Step 3430: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:51.917680: Step 3430: Cross entropy = 0.000599
INFO:tensorflow:2018-06-30 18:57:51.985023: Step 3430: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:52.622720: Step 3440: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:52.622914: Step 3440: Cross entropy = 0.000414
INFO:tensorflow:2018-06-30 18:57:52.685702: Step 3440: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:55.709443: Step 3450: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:55.709665: Step 3450: Cross entropy = 0.000462
INFO:tensorflow:2018-06-30 18:57:55.781261: Step 3450: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:56.434138: Step 3460: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:56.434433: Step 3460: Cross entropy = 0.000445
INFO:tensorflow:2018-06-30 18:57:56.501106: Step 3460: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:57.291111: Step 3470: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:57.291294: Step 3470: Cross entropy = 0.000355
INFO:tensorflow:2018-06-30 18:57:57.389060: Step 3470: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:58.086207: Step 3480: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:58.086403: Step 3480: Cross entropy = 0.000518
INFO:tensorflow:2018-06-30 18:57:58.150237: Step 3480: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:58.816294: Step 3490: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:58.816636: Step 3490: Cross entropy = 0.000362
INFO:tensorflow:2018-06-30 18:57:58.906231: Step 3490: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:57:59.590677: Step 3500: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:57:59.590862: Step 3500: Cross entropy = 0.000504
INFO:tensorflow:2018-06-30 18:57:59.660337: Step 3500: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:00.319708: Step 3510: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:00.320014: Step 3510: Cross entropy = 0.000543
INFO:tensorflow:2018-06-30 18:58:00.384301: Step 3510: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:01.044321: Step 3520: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:01.044527: Step 3520: Cross entropy = 0.000449
INFO:tensorflow:2018-06-30 18:58:01.108547: Step 3520: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:01.792542: Step 3530: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:01.792743: Step 3530: Cross entropy = 0.000487
INFO:tensorflow:2018-06-30 18:58:01.859343: Step 3530: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:02.677008: Step 3540: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:02.677204: Step 3540: Cross entropy = 0.000471
INFO:tensorflow:2018-06-30 18:58:02.748100: Step 3540: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:03.396435: Step 3550: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:03.396636: Step 3550: Cross entropy = 0.000397
INFO:tensorflow:2018-06-30 18:58:03.462803: Step 3550: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:04.140221: Step 3560: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:04.140415: Step 3560: Cross entropy = 0.000465
INFO:tensorflow:2018-06-30 18:58:04.202551: Step 3560: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:04.852294: Step 3570: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:04.852483: Step 3570: Cross entropy = 0.000484
INFO:tensorflow:2018-06-30 18:58:04.920493: Step 3570: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:12.703545: Step 3580: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:12.703752: Step 3580: Cross entropy = 0.000256
INFO:tensorflow:2018-06-30 18:58:12.766357: Step 3580: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:14.182924: Step 3590: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:14.183176: Step 3590: Cross entropy = 0.000401
INFO:tensorflow:2018-06-30 18:58:14.248907: Step 3590: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:15.208443: Step 3600: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:15.208664: Step 3600: Cross entropy = 0.000604
INFO:tensorflow:2018-06-30 18:58:15.272518: Step 3600: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:15.928368: Step 3610: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:15.928562: Step 3610: Cross entropy = 0.000445
INFO:tensorflow:2018-06-30 18:58:15.996905: Step 3610: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:16.648351: Step 3620: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:16.648543: Step 3620: Cross entropy = 0.000445
INFO:tensorflow:2018-06-30 18:58:16.845963: Step 3620: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:18.514342: Step 3630: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:18.514572: Step 3630: Cross entropy = 0.000352
INFO:tensorflow:2018-06-30 18:58:18.577342: Step 3630: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:19.234586: Step 3640: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:19.234788: Step 3640: Cross entropy = 0.000350
INFO:tensorflow:2018-06-30 18:58:19.301392: Step 3640: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:20.585336: Step 3650: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:20.585543: Step 3650: Cross entropy = 0.000370
INFO:tensorflow:2018-06-30 18:58:20.652234: Step 3650: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:21.306238: Step 3660: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:21.306431: Step 3660: Cross entropy = 0.000527
INFO:tensorflow:2018-06-30 18:58:21.372678: Step 3660: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:22.029864: Step 3670: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:22.030064: Step 3670: Cross entropy = 0.000505
INFO:tensorflow:2018-06-30 18:58:22.102688: Step 3670: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:22.756149: Step 3680: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:22.756405: Step 3680: Cross entropy = 0.000201
INFO:tensorflow:2018-06-30 18:58:22.820516: Step 3680: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:23.471484: Step 3690: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:23.471678: Step 3690: Cross entropy = 0.000391
INFO:tensorflow:2018-06-30 18:58:23.539628: Step 3690: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:24.202779: Step 3700: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:24.203184: Step 3700: Cross entropy = 0.000379
INFO:tensorflow:2018-06-30 18:58:24.265756: Step 3700: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:24.907653: Step 3710: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:24.907851: Step 3710: Cross entropy = 0.000471
INFO:tensorflow:2018-06-30 18:58:24.972609: Step 3710: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:25.634256: Step 3720: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:25.634447: Step 3720: Cross entropy = 0.000207
INFO:tensorflow:2018-06-30 18:58:25.697210: Step 3720: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:26.404998: Step 3730: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:26.405201: Step 3730: Cross entropy = 0.000305
INFO:tensorflow:2018-06-30 18:58:26.469257: Step 3730: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:27.343180: Step 3740: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:27.343389: Step 3740: Cross entropy = 0.000571
INFO:tensorflow:2018-06-30 18:58:27.422130: Step 3740: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:28.088221: Step 3750: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:28.088426: Step 3750: Cross entropy = 0.000506
INFO:tensorflow:2018-06-30 18:58:28.161725: Step 3750: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:28.814960: Step 3760: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:28.815161: Step 3760: Cross entropy = 0.000531
INFO:tensorflow:2018-06-30 18:58:28.882014: Step 3760: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:29.508497: Step 3770: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:29.508710: Step 3770: Cross entropy = 0.000391
INFO:tensorflow:2018-06-30 18:58:29.574885: Step 3770: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:30.209263: Step 3780: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:30.209451: Step 3780: Cross entropy = 0.000384
INFO:tensorflow:2018-06-30 18:58:30.274042: Step 3780: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:30.914613: Step 3790: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:30.914823: Step 3790: Cross entropy = 0.000405
INFO:tensorflow:2018-06-30 18:58:30.985566: Step 3790: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:31.607463: Step 3800: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:31.607687: Step 3800: Cross entropy = 0.000429
INFO:tensorflow:2018-06-30 18:58:31.671312: Step 3800: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:32.314591: Step 3810: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:32.314787: Step 3810: Cross entropy = 0.000263
INFO:tensorflow:2018-06-30 18:58:32.376456: Step 3810: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:33.020238: Step 3820: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:33.020442: Step 3820: Cross entropy = 0.000381
INFO:tensorflow:2018-06-30 18:58:33.085817: Step 3820: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:33.802865: Step 3830: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:33.803045: Step 3830: Cross entropy = 0.000441
INFO:tensorflow:2018-06-30 18:58:33.881046: Step 3830: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:34.512331: Step 3840: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:34.512547: Step 3840: Cross entropy = 0.000334
INFO:tensorflow:2018-06-30 18:58:34.572717: Step 3840: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:35.254777: Step 3850: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:35.254973: Step 3850: Cross entropy = 0.000547
INFO:tensorflow:2018-06-30 18:58:35.318834: Step 3850: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:36.003780: Step 3860: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:36.003979: Step 3860: Cross entropy = 0.000470
INFO:tensorflow:2018-06-30 18:58:36.066621: Step 3860: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:36.692918: Step 3870: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:36.693129: Step 3870: Cross entropy = 0.000451
INFO:tensorflow:2018-06-30 18:58:36.756377: Step 3870: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:37.389864: Step 3880: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:37.390054: Step 3880: Cross entropy = 0.000397
INFO:tensorflow:2018-06-30 18:58:37.462782: Step 3880: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:38.081132: Step 3890: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:38.081324: Step 3890: Cross entropy = 0.000330
INFO:tensorflow:2018-06-30 18:58:38.146596: Step 3890: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:38.773816: Step 3900: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:38.774019: Step 3900: Cross entropy = 0.000449
INFO:tensorflow:2018-06-30 18:58:38.833622: Step 3900: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:39.476689: Step 3910: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:39.476878: Step 3910: Cross entropy = 0.000202
INFO:tensorflow:2018-06-30 18:58:39.540160: Step 3910: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:40.177978: Step 3920: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:40.178224: Step 3920: Cross entropy = 0.000417
INFO:tensorflow:2018-06-30 18:58:40.238206: Step 3920: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:40.890371: Step 3930: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:40.890571: Step 3930: Cross entropy = 0.000386
INFO:tensorflow:2018-06-30 18:58:40.957640: Step 3930: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:41.597353: Step 3940: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:41.597544: Step 3940: Cross entropy = 0.000449
INFO:tensorflow:2018-06-30 18:58:41.657547: Step 3940: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:42.387800: Step 3950: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:42.387997: Step 3950: Cross entropy = 0.000339
INFO:tensorflow:2018-06-30 18:58:42.450447: Step 3950: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:43.108569: Step 3960: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:43.108774: Step 3960: Cross entropy = 0.000366
INFO:tensorflow:2018-06-30 18:58:43.172828: Step 3960: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:46.976251: Step 3970: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:46.976453: Step 3970: Cross entropy = 0.000446
INFO:tensorflow:2018-06-30 18:58:47.043257: Step 3970: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:51.183289: Step 3980: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:51.183488: Step 3980: Cross entropy = 0.000340
INFO:tensorflow:2018-06-30 18:58:51.246083: Step 3980: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:51.877327: Step 3990: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:51.877529: Step 3990: Cross entropy = 0.000321
INFO:tensorflow:2018-06-30 18:58:51.943013: Step 3990: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:58:52.531434: Step 3999: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:58:52.531635: Step 3999: Cross entropy = 0.000385
INFO:tensorflow:2018-06-30 18:58:52.597795: Step 3999: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:Final test accuracy = 93.5% (N=31) #Unser Trainingsgenauigkeit liegt bei 93.5%
INFO:tensorflow:Froze 2 variables.
Converted 2 variables to const ops.
# 4000 trainingssteps wurden durchlaufen
# Es wird angegeben zu wie viel Prozent die verschiedenen Bilder erkannt wurden
#Der Trainingsdurchlauf ist beendet
#Das neuronale Netzwerk wurde mit verschiedenen Bildern aus allen Bereichen trainiert

#Ein einzelnes Bild kann nun klassifiziert werden 


# Ein bestimmtes Bild wird eingelesen zur Analyse
/Users/manjangoc/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
python -m scripts.label_image \
>     --graph=tf_files/retrained_graph.pb  \
>     --image=tf_files/KI/Menschen/60.jpg #Bild das klassifiziert werden soll 
# Klassifizierung des Bildes

Evaluation time (1-image): 1.845s
# Zeit zum klassifizieren des Bildes

menschen (score=1.00000)
tiere (score=0.00000)
flecken (score=0.00000)
text (score=0.00000)
# Das Bild ist zu 100% der Kategorie Menschen zuzuordnen

# Informationen zur Ausgabe

# Um unsere Ausgabe zu visualisieren haben wir eine App zur Objekterkennung an unser Projekt angepasst  
# Die App wird in Android-Studio erstellt
# Mit diesem Programm werden unsere Datein zur KI geöfftnet 
# Nach dem Testlauf wurden die "Default" Graphen mit unserem "retrained_graph.pb" ausgetauscht. Und die txt Datei wurde mit einer Datei mit unseren Labels ersetzt, also Mensch, Tier, Text und Flecken (retrained_labels.txt aus dem Training).
#In der Classifier.Activity.java Datei musste der Input und der Output auf "input" und "final_result" gesetzt werden
#Anschließend wurde die Build.Gradle Datei erneut geöffnet und eine neue APK Datei erstellt. Diese APK Datei wurde erneut auf das Android Gerät übertragen, geöffnet und installiert.
# Nun kann die App, die auf unseren Trainingsdaten beruht, auf einem Handy ausgeführt werden 



