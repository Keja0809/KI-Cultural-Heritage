Projekt Vorstellung 

Idee: 
- Analyse und Kategorisierung der bereitgestellten Bilder
- Kategorien: Mensch, Tier, Text, Andere(Flecken)
- Ein Bild wird ins Programm eingelesen und den Bereichen prozentual zugeordnet
- Ausgabe: Das Programm gibt auf Grundlage des Trainings eine Einschätzung, zu wie viel Prozent das auf dem Bild zu erkennende Objekt den Kategorien entspricht
- Jedes Bild kann eingelesen werden und das Programm erkennt, ob ein Zusammenhang zu den Trainingsbildern besteht 


Technologie: Wir nutzen Tensorflow als Programm und arbeiten nach dem Prinzip von Tensorflow for Poets 2 

Code: 

// Zunächst wird das git repository von Tensorflow for poets runtergeladen
Manjas-MacBook-Pro:~ manjangoc$ git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
Cloning into 'tensorflow-for-poets-2'...
remote: Counting objects: 405, done.
remote: Total 405 (delta 0), reused 0 (delta 0), pack-reused 405
Receiving objects: 100% (405/405), 33.96 MiB | 7.08 MiB/s, done.
Resolving deltas: 100% (149/149), done.
Checking out files: 100% (142/142), done.

//Der Ordner wird aufgerufen, damit man anschließend dadrin arbeiten kann. 
Manjas-MacBook-Pro:~ manjangoc$ cd tensorflow-for-poets-2

// Aufrufen der einzelnen Ordner // Kategorien zum Analysieren
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ ls tf_files/KI
Flecken   Menschen  Text    Tiere

// TensorBoard wird aufgerufen, damit graphische Analyse im Browser angezeigt werden kann.
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ tensorboard --logdir tf_files/training_summaries &
[1] 8595

// Phyton wird aufgerufen zum trainieren
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ python -m scripts.retrain -h
/Users/manjangoc/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
/Users/manjangoc/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/KI
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

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Path to folders of labeled images.
  --output_graph OUTPUT_GRAPH
                        Where to save the trained graph.
  --intermediate_output_graphs_dir INTERMEDIATE_OUTPUT_GRAPHS_DIR
                        Where to save the intermediate graphs.
  --intermediate_store_frequency INTERMEDIATE_STORE_FREQUENCY
                        How many steps to store intermediate graph. If "0"
                        then will not store.
  --output_labels OUTPUT_LABELS
                        Where to save the trained graph's labels.
  --summaries_dir SUMMARIES_DIR
                        Where to save summary logs for TensorBoard.
  --how_many_training_steps HOW_MANY_TRAINING_STEPS
                        How many training steps to run before ending.
  --learning_rate LEARNING_RATE
                        How large a learning rate to use when training.
  --testing_percentage TESTING_PERCENTAGE
                        What percentage of images to use as a test set.
  --validation_percentage VALIDATION_PERCENTAGE
                        What percentage of images to use as a validation set.
  --eval_step_interval EVAL_STEP_INTERVAL
                        How often to evaluate the training results.
  --train_batch_size TRAIN_BATCH_SIZE
                        How many images to train on at a time.
  --test_batch_size TEST_BATCH_SIZE
                        How many images to test on. This test set is only used
                        once, to evaluate the final accuracy of the model
                        after training completes. A value of -1 causes the
                        entire test set to be used, which leads to more stable
                        results across runs.
  --validation_batch_size VALIDATION_BATCH_SIZE
                        How many images to use in an evaluation batch. This
                        validation set is used much more often than the test
                        set, and is an early indicator of how accurate the
                        model is during training. A value of -1 causes the
                        entire validation set to be used, which leads to more
                        stable results across training iterations, but may be
                        slower on large training sets.
  --print_misclassified_test_images
                        Whether to print out a list of all misclassified test
                        images.
  --model_dir MODEL_DIR
                        Path to classify_image_graph_def.pb,
                        imagenet_synset_to_human_label_map.txt, and
                        imagenet_2012_challenge_label_map_proto.pbtxt.
  --bottleneck_dir BOTTLENECK_DIR
                        Path to cache bottleneck layer values as files.
  --final_tensor_name FINAL_TENSOR_NAME
                        The name of the output classification layer in the
                        retrained graph.
  --flip_left_right     Whether to randomly flip half of the training images
                        horizontally.
  --random_crop RANDOM_CROP
                        A percentage determining how much of a margin to
                        randomly crop off the training images.
  --random_scale RANDOM_SCALE
                        A percentage determining how much to randomly scale up
                        the size of the training images by.
  --random_brightness RANDOM_BRIGHTNESS
                        A percentage determining how much to randomly multiply
                        the training image input pixels up or down by.
  --architecture ARCHITECTURE
                        Which model architecture to use. 'inception_v3' is the
                        most accurate, but also the slowest. For faster or
                        smaller models, chose a MobileNet with the form
                        'mobilenet_<parameter size>_<input_size>[_quantized]'.
                        For example, 'mobilenet_1.0_224' will pick a model
                        that is 17 MB in size and takes 224 pixel input
                        images, while 'mobilenet_0.25_128_quantized' will
                        choose a much less accurate, but smaller and faster
                        network that's 920 KB on disk and takes 128x128
                        images. See https://research.googleblog.com/2017/06
                        /mobilenets-open-source-models-for.html for more
                        information on Mobilenet.

// Spezifischer Code für Training
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ python -m scripts.retrain \
>   --bottleneck_dir=tf_files/bottlenecks \                                        // Kategorien werden erstellt
>   --how_many_training_steps=500 \                                                // Anzahl der Trainingsstufen
>   --model_dir=tf_files/models/ \
>   --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \                
>   --output_graph=tf_files/retrained_graph.pb \
>   --output_labels=tf_files/retrained_labels.txt \
>   --architecture="${ARCHITECTURE}" \
>   --image_dir=tf_files/KI                                                       //Ordner, in denen die Kategorien sind

// Jede einzelne Datei in dem Ordner mit Kategorien wird eingelesen und analysiert
TensorBoard 1.7.0 at http://Manjas-MacBook-Pro.local:6006 (Press CTRL+C to quit)
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
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/12_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/14_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/16_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/14_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/8_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/8_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/14_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/30_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/8_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/12_6.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/34_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/12_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/10_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/8_5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/8_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/34_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/10_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/17_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/28.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/9_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/33_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/28_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/33_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/9_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/17.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/13_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/28_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/13_5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/15_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/9_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/17_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/16.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/33_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/35_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/15_6.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/13.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/35_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/13_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/13_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/15_5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/10.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/21.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/35.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/20_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/1_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/20.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/22.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/5_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/20_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/1_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/5_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/24_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/1_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/19_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/18.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/24.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/30.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/5_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/24_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/7_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/5_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/19.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/2_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/7.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/6_5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/6_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/6.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/27_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/4_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/18_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/4_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/6_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/27_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/27_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/4_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/18_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/18_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/6_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/12_5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/12_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/15_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/28_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/12.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/13_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/15_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/20_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/27.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/7_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/7_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/6_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/16_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/12_3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/10_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/34_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/14.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/17_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:100 bottleneck files created.
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/33_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/35_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/1_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/7_4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/34.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/33.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/1_5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/19_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Flecken/2_1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/63.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0050 Kopie.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/62.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/60.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0168.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/61.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/59.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/65.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/64.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/58.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/8.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0027.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/9.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/14.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0069.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0054.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/29.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/15.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/17.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0056.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0057.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0043.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/16.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/12.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0052.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0046.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/13.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/11.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0124.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0044.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0045.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/10.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/38.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/35.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0074.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0048.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0049.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/34.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/20.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/36.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/22.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0117.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0063.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/23.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/37.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/33.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/27.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0072.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/26.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/32.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/24.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/30.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0071.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0065.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0070.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0058.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/31.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/25.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/19.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/42.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/56.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0163.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0177.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/5.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/57.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/43.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/7.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/41.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0028.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0029.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/40.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/54.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/6.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/50.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/44.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0165.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0164.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/45.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/3.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/47.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0166.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0058 Kopie.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/49.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0047.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/39.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0050.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0064.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/4.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/53.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:200 bottleneck files created.
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0172.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0167.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0142.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0156.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/48.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/28.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0053.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0057 Kopie.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/BOOK-0824731-0051.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/21.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/18.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/55.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/51.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/52.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Menschen/46.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Seitenname.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Seitenzahl.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Literaturverzeichnis.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Objektbeschreibung.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Diagramm.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Überschrift.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Bildbeschreibung.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Fußnote.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Text/Textauschnitt1.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0142.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0156.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0157.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0169.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0109.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0054.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0097.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0108.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0134.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0120.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0056.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0043.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0123.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0137.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0127.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0047.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0053.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0052.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0085.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0087.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0050.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0044.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0092.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0100.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0114.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0128.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0062.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0116.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0112.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0106.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0072.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0113.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0139.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0050_2.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0059.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0058.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0138.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0149.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0161.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0164.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0146.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0122.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0118.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0051.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0075.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0073.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0163.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0148.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0140.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0096.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0084.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0079.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0119.jpg_mobilenet_0.50_224.txt
INFO:tensorflow:Creating bottleneck at tf_files/bottlenecks/Tiere/BOOK-0824731-0060.jpg_mobilenet_0.50_224.txt
WARNING:tensorflow:From /Users/manjangoc/tensorflow-for-poets-2/scripts/retrain.py:790: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.

// Trainingsschritte zu wie viel Prozent etwas erkannt wird. 

INFO:tensorflow:2018-06-30 18:40:20.015326: Step 0: Train accuracy = 87.0%
INFO:tensorflow:2018-06-30 18:40:20.015570: Step 0: Cross entropy = 0.376785
INFO:tensorflow:2018-06-30 18:40:21.390891: Step 0: Validation accuracy = 83.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:27.275231: Step 10: Train accuracy = 98.0%
INFO:tensorflow:2018-06-30 18:40:27.275424: Step 10: Cross entropy = 0.055775
INFO:tensorflow:2018-06-30 18:40:27.670974: Step 10: Validation accuracy = 98.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:29.167675: Step 20: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:29.167931: Step 20: Cross entropy = 0.027123
INFO:tensorflow:2018-06-30 18:40:29.238386: Step 20: Validation accuracy = 97.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:29.917027: Step 30: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:29.917320: Step 30: Cross entropy = 0.021062
INFO:tensorflow:2018-06-30 18:40:29.988464: Step 30: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:30.662631: Step 40: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:30.662822: Step 40: Cross entropy = 0.027739
INFO:tensorflow:2018-06-30 18:40:30.725751: Step 40: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:31.381689: Step 50: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:31.381890: Step 50: Cross entropy = 0.016651
INFO:tensorflow:2018-06-30 18:40:31.446200: Step 50: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:32.107445: Step 60: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:32.107634: Step 60: Cross entropy = 0.015323
INFO:tensorflow:2018-06-30 18:40:32.170373: Step 60: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:32.835979: Step 70: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:32.836172: Step 70: Cross entropy = 0.016579
INFO:tensorflow:2018-06-30 18:40:32.910835: Step 70: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:33.682082: Step 80: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:33.682281: Step 80: Cross entropy = 0.011151
INFO:tensorflow:2018-06-30 18:40:33.756999: Step 80: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:34.433466: Step 90: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:34.433665: Step 90: Cross entropy = 0.008042
INFO:tensorflow:2018-06-30 18:40:34.498004: Step 90: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:35.176713: Step 100: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:35.176914: Step 100: Cross entropy = 0.010080
INFO:tensorflow:2018-06-30 18:40:35.245079: Step 100: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:35.931211: Step 110: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:35.931407: Step 110: Cross entropy = 0.013815
INFO:tensorflow:2018-06-30 18:40:35.999019: Step 110: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:36.669129: Step 120: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:36.669315: Step 120: Cross entropy = 0.008527
INFO:tensorflow:2018-06-30 18:40:36.740692: Step 120: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:37.431356: Step 130: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:37.431544: Step 130: Cross entropy = 0.011390
INFO:tensorflow:2018-06-30 18:40:37.506180: Step 130: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:38.226764: Step 140: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:38.227058: Step 140: Cross entropy = 0.006238
INFO:tensorflow:2018-06-30 18:40:38.293276: Step 140: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:38.994808: Step 150: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:38.995209: Step 150: Cross entropy = 0.007960
INFO:tensorflow:2018-06-30 18:40:39.063383: Step 150: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:40.416418: Step 160: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:40.416683: Step 160: Cross entropy = 0.008914
INFO:tensorflow:2018-06-30 18:40:40.486448: Step 160: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:47.739714: Step 170: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:47.739914: Step 170: Cross entropy = 0.009257
INFO:tensorflow:2018-06-30 18:40:47.812077: Step 170: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:49.956690: Step 180: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:49.956885: Step 180: Cross entropy = 0.005113
INFO:tensorflow:2018-06-30 18:40:50.068829: Step 180: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:50.817654: Step 190: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:50.817858: Step 190: Cross entropy = 0.007037
INFO:tensorflow:2018-06-30 18:40:50.887521: Step 190: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:52.073298: Step 200: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:52.073486: Step 200: Cross entropy = 0.004352
INFO:tensorflow:2018-06-30 18:40:52.141493: Step 200: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:53.411194: Step 210: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:53.411422: Step 210: Cross entropy = 0.005154
INFO:tensorflow:2018-06-30 18:40:53.473972: Step 210: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:54.196594: Step 220: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:54.196822: Step 220: Cross entropy = 0.006646
INFO:tensorflow:2018-06-30 18:40:54.259644: Step 220: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:55.872487: Step 230: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:55.872675: Step 230: Cross entropy = 0.004645
INFO:tensorflow:2018-06-30 18:40:55.942281: Step 230: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:56.912234: Step 240: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:56.912427: Step 240: Cross entropy = 0.005343
INFO:tensorflow:2018-06-30 18:40:56.972462: Step 240: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:57.622281: Step 250: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:57.622462: Step 250: Cross entropy = 0.003866
INFO:tensorflow:2018-06-30 18:40:57.682142: Step 250: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:58.368907: Step 260: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:58.369107: Step 260: Cross entropy = 0.005045
INFO:tensorflow:2018-06-30 18:40:58.433688: Step 260: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:59.084993: Step 270: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:59.085207: Step 270: Cross entropy = 0.004786
INFO:tensorflow:2018-06-30 18:40:59.153764: Step 270: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:40:59.789662: Step 280: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:40:59.789859: Step 280: Cross entropy = 0.004582
INFO:tensorflow:2018-06-30 18:40:59.851830: Step 280: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:00.529621: Step 290: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:00.529808: Step 290: Cross entropy = 0.004204
INFO:tensorflow:2018-06-30 18:41:00.604856: Step 290: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:01.286087: Step 300: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:01.286270: Step 300: Cross entropy = 0.005187
INFO:tensorflow:2018-06-30 18:41:01.351089: Step 300: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:02.012916: Step 310: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:02.013121: Step 310: Cross entropy = 0.003378
INFO:tensorflow:2018-06-30 18:41:02.078082: Step 310: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:02.768516: Step 320: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:02.768825: Step 320: Cross entropy = 0.004517
INFO:tensorflow:2018-06-30 18:41:02.836337: Step 320: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:03.544509: Step 330: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:03.544696: Step 330: Cross entropy = 0.003564
INFO:tensorflow:2018-06-30 18:41:03.606167: Step 330: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:04.305239: Step 340: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:04.305425: Step 340: Cross entropy = 0.003780
INFO:tensorflow:2018-06-30 18:41:04.385620: Step 340: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:05.108359: Step 350: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:05.108552: Step 350: Cross entropy = 0.004653
INFO:tensorflow:2018-06-30 18:41:05.184960: Step 350: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:06.040681: Step 360: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:06.040874: Step 360: Cross entropy = 0.003837
INFO:tensorflow:2018-06-30 18:41:06.112852: Step 360: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:06.878471: Step 370: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:06.878770: Step 370: Cross entropy = 0.003875
INFO:tensorflow:2018-06-30 18:41:06.950194: Step 370: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:07.638119: Step 380: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:07.638411: Step 380: Cross entropy = 0.003834
INFO:tensorflow:2018-06-30 18:41:07.706038: Step 380: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:08.390165: Step 390: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:08.390425: Step 390: Cross entropy = 0.003517
INFO:tensorflow:2018-06-30 18:41:08.459186: Step 390: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:09.339012: Step 400: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:09.339209: Step 400: Cross entropy = 0.003533
INFO:tensorflow:2018-06-30 18:41:09.402558: Step 400: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:17.458660: Step 410: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:17.458846: Step 410: Cross entropy = 0.003898
INFO:tensorflow:2018-06-30 18:41:17.521170: Step 410: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:22.914809: Step 420: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:22.915006: Step 420: Cross entropy = 0.003540
INFO:tensorflow:2018-06-30 18:41:22.981010: Step 420: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:23.641196: Step 430: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:23.641382: Step 430: Cross entropy = 0.002779
INFO:tensorflow:2018-06-30 18:41:23.700913: Step 430: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:24.353306: Step 440: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:24.353592: Step 440: Cross entropy = 0.004508
INFO:tensorflow:2018-06-30 18:41:24.428591: Step 440: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:25.073284: Step 450: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:25.073489: Step 450: Cross entropy = 0.003252
INFO:tensorflow:2018-06-30 18:41:25.147701: Step 450: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:25.817370: Step 460: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:25.817561: Step 460: Cross entropy = 0.003014
INFO:tensorflow:2018-06-30 18:41:25.881169: Step 460: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:26.629681: Step 470: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:26.629877: Step 470: Cross entropy = 0.002814
INFO:tensorflow:2018-06-30 18:41:26.689713: Step 470: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:27.502182: Step 480: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:27.502389: Step 480: Cross entropy = 0.002686
INFO:tensorflow:2018-06-30 18:41:27.567117: Step 480: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:28.239312: Step 490: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:28.239507: Step 490: Cross entropy = 0.003117
INFO:tensorflow:2018-06-30 18:41:28.306309: Step 490: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:2018-06-30 18:41:31.761762: Step 499: Train accuracy = 100.0%
INFO:tensorflow:2018-06-30 18:41:31.761961: Step 499: Cross entropy = 0.002341
INFO:tensorflow:2018-06-30 18:41:31.828510: Step 499: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:Final test accuracy = 96.8% (N=31)
INFO:tensorflow:Froze 2 variables.
Converted 2 variables to const ops.

// Code zum Abrufen vom Bild
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ python -m scripts.label_image \
>     --graph=tf_files/retrained_graph.pb  \
>     --image=tf_files/KI/Menschen/1.jpg

/Users/manjangoc/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
python -m scripts.label_image \
>     --graph=tf_files/retrained_graph.pb  \
>     --image=tf_files/KI/Menschen/60.jpg

Evaluation time (1-image): 1.845s // Dauer der Analyse
 
// Anzeige zu wie viel Prozent das Bild die Kategorie ist. 
menschen (score=1.00000) // Bild ist 100% ein Mensch
tiere (score=0.00000)
flecken (score=0.00000)
text (score=0.00000)

