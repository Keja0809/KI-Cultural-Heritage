Projekt Vorstellung 

Idee: 
- Analyse und Kategorisierung der bereitgestellten Bilder
- Kategorien: Mensch, Tier, Text, Andere(Flecken)
- Ein Bild wird ins Programm eingelesen und den Bereichen prozentual zugeordnet
- Ausgabe: Das Programm gibt auf Grundlage des Trainings eine Einschätzung, zu wie viel Prozent das auf dem Bild zu erkennende Objekt den Kategorien entspricht
- Jedes Bild kann eingelesen werden und das Programm erkennt, ob ein Zusammenhang zu den Trainingsbildern besteht 


Technologie: Wir nutzen Tensorflow als Programm und arbeiten nach dem Prinzip von Tensorflow for Poets 2 

Code Eingabe: 

Zunächst wird das git repository von Tensorflow for poets runtergeladen
> Manjas-MacBook-Pro:~ manjangoc$ git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
> Cloning into 'tensorflow-for-poets-2'...
> remote: Counting objects: 405, done.
> remote: Total 405 (delta 0), reused 0 (delta 0), pack-reused 405
> Receiving objects: 100% (405/405), 33.96 MiB | 7.08 MiB/s, done.
> Resolving deltas: 100% (149/149), done.
> Checking out files: 100% (142/142), done.

Der Ordner wird aufgerufen, damit man anschließend dadrin arbeiten kann. 
> Manjas-MacBook-Pro:~ manjangoc$ cd tensorflow-for-poets-2

Aufrufen der einzelnen Ordner // Kategorien zum Analysieren
> Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ ls tf_files/KI
> Flecken   Menschen  Text    Tiere

TensorBoard wird aufgerufen, damit graphische Analyse im Browser angezeigt werden kann.
> Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ tensorboard --logdir tf_files/training_summaries &
> [1] 8595

Retrain skript wird mit phyton abgerufen -> TensorFlow Hub repo
> Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ python -m scripts.retrain -h

Training der verschiedenen Klassen
1. Skript wird abgerufen
2. Kategorien werden erstellt

> Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ 
> python -m scripts.retrain \                                                        
>   --bottleneck_dir=tf_files/bottlenecks \                                        
>   --how_many_training_steps=500 \                                                
>   --model_dir=tf_files/models/ \                                              
>   --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \                
>   --output_graph=tf_files/retrained_graph.pb \
>   --output_labels=tf_files/retrained_labels.txt \
>   --architecture="${ARCHITECTURE}" \
>   --image_dir=tf_files/KI                                                     

Testen der Klassifizierung, indem Bild abgerufen wird
Manjas-MacBook-Pro:tensorflow-for-poets-2 manjangoc$ python -m scripts.label_image \
>     --graph=tf_files/retrained_graph.pb  \
>     --image=tf_files/KI/Menschen/1.jpg                                        

// Dauer der Analyse
> Evaluation time (1-image): 1.845s 
 
Ausgabe Ergebnis, zu viel viel Prozent, das Bild der Kategorie entspricht -> Bild zeigt 100% Mensch an
> menschen (score=1.00000) 
> tiere (score=0.00000)
> flecken (score=0.00000)
> text (score=0.00000)


Erstellung einer Visuellen Ausgabe:

Grundlage: retrained_graph.pb

Verwendetes Programm: 
Android Studio


Testlauf mit den Graphen welche bereits im Download Ordner Tensorflow for poets 2 enthalten sind.

1. Öffnen der Build.gradle Datei
2. Sync der Gradle Datei
3. Build APK
4. Erstellte APK Datei auf ein Android Gerät übertragen
5. Auf dem Android Gerät Datei ausführen und installieren
6. Öffnen; App funktioniert

Nach dem Testlauf wurden die "Default" Graphen mit unserem "retrained_graph.pb" ausgetauscht. Und die txt Datei wurde mit einer Datei mit unseren Labels ersetzt, also Mensch, Tier, Text und Flecken.

> cp tf_files/rounded_graph.pb android/tfmobile/assets/graph.pb
> cp tf_files/retrained_labels.txt android/tfmobile/assets/labels.txt


In der Classifier.Activity.java Datei musste der Input und der Output auf "input" und "final_result" gesetzt werden. Dies war notwendig, da die Default Graphen andere Input und Output Names haben. Unsere haben wir über das Tensorboard direkt von unserem Graphen abgelesen. 
Das Tauschen der Input und Outpur Names wurde mit diesem Befehl durchgeführt (kann aber auch manuell erfolgen):

> private static final String INPUT_NAME = "input";
> private static final String OUTPUT_NAME = "final_result";


Anschließend wurde die Build.Gradle Datei erneut geöffnet und eine neue APK Datei erstellt. 
Diese APK Datei wurde erneut auf das Android Gerät übertragen, geöffnet und installiert.

Nun Läuft die App auf unserem Gerät mit den Labels Mensch, Tier, Flecken, Text und erkennt die entsprechenden Höhlenmalereien. 



