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
1. Skript wird angeufen
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

> Evaluation time (1-image): 1.845s // Dauer der Analyse
 
Ergebnis, zu viel viel Prozent, das Bild der Kategorie entspricht -> Bild zeigt 100% Mensch an
> menschen (score=1.00000) 
> tiere (score=0.00000)
> flecken (score=0.00000)
> text (score=0.00000)

