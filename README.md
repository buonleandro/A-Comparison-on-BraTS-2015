# A-Comparison-on-BraTS-2015

Questo lavoro mira a comparare tre reti per la segmentazione di immagini 2d di tumori cerebrali. Il dataset usato è scaricabile a: https://academictorrents.com/details/c4f39a0a8e46e8d2174b8a8a81b9887150f44d50

## Dataset

Scaricato il dataset, estrarre le due cartelle contenute
Modificare la riga 4 del file Data_to_File.py, indicando il percorso della cartella contenente i file mha da estrarre
Specificare un nome della cartella che conterrà i file estratti (es. se chiamiamo la cartella "folder", le immagini si troveranno in ./folder)
A seconda che i file corrispondano a MRI o a maschere, commentare le righe corrispondenti (es. se stiamo estraendo MRI andranno commentate le righe 17 e 19, altrimenti le righe 13-16 e 18)
Eseguire Data_to_File
Modificare il file File_to_Numpy.py, indicando: il numero di immagini che il file per il dataset dovrà contenere (riga 5), il persorso delle immagini (riga 6) e il nome del file contenente il set (riga 7) (sostituire 'mask'a 'mri' nel caso si stia costruendo un train set per le maschere)

## Addestramento

Configurare l'addestramento tramite le righe 6-10 del file Train.py specificando il numero di epoche, la dimensione del batch e i dettagli sul nome della cartella del modello salvato a ogni checkpoint
Definire i persorsi dei train set e validation set (righe 16-17)
De-commentare una tra le reti che si vuole addestrare tra le righe 12, 13 e 14
Eseguire Train.py
