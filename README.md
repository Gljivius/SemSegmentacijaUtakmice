# SemSegmentacijaUtakmice
Semantička segmentacija objekata na nogometnim utakmicama napravljena u Tensorflowu

Ovo je Python program koji služi za semantičku semgentaciju slika, u ovom slučaju objekata na nogometnim utakmicama. Napisan je uz pomoć biblioteke Tensorflow i sastoji se od potpuno konvolucijske neuronske mreže koja koristi modificirane težine VGG16 mreže.

<b>Upute za instalaciju potrebnih stvari</b>

Prva stvar koja je potrebna je Python 3.5 ili više. Također je potrebno još par dodatnih biblioteka, a najlakši način za sve to nabaviti odjednom je instalacija Anaconde (https://www.continuum.io/downloads). Ako slučajno nedostaje koja biblioteka "pip" vam je prijatelj :)

Drugo što je potrebno je biblioteka Tensorflow. Upute za njezinu instalaciju nalaze se na sljedećoj stranici: https://www.tensorflow.org/install/ Odaberite željeni operacijski sustav i snimite prema uputama koje su tamo. Moja preporuka je to instalirati prema uputama za Anacondu. Ako imate Nvidijin GPU novije generacije preporuka je instalirati verziju za GPU jer se dobici u brzini treniranja modela i izvođenju ogromni.

Kad se sve to instalira to bi trebalo biti to :)

Program je isproban i testiran na verziji Pythona 3.5, Tensoflowu 1.0 te na operacijskom sustavu Linux. Za prebacivanje na Windows bi trebalo samo napraviti malo prepravku sa načinom dohvata datoteka i to bi trebalo biti to. Za MacOS se sami snađite :)
