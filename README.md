# Music Generation Using Deep Learning
This project involves the implementation of a music generating model trained by a Long Short-Term Memory artificial neural network. The final model generates music which displays characteristics common in that of the folk music genre:

* **monophonic**, a simple melody without any accompanying harmony or chords
* of the **AABB form** (binary form) which represents the repetition of a melody *A* and a melody *B*
* **ends on the tonic**, if the key of a song is in G, the song should end with a G note

Music in general is the result of a sequence of musical notes played and organised within a length of time. In this way, music is structured and sequential, much like natural language, where musical notes are words and measures are sentences.
The final generator is a result of data preparation, data processing, and neural network training, all implemented in Python.

## The Data

The dataset that was used for this project was the [Nottingham Database](http://abc.sourceforge.net/NMD/) which containes 1200 monophonic British and American folk tunes all written in [**abc notation**](http://abcnotation.com/). `nottingham_data_prep.py` prepares the data by removing all comments and meta-data but the meter *M*, the key *K*, and the tune (excluding the chords which are written between quotation marks). An example of a prepared tuned is as follows

    M:4/4
    K:G
    f|: || |:g3/2a/2 gf|gd2e|dc BA|BG2A|BG2A/2B/2|\
    cA2B/2c/2|Bd cB| [1Ad ef:|[2A2 A2||
    BG2A/2B/2|cA dB|ec2d/2e/2|fzde/2f/2|
    g3/2a/2 ge|df gB|ce d/2e/2d/2c/2|BG G2:|

All the tunes are saved in a single file: `nottingham_parsed.txt` in the `datasets/nottingham_database/` directory.