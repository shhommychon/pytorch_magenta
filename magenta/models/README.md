# Models

This directory contains Magenta models.

* [**Coconet**](/magenta/models/coconet): Counterpoint by Convolution train a convolutional neural network to complete partial musical scores.
* [**Drums RNN**](/magenta/models/drums_rnn): Applies language modeling to drum track generation using an LSTM.
* [**GANSynth**](/magenta/models/gansynth): GANSynth is an algorithm for synthesizing audio with generative adversarial networks.
* [**Improv RNN**](/magenta/models/improv_rnn): Generates melodies a la [Melody RNN](/magenta/models/melody_rnn), but conditions the melodies on an underlying chord progression.
* [**Melody RNN**](/magenta/models/melody_rnn): Applies language modeling to melody generation using an LSTM.
* [**Music VAE**](/magenta/models/music_vae): A hierarchical recurrent variational autoencoder for music.
* [**NSynth**](/magenta/models/nsynth): "Neural Audio Synthesis" as described in [*NSynth: Neural Audio Synthesis with WaveNet Autoencoders*](https://arxiv.org/abs/1704.01279).
* [**Onsets and Frames**](/magenta/models/onsets_frames_transcription): Automatic piano music transcription model as described in [*Onsets and Frames: Dual-Objective Piano Transcription*](https://arxiv.org/abs/1710.11153)
* [**Performance RNN**](/magenta/models/performance_rnn): Applies language modeling to polyphonic music using a combination of note on/off, timeshift, and velocity change events.
* [**Piano Genie**](/magenta/models/piano_genie): Piano Genie is a system for learning a low-dimensional discrete representation of piano music. It uses an encoder RNN to compress piano sequences (88 keys) into many fewer buttons (e.g. 8). A decoder RNN is responsible for converting the simpler sequences back to piano space.
* [**Pianoroll RNN-NADE**](/magenta/models/pianoroll_rnn_nade): Applies language modeling to polyphonic music generation using an LSTM combined with a NADE, an architecture called an RNN-NADE. Based on the architecture described in [*Modeling Temporal Dependencies in High-Dimensional Sequences:
Application to Polyphonic Music Generation and Transcription*](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf).
* [**Polyphony RNN**](/magenta/models/polyphony_rnn): Applies language modeling to polyphonic music generation using an LSTM. Based on the [BachBot](https://github.com/feynmanliang/bachbot/) architecture described in [*Automatic Stylistic Composition of Bach Choralies with Deep LSTM*](https://www.microsoft.com/en-us/research/publication/automatic-stylistic-composition-of-bach-chorales-with-deep-lstm/).
* [**RL Tuner**](/magenta/models/rl_tuner): Takes an LSTM that has been trained to predict the next note in a monophonic melody and enhances it using reinforcement learning (RL). Described in [*Tuning Recurrent Neural Networks with Reinforcement Learning*](https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/) and [*Sequence Tutor: Conservative Fine-Tuning of Sequence Generation Models with KL-control*](https://arxiv.org/abs/1611.02796)
* [**Score2Perf and Music Transformer**](/magenta/models/score2perf): Score2Perf is a collection of Tensor2Tensor problems for generating musical performances, either unconditioned or conditioned on a musical score.
