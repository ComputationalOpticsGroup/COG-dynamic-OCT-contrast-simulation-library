# Dynamic-optical-coherence-tomography simulatior library by computational optics group (COG DOCT-simulator library) 


The COG DOCT-simulator library is a Python library which provides the classes and functions to simulate a time-sequence of OCT speckle with the modeling of the scatterer motions.

About authors
--------------
This library is released by [Computational Optics Group at the University of Tsukuba](https://cog-news.blogspot.com/), and was developed as a part of our research projects of computationally augmented optical coherence microscope and optical coherence tomography.

Basement theory
---------------------------
Some theoretical and scientifical bouckgorunds of this libaray is available at the following references.
1. Y.K. Feng et al., "Dynamic OCT simulator based on mathematical models of intratissue dynamics, image formation, and measurement noise
", Proc. SPIE 13305, Optical Coherence Tomography and Coherence Domain Optical Methods in Biomedicine XXIX, (2025) [PDF](documents\SPIE_Proceeding_2025.pdf).
2. Y.K. Feng et al., "Mathematical modeling of intracellular and intratissue activities for understanding dynamic optical coherence tomography signals", Proc. SPIE 12830, Optical Coherence Tomography and Coherence Domain Optical Methods in Biomedicine XXVIII, 128300H (12 March 2024) [PDF](documents\SPIE_Proceeding_2024.pdf).

Scatterer motion models are as follows
- Random ballistic model  
- Diffsusion model  
- Mono-direction traslation model

Manuals
------------------------
- Example code to generate the 32-time time-sequence of OCT speckle considering the scatterer motion model as the random ballistic model using this library can be found [here](example.py).
- Module design can be found [here](documents\Module_design.docx).
- More detailed manuals to be available in near future.

License
-----------------------
This library is licensed under either of the following options.
1. [GNU lesser general public license version 3 (LGPLv3)](LICENSE_GnuLGPLv3.md).
2. Any licenses except for GNU LGPLv3 as far as the authors and the uses agree for it. It may include business licenses, closed-source licenses, and others. 
 
Option 1 (GNU LGPLv3) can be selected without notifying the authors. (However, it is recommended to cite this web-site and the proper reference listed in "Basement theory" sections when you publish research papers using this library.)
If you want to select any other licensing conditions except for GNU LGPLv3 (i.e., Option 2), please contact the corresponding author (Yoshiaki Yasuno, University of Tsukuba, <yoshiaki.yasuno@cog-labs.org>) to obtain an explicit agreement.
