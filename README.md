## Update - Nikon Instrument Software (NIS) Compatability
The software can now store 16bit Tiff images of the separated fluorescences of 2.8um and 4.5um beads in folders NIS28 and NIS45 respectively, after calibration or prediction.
This way users can seperate the beads with the same fluorescence into separate fluorecence images for each bead size and input each image to measure fluorescence using NIS. 
For more information, please see the NIS Fluorescence Measurement Section in the User Manual.

# Multiplexed ELISA on a Bead Assays 
### Author and Developer: Jaden Sequeira


Microscopy based multiplexed assays are an important tool for comparing stem cell
derived beta cells and donor beta cells through their secretion profiles. The limited fluorescent
range used by microscopes limits multiplexed assays to the detection of 3 to 4 four biomolecules.
This is because fluorescent labels emit a range of fluorescent wavelengths and overlapping
ranges lead to false detections. The proposed approach to overcome this limitation is to apply a
machine learning model to segment different bead sizes which contain different bioreceptors.
After biomolecules bind to the bioreceptors, they can be fluorescently labeled. The fluorescence
intensity of each biomolecule can then be calculated using the segmented masks of the different
bead sizes. Finally, the mean fluorescence intensities can be converted into concentrations for
each biomolecule using standard curves.


![alt text](Images/Image1.JPG)




## Software Overview
MultiELISAB folder holds the software for multiplexed bead ELISA measurements.\
Tutorials folder holds links (google drive links) to the 3 video tutorials.\
Documentation folder holds the User Manual and Overview Document.\
Model folder holds an example U-Net Model file that was trained for 2.8um and 4.5um bead segmentation.
