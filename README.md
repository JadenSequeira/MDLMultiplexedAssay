## Update - Nikon Instrument Software (NIS) Compatability
The software can now store 16bit Tiff images of the separated fluorescences of 2.8um and 4.5um beads in folders NIS28 and NIS45 respectively, after calibration or prediction.
This way users can seperate the beads with the same fluorescence into separate fluorecence images for each bead size and input each image to measure fluorescence using NIS. 
For more information, please see the NIS Fluorescence Measurement Section in the User Manual.

# Multiplexed ELISA on a Bead Assays 
### Developed by Jaden Sequeira


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

Pancreatic beta cells are pancreatic cells that secrete insulin – an important biomolecule
for body processes such as managing blood glucose levels through cellular glucose uptake. Type
1 Diabetes (T1D) is an autoimmune disease where pancreatic cells are destroyed by the immune
system, thus resulting in less insulin production. Current treatments include lifelong insulin injections,
and more recently, pancreatic beta cell transplants. Donor beta cells are in short supply and thus there is
a critical need for new beta cells sources. Recent research has focussed on the differentiation 
and genetic engineering of pluripotent stem cells into stem cell derived beta cells (SCβ) to be used as a new source for beta cell transplants. 

![alt text](Images/Image1.JPG)


Recently, nanowell technology and fluorescent microscopy was used to investigate
the heterogeneity of glucose stimulated insulin secretion (GSIS) for single pancreatic beta cells.
This study provided insight into the characteristics required of stem cell derived beta cells before
they can be used as a transplant treatment for Type 1 Diabetes. However, there is still a need for
better characterization of donor and SCβ cells regarding their insulin, glucagon, and somatostatin
hormone secretion levels. This will help in understanding if SCβ cells provide the same secretion
standards of donor cells, or if alternative methods such as secretion-based cell
selection if needed.

![alt text](Images/Image2.JPG)

Multiplexed single cell secretion assays can be used to profile donor and stem cell
derived islet cells through their secretion of insulin, glucagon, and somatostatin. This can be
done at the single cell level using nanowells to separate the cells and microscopy to image the
fluorescence of detection beads in each nanowell. These detection beads increase in fluorescence
intensity when the biomolecule of interest increases (e.g. insulin). However, multiplexed single
cell assays that employ microscopy are usually limited to detecting 3-4 different biomolecules. 
This is due to the limited fluorescence emission range and the fact that fluorophores attached to
the beads emit a range of fluorescent wavelengths. Generally, when conducting single cell
assays, two fluorescence stains are used to check if the cell is dead or alive. As a result, there is
not enough space on the fluorescence emission range for an additional three fluorophores for
insulin, glucagon, and somatostatin.


## Software Overview
MultiELISAB folder holds the software for multiplexed bead ELISA measurements.\
Tutorials folder holds links (google drive links) to the 3 video tutorials.\
Documentation folder holds the User Manual and Overview Document.\
Model folder holds an example U-Net Model file that was trained for 2.8um and 4.5um bead segmentation.
