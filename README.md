## Directory Structure

ua_vision_training/
 * base_models/
 * data/
   * [insert git submodule for labelimg software]
   * images.jpg
   * annotations.xml
 * tools/
   * [LabelImg](https://github.com/tzutalin/labelImg)
 * training/
 * configuration.yaml
 * prepare_training.py
 * run_training.py
 * cleanup.py
 
 ## Input data
 
 Each training example shall be represented by a JPEG image and a
 corresponding XML file containing annotations.  The XML schema shall 
 abide by the format standardized by the PASCAL Visual Object Classes (
 VOC) datasets.  Fortunately, there are plenty of open source tools 
 available for creating image annotations in this format.  For example, 
 this repository includes a Git submodule (inside the "tools" directory) 
 linking to a tool called 
 [LabelImg](https://github.com/tzutalin/labelImg)
 that can be used to create annotations in the correct format.
 
 Image and annotation files shall be placed directly inside of the
 "data" subdirectory.

