# Shadow Free Images with Bi-Illuminate Dichromatic Reflection Model

### **Nelson Farrell & Michael Massone**  
**Date:** 10.24.2024  
**Course:** CS 7180 Advanced Perception - Prof. Bruce Maxwell, PhD  
**Time Travel Days:** 5  

---

## **Project Overview**

The goal of this project is to address image shadow removal using the physics of illumination. This work builds upon prior research by Maxwell et al., who applied the **Bi-Illuminate Dichromatic Reflection (BIDR)** model for shadow removal. The method estimates the **Illumination Spectral Direction (ISD)** of an image in log space and projects the log-space pixels onto a plane orthogonal to the ISD. This projection generates a new **log chromaticity space**, which is invariant to the image's illumination, resulting in shadow-free images.

Our contributions to this problem are twofold:

1. **Data Annotation in the Spectral Dataset:**
   - Identify lit and shadowed pixel pairs in images for training.

2. **Develop a Conversion Program:**  
   - Transform 16-bit `.tiff` image files into **log-space chromaticity** images.

---

## **Operating System**

- **MacOS** on Apple M1 chip  

---

## **Annotations**

We followed annotation guidelines created by fellow students to complete pixel annotations for the dataset. For detailed documentation, visit the following link:  
[Annotation Guidelines Documentation](https://northeastern-my.sharepoint.com/:w:/g/personal/liu_chang31_northeastern_edu/EenBIDDZGH1JsPHz4Y00se0Bu2qQBE93aaZSsiSaRRT69w?e=LMiQ7I&xsdata=MDV8MDJ8bWFzc29uZS5tQG5vcnRoZWFzdGVybi5lZHV8M2VjNmU2MjZhOTZiNDQzNzYxYTkwOGRjZjIyN2VjOTd8YThlZWMyODFhYWEzNGRhZWFjOWI5YTM5OGI5MjE1ZTd8MHwwfDYzODY1MTUwMDQ5MjcxNTUxN3xVbmtub3dufFRXRnBiR1pzYjNkOGV5SldJam9pTUM0d0xqQXdNREFpTENKUUlqb2lWMmx1TXpJaUxDSkJUaUk2SWsxaGFXd2lMQ0pYVkNJNk1uMD18MHx8fA%3d%3d&sdata=aHlxSnp3R3FaeEgzcnlYUmp1aGVadDJGajgyTkZpYkZZRDZVRExPRVV6Yz0%3d)

---

## **Instructions**

1. **Download** a folder of images from the Spectral Dataset directory on Discovery using the instructions in the documentation linked above.
2. **Save** the folder in the working directory, along with the `log_space_chromaticity.py` script.
3. **Ensure** the annotation `.csv` file is located in the data folder along with the `/done` subdirectory.
4. **Run the script** from the command line:
   ```bash
   python log_space_chromaticity.py path/to/data/dir
5.	**Processed Images**:
    * The processed .tiff files will be saved in the /processed folder within the data directory.
    * A new folder called /chromaticity_results will appear in the working directory, containing comparison images with:
        * Original Image
        * Standard Chromaticity Image
        * Log Space Chromaticity Image


## Acknowledgements
1. Bruce Maxwell, PhD
2. Chang Liu, Yunyi Chi, Ping He
3. [Maxwell, Bruce & Friedhoff, Richard & Smith, Casey. (2008). A bi-illuminant dichromatic reflection model for understanding images. 26th IEEE Conference on Computer Vision and Pattern Recognition, CVPR. 10.1109/CVPR.2008.4587491.](https://www.researchgate.net/publication/221361367_A_bi-illuminant_dichromatic_reflection_model_for_understanding_images)
4. [ruce A. Maxwell, Casey A. Smith, Maan Qraitem, Ross Messing, Spencer Whitt, Nicolas Thien, and Richard M. Friedhoff. Real-time physics-based removal of shadows and shading from road surfaces. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1277–1285, 2019.](https://cs.colby.edu/maxwell/papers/pdfs/Maxwell-WAD-2019.pdf)