# SignSync
Try Execution 
step1- Download the code on your system and go to cmd promt
step2- download your python env if already there Update it (Run this if you still get an error------ python -m pip install --upgrade pip setuptools ------)
step3- python -m pip install -r install_packages.txt
step4- python -m pip install -r install_packages_gpu.txt
step5- python set_hand_histogram.py
       More steps within 5
        -->Press 'C': When the 'c' key is pressed, it captures the pixel data from the grid generated by build_squares. This pixel data is then used to create a color histogram (hist), which represents the hand color.
        --> Press 'S': When the 's' key is pressed, it exits the loop, indicating that the hand histogram is ready.
        -->Note--With new gesture you need to press c again if its in the frame before showing the next one
        -->also press S at the last once You have done with the C part 
        
