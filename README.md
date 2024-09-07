# SignSync
Setup Instructions
Set Up the Environment:

1)Use the command prompt to set up your Python environment with the required packages:
    -->python -m pip install -r install_packages.txt
    -->python -m pip install -r install_packages_gpu.txt
                    This will install all the necessary libraries for the project.

2)Prepare the Histogram:
    -->python set_hand_histogram.py 
      script to set the hand histogram for creating gestures. This script will help you capture the hand histogram needed for gesture recognition.
      Run the set_hand_histogram.py
      Note: To allow capturing multiple histograms without restarting the script, you need to modify the set_hand_histogram.py code as follows:
                          Store Multiple Histograms: Adjust the code to store multiple histograms and update them each time 'C' is pressed.
                          Add a Mechanism to Select and Save Histograms: Implement a way to choose which histogram to save or save all histograms to different files.
3)Create Gestures:
      Capture and label gestures using OpenCV with the create_gestures.py script, which stores the gestures in a database.
      -->python create_gestures.py

4)Add Variations to Captured Gestures:
      Use the Rotate_images.py script to add variations to the captured gestures by flipping the images.
      -->python Rotate_images.py

5)Prepare Data for Training:
      Split all the captured gestures into training, validation, and test sets using the load_images.py script.
      -->python load_images.py

6)View Gestures:
      View all the captured gestures by running the display_gestures.py script.
      -->python display_gestures.py

7)Train the Model:
      Train the gesture recognition model using Keras with the cnn_model_train.py script.
      -->python cnn_model_train.py

8)Run Gesture Recognition:
      Execute the final.py script to open the gesture recognition window, which will use your webcam to interpret the trained American Sign Language gestures.
      -->python final.py
