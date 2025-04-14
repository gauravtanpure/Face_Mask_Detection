# -*- coding: utf-8 -*-
"""
Updated Face Mask Detection with Unicode Handling
"""

import numpy as np
import cv2
import datetime
import tensorflow as tf
import os
import sys

# Fix Unicode console encoding issue
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a function to build the model
def build_mask_detector_model(input_shape=(150, 150, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_mask_detector_model():
    model_path = 'mymodel.keras'
    
    # Check if model already exists
    if os.path.exists(model_path):
        print("Loading saved model...")
        try:
            mymodel = load_model(model_path)
            return mymodel
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Will train a new model instead.")
    
    print("Building and training new model...")
    # Build the model
    model = build_mask_detector_model()
    
    # Ensure directory paths exist and print their contents
    for dir_path in ['train', 'test']:
        if not os.path.exists(dir_path):
            print(f"Error: Directory '{dir_path}' not found!")
            print(f"Please create the directory structure with mask/no mask images")
            return None
        
        print(f"Found directory: {dir_path}")
        subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        print(f"Subdirectories: {subdirs}")
        
        # Check if we have proper class folders
        if 'with_mask' not in subdirs or 'without_mask' not in subdirs:
            print(f"Warning: Expected 'with_mask' and 'without_mask' subdirectories in {dir_path}")
            
        # Count images in each subdirectory
        for subdir in subdirs:
            subdir_path = os.path.join(dir_path, subdir)
            files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  - {subdir}: {len(files)} images")
    
    # Set up data generators with ASCII validation to avoid encoding errors
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    try:
        # Use binary_crossentropy_with_flow_directory for ASCII validation
        training_set = train_datagen.flow_from_directory(
            'train',
            target_size=(150, 150),
            batch_size=16,
            class_mode='binary')

        test_set = test_datagen.flow_from_directory(
            'test',
            target_size=(150, 150),
            batch_size=16,
            class_mode='binary')
            
        print(f"Found {training_set.samples} training images in {len(training_set.class_indices)} classes")
        print(f"Found {test_set.samples} test images in {len(test_set.class_indices)} classes")
        
        # Train the model with error handling
        try:
            history = model.fit(
                training_set,
                epochs=10,
                validation_data=test_set,
                verbose=1
            )
            
            # Save the model
            model.save(model_path)
            print(f"Model successfully saved to {model_path}")
            return model
            
        except Exception as e:
            print(f"Error during model training: {e}")
            return None
            
    except Exception as e:
        print(f"Error setting up data generators: {e}")
        return None

# Function to test individual images
def test_individual_image(model, image_path):
    try:
        test_image = load_img(image_path, target_size=(150, 150))
        test_image = img_to_array(test_image)
        test_image = test_image / 255.0  # Normalize
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image, verbose=0)[0][0]
        return prediction
    except Exception as e:
        print(f"Error testing image: {e}")
        return None

# Function for live face mask detection
def detect_mask_in_video(model):
    if model is None:
        print("No valid model available for detection.")
        return
        
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        return
    
    # Load face detector
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        print(f"Error: Face cascade file not found at {face_cascade_path}")
        print("Please download it from OpenCV GitHub repository")
        print("You can download it from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
        return
        
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    print("Starting video capture. Press 'q' to quit.")
    
    temp_path = 'temp.jpg'
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_img = img[y:y+h, x:x+w]
            
            # Save temporary image
            try:
                cv2.imwrite(temp_path, face_img)
            except Exception as e:
                print(f"Error saving temporary image: {e}")
                continue
            
            # Process and predict
            try:
                test_image = load_img(temp_path, target_size=(150, 150))
                test_image = img_to_array(test_image)
                test_image = test_image / 255.0  # Normalize
                test_image = np.expand_dims(test_image, axis=0)
                
                pred = model.predict(test_image, verbose=0)[0][0]
                
                # Draw rectangle and text based on prediction
                if pred > 0.5:  # Threshold might need adjustment
                    # No mask
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(img, 'NO MASK', ((x+w)//2, y+h+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Mask
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(img, 'MASK', ((x+w)//2, y+h+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error in prediction: {e}")
        
        # Add timestamp (using only ASCII characters)
        datet = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, datet, (10, img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Face Mask Detection', img)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Clean up temp file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Error removing temporary file: {e}")

# Main execution
if __name__ == "__main__":
    # Fix potential Unicode issues in file paths
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Train or load the model
    model = train_mask_detector_model()
    
    if model is not None:
        # For live detection
        detect_mask_in_video(model)
    else:
        print("Failed to load or train model. Please check the errors above.")