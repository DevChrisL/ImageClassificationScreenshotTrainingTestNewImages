import subprocess
import os
import sys

def main():
    #check if there are images in data/imagespredict
    if not any(os.path.isfile(os.path.join("data/imagepredict", f)) for f in os.listdir("data/imagepredict")):#loop that grabs directory and list of files. Then checks if items are files in directory, any() returns true
        print("No images found in data/imagepredict. Stopping execution.")
        sys.exit(1)#exit no images found

    #check if there are images in data/ogimages and training model
    if (os.path.isfile("trained_model.h5")) and not (any(os.path.isfile(os.path.join("data/ogimages", f)) for f in os.listdir("data/ogimages"))):
        print("A trained model already exists.\nBut no original images found to train model. Jumping to prediction.\n")
        runPrediction()
        sys.exit(1)#end before loop
    
    #check if there are images in data/ogimages
    if not any(os.path.isfile(os.path.join("data/ogimages", f)) for f in os.listdir("data/ogimages")):#loop that grabs directory and list of files. Then checks if items are files in directory, any() returns true
        #loop that grabs directory and list of files. Then checks if items are files in directory, any() returns true
        if not (any(os.path.isfile(os.path.join("data/processed/fakepreprocessed", f)) for f in os.listdir("data/processed/fakepreprocessed"))) and not (any(os.path.isfile(os.path.join("data/processed/realpreprocessed", f)) for f in os.listdir("data/processed/realpreprocessed"))):
            print("No images found in data/ogimages or data/processed. Stopping execution.")
            sys.exit(1)#exit no images found
        else:
            print("No trained model found, training the model with available processed images...\n")
            runScripts()#run all steps if user chooses re-train
            runPrediction()
            sys.exit(1)#exit
            

    if os.path.isfile("trained_model.h5"):#check for training model
        while True:#loop for user input
            choice = input("A trained model already exists and images for training found.\n1. Re-train model\n2. Jump to prediction\nYour choice:")
            if choice == "1":
                runScripts()#run all steps if user chooses re-train
                runPrediction()
                break
            elif choice == "2":
                runPrediction()#just predict
                break
            else:
                print("\nInvalid choice, try again.\n")#if anything but 1 or 2 is typed in then redo loop.
    else:
        print("No trained model found, training the model...")
        runScripts()#run all steps if no trained model exists
        runPrediction()#run image predict

def runScripts():
    scripts = ["renumberimage.py", "makefakes.py", "resize.py", "imageclassification.py"]
    for script in scripts:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)#run each script in order
        print(f"Finished {script}")

def runPrediction():
    subprocess.run(["python", "imagepredict.py"], check=True)#run prediction script

if __name__ == "__main__":
    main()