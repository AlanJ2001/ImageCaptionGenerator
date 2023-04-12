Guide to Running the Software

Training the model
	run "Data pre-processing.py" which will generate four npy files: img_feature_vectors.npy, captions_dict.npy, captions_dict_encoded.npy, vocab.npy.
		should be kept in the "data" folder. 
	run "generate_training_data.py" which will generate the training data and testing data as npy files. keep these npy files in the "data" folder. 
	run "training.py" which will train the model and save the weights of the model after each epoch as an h5 file. 

Generating Captions
	add image to the same folder as "Inference.py"
	in the main method of "Inference.py" modify filename variable to match the candidate image
	run "Inference.py" to generate a caption for the image using Greedy Search, Beam Search and Nucleus Sampling
	In "Inference.py", "training.py" and "BLEU evaluation.py", the parameter vocab_size may need to be updated to match the size of the vocab dictionary.

Using the Web App
	navigate to the web_app directory
	run "streamlit run web_app.py"
	the web app allows you to upload an image, choose one of three inference algorithms and will generate a caption for the image.

Some files have not been included in the submission as they are too large. 
	Flickr30k dataset can be downloaded at https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
	all other data files not included in submission can be downloaded at https://drive.google.com/drive/folders/1CciO8bk4lnaXyneP70uYyhvuDoal1v2g 
	
![figure10](https://user-images.githubusercontent.com/77545869/231460437-ffdded1f-3e7f-4cf8-a996-d6a4acd9964c.PNG)
