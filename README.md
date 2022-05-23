# facial-expression-recognition

Make sure you have at least python 3.9 installed. If the dlm folder is empty, you need to clone the repository.

```
git clone https://github.com/GiulioZani/dlpm
```

Then you can run the following command to install the dependencies:

```
pip install -r requirements.txt
```

## Models
Models are located at `dl/models/`, each folder name represents a model and the file `default_parameters.json` reperents the parameters for the model.

## Running
To train a model, you can use the following command:

```
python -m dlm train <model_name>
```

For example:
```
python -m dlm train conv2d_pool_fchannel_1024
```

To test a model, you can use the following command:

```
python -m dlm test <model_name>
```

To run the webcam visualization and real-time classification of the model, you can use the following command:

```
python -m dlm custom-test <model_name>
```

### Setting parameters for the model
For each model, you can set the parameters by editing the file `default_parameters.json`, for example setting cuda to false will run the model on cpu.

