# MMPretrain

Allows processing of images with [MMPretrain](https://github.com/open-mmlab/mmpretrain).

Uses PyTorch 1.9.0 and CPU.

## Version

MMPretrain github repo tag/hash:

```
v1.2.0
17a886cb5825cd8c26df4e65f7112d404b99fe12
```

and timestamp:

```
January, 5th 2024
```

## Quick start

### Inhouse registry

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmpretrain:1.2.0_cpu
  ```

### Docker hub

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --shm-size 8G \
    -v /local/dir:/container/dir \
    -it waikatodatamining/mmpretrain:1.2.0_cpu
  ```

### Build local image

* Build the image from Docker file (from within /path_to/mmpretrain/1.2.0_cpu)

  ```bash
  docker build -t mmpre .
  ```
  
* Run the container

  ```bash
  docker run --shm-size 8G -v /local/dir:/container/dir -it mmpre
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

## Publish images

### Build

```bash
docker build -t mmpretrain:1.2.0_cpu .
```

### Inhouse registry  

* Tag

  ```bash
  docker tag \
    mmpretrain:1.2.0_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmpretrain:1.2.0_cpu
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmpretrain:1.2.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

### Docker hub  

* Tag

  ```bash
  docker tag \
    mmpretrain:1.2.0_cpu \
    waikatodatamining/mmpretrain:1.2.0_cpu
  ```
  
* Push

  ```bash
  docker push waikatodatamining/mmpretrain:1.2.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login
  ``` 

## Scripts

The following scripts are available:

* `mmpre_config` - for expanding/exporting default configurations (calls `/mmpretrain/tools/misc/print_config.py`)
* `mmpre_train` - for training a model (calls `/mmpretrain/tools/train.py`)
* `mmpre_predict_poll` - for applying a model to images (uses file-polling, calls `/mmpretrain/tools/predict_poll.py`)
* `mmpre_predict_redis` - for applying a model to images (via [Redis](https://redis.io/) backend), 
  add `--net=host` to the Docker options (calls `/mmpretrain/tools/predict_redis.py`)


## Usage

* The dataset has a simple format, with each sub-folder representing a class.
  
* Store class names in an environment variable called `MMPRE_CLASSES` **(inside the container)**:

  ```bash
  export MMPRE_CLASSES=\'class1\',\'class2\',...
  ```
  
* Alternatively, have the class anmes stored in a text file with the classes separated by commas and the `MMPRE_CLASSES`
  environment variable point at the file.
  
  * The classes are stored in `/data/labels.txt` either as comma-separated list (`class1,class2,...`) or one per line.
  
  * Export `MMPRE_CLASSES` as follows:

    ```bash
    export MMPRE_CLASSES=/data/labels.txt
    ```

* Use `mmpre_config` to export the config file (of the model you want to train) from `/mmpretrain/configs` 
  (inside the container), then follow [these instructions](#config).

* Train

  ```bash
  mmpre_train /path_to/your_data_config.py \
      --work-dir /where/to/save/everything
  ```

* Predict and output JSON files with the classes and their associated scores

  ```bash
  mmpre_predict_poll \
      --model /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --prediction_in /path_to/test_imgs \
      --prediction_out /path_to/test_results
  ```
  Run with `-h` for all available options.

* Predict via Redis backend

  You need to start the docker container with the `--net=host` option if you are using the host's Redis server.

  The following command listens for images coming through on channel `images` and broadcasts
  predicted images on channel `predictions`:

  ```bash
  mmpre_predict_redis \
      --model /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --redis_in images \
      --redis_out predictions
  ```
  
  Run with `-h` for all available options.


## Example config files

You can output example config files using (stored under `/mmpretrain/configs` for the various network types):

```bash
mmpre_config /path/to/my_config.py
```

You can browse the config files [here](https://github.com/open-mmlab/mmpretrain/tree/v0.25.0/configs).


## <a name="config">Preparing the config file</a>

* If necessary, change `num_classes` to number of labels (background not counted).
* Change `dataset_type` to `ExternalDataset` and any occurrences of `type` in the `train`, `test`, `val` 
  sections of the `data` dictionary.
* Change `data_prefix` to the path of your dataset parts (the directory containing `train` and `val` directories).
* Set `ann_file` occurrences to `None`   
* Interval in `checkpoint_config` will determine the frequency of saving models while training 
  (10 for example will save a model after every 10 epochs).
* In the `runner` property, change `max_epocs` to how many epochs you want to train the model for.
* Change `load_from` to the file name of the pre-trained network that you downloaded from the model zoo instead
  of downloading it automatically.
* If you want to include the validation set, add `, ('val', 1)` to `workflow`.

_You don't have to copy the config file back, just point at it when training._

**NB:** A fully expanded config file will get placed in the output directory with the same
name as the config plus the extension *.full*.


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```bash
docker run -u $(id -u):$(id -g) -e USER=$USER ...
```

## Caching models

PyTorch downloads base models, if necessary. However, by using Docker, this means that 
models will get downloaded with each Docker image, using up unnecessary bandwidth and
slowing down the startup. To avoid this, you can map a directory on the host machine
to cache the base models for all processes (usually, there would be only one concurrent
model being trained):  

```
-v /somewhere/local/cache:/.cache
```

Or specifically for PyTorch:

```
-v /somewhere/local/cache/torch:/.cache/torch
```

**NB:** When running the container as root rather than a specific user, the internal directory will have to be
prefixed with `/root`. 
