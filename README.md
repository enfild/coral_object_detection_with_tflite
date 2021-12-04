# Simple example for runtime Google Coral (C++)

## For using this code you should install tflite and libedgetpu-dev

### tflite (tf2.5 is tested):
```
git clone https://github.com/tensorflow/tensorflow
# build shell
cd tensorflow/tensorflow/lite/tools/make/
./download_dependencies.sh
./build_lib.sh
# build cmake
mkdir tflite_build
cd tflite_build
cmake ../tensorflow/tensorflow/lite
```

### libedgetpu-dev:
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
echo "deb https://packages.cloud.google.com/apt coral-cloud-stable main" | sudo tee /etc/apt/sources.list.d/coral-cloud.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt install libedgetpu-dev
```
