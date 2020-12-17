# Application usage

## Setup the enviroment:

**At first, launch the cmd and move to your working directory of openvino.**

```python
    $cd /Program Files (x86)/IntelSWTools/openvino/bin/
    setupvars.bat
```

**Modify the test_20201210_for_intel.py:**

```
Line 20:   model_xml = r"D:\Project\Thermal_python\face-detection-0200\FP32\face-detection-0200.xml"

Change the model path for your model path.

```

**And than, move to our directory:**

```python
    $cd [our working directory]
```

## Running the scricpt:
```python
    $python test_20201210_for_intel.py
```

※If wanna chage the camera can use ```-i``` + ```your cam number``` , default is 0.
