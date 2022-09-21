# PyTouch Sensors

Python interface for the tactile sensors
This will become a generic interface for connecting with numerous tactile sensors

## Installation

Clone the repository and install the package using:

```bash
git clone https://github.com/facebookresearch/pytouch.git
cd pytouch/sensors
pip install -r requirements.txt
python setup.py install
```

If you cannot access the device by serial number on your system follow [adding udev Rule](#adding-udev-rule)

### Adding udev Rule
Add your user to the ```plugdev``` group,

```
sudo adduser username plugdev
```

Copy udev rule,

```
sudo cp ./udev-rules/*.rules /lib/udev/rules.d/
```

Reload rules,

```
sudo udevadm control --reload
sudo udevadm trigger
```

Replug the tactile sensor into host.
