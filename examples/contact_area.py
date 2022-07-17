# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pytouch
from pytouch.handlers import ImageHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import ContactArea


def extract_surface_contact():
    base_img_path = "./path/to/img/"
    sample_img_path = "./path/to/img"

    base_img = ImageHandler(base_img_path).nparray
    sample_img = ImageHandler(sample_img_path).nparray
    sample_img_2 = sample_img.copy()

    # initialize with default configuration of ContactArea task
    pt = pytouch.PyTouch(DigitSensor, tasks=[ContactArea])
    major, minor = pt.ContactArea(sample_img, base=base_img)

    print("Major Axis: {0}, minor axis: {1}".format(*major, *minor))
    ImageHandler.save("surface_contact_1.png", sample_img)

    # initialize with custom configuration of ContactArea task
    contact_area = ContactArea(base=base_img, contour_threshold=10)
    major, minor = contact_area(sample_img_2)

    print("Major Axis: {0}, minor axis: {1}".format(*major, *minor))
    ImageHandler.save("surface_contact_2.png", sample_img_2)


def extract_surface_contact_real_time():
    """
    This function demonstrates how to use the ContactArea task with real-time mode with DIGIT sensor.
    Make sure to have a DIGIT sensor connected to your computer and setup the digit-interface library.
    Output:
        Displays a real-time object contact area by fitting an ellipse.
    """
    from digit_interface import Digit
    import cv2
    base_img_path = "/home/shuk/digits2/images/0000/0041.png"
    base_img = ImageHandler(base_img_path).nparray
    # setting up the DIGIT sensor
    digit = Digit("Dxxxxx")  # replace xxxxx with your DIGIT serial number
    digit.connect()
    while True:
        frame=digit.get_frame()
        img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # target image frame for __call__ method
        contact_area=ContactArea(base=base_img, contour_threshold=10,real_time=True)
        contact_area(target=img) # __call__ method
        cv2.imshow('surface contact', img)
        k=cv2.waitKey(1)
        if k%256==27:
            #ESC hit
            print("Escape hit, closing...")
            break


if __name__ == "__main__":
    #extract_surface_contact()
    extract_surface_contact_real_time()
