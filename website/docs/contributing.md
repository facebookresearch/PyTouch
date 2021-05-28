---
id: contributing
title: Contributing to PyTouch
sidebar_label: Contributing
---

We welcome the touch sensing community to contribute new tasks to the PyTouch library.

![Contribution Process](/img/contributing/process.png)

### Experiment

Perform your experiment with any suitable platform and export the models as `PyTorch` models or `ONNX` models.

### PyTouch Model

Create a new model file `task_name.py` in `pytouch/models`. This model file can contain the necessary setup for initializing from pre-trained weights and parameters.

### PyTouch Task

Create a new task file `task_name.py` in `pytouch/tasks`.

An example template for a task,

```python
class MyNewTask(nn.Module):
    def __init__(
        self,
        sensor,
        zoo_model="default_model_to_use",
        model_path=None,
        transform=None,
        defaults=MyNewTaskDefaults,
        **kwargs
    ):
        self.sensor = sensor
        self.model_path = model_path
        self.transform = transform if transform is not None else self._transforms()
        self.defaults = defaults

        if model_path is not None:
            # load custom model from path
            state_dict = PyTouchZoo.load_model(model_path)
        else:
            # load model from pytouch zoo
            zoo = PyTouchZoo()
            state_dict = zoo.load_model_from_zoo(zoo_model, sensor)
        self.model = MyNewTaskModel(state_dict=state_dict)

    def __call__(self, input):
        return self.task(frame)

    def task(self, input):
        output = ... #process inputs
        return output
```

### Verification

Create a new PR on the GitHub repository and provide the following info:

- Model file(s)
- Model description
- Model performance information

### Submission

Upon PR acceptance, the model files will be available through the PyTouch Model Zoo.
