/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

module.exports = {
  docs: {
    Introduction: ['intro', 'install', 'contributing'],
    Tutorials: [
      'tutorials/intro',
      {
        type: 'category',
        label: 'Basic Tutorial',
        items: [
          'tutorials/basic/image_input',
          'tutorials/basic/video_input',
          'tutorials/basic/sensor_input',
        ],
      },
      {
        type: 'category',
        label: 'Task Tutorial',
        items: [
          'tutorials/tasks/touch',
          'tutorials/tasks/slip',
          'tutorials/tasks/contact_area',
        ],
      },
    ],
  },
};
