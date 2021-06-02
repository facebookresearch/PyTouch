/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'PyTouch',
  tagline: 'A Machine Learning Library for Touch Processing',
  url: 'https://www.touch-sensing.org/',
  baseUrl: '/PyTouch/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'facebookresearch',
  projectName: 'pytouch',
  themeConfig: {
    navbar: {
      title: 'PyTouch',
      logo: {
        alt: 'PyTouch Project Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: 'docs/',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        // {to: 'blog', label: 'Blog', position: 'left'},
        // // Please keep GitHub link to the right for consistency.
        // {
        //   href: 'https://github.com/facebook/docusaurus',
        //   label: 'GitHub',
        //   position: 'right',
        // },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'About PyTouch',
              to: 'docs/',
            },
            {
              label: 'Installation',
              to: 'docs/install',
            },
            {
              label: 'Tutorials',
              to: 'docs/tutorials/intro',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Touch Sensing',
              href: 'https://www.touch-sensing.org/',
            },
            {
              label: 'DIGIT',
              href: 'https://digit.ml',
            },
            {
              label: 'TACTO',
              href: 'https://github.com/facebookresearch/tacto',
            },
          ],
        },
        {
          title: 'More',
          items: [
            // {
            //   label: 'Blog',
            //   to: 'blog',
            // },
            {
              label: 'GitHub',
              href: 'https://github.com/facebookresearch/pytouch',
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
            },
            {
              label: 'Data Policy',
              href: 'https://opensource.facebook.com/legal/data-policy/',
            },
            {
              label: 'Cookie Policy',
              href: 'https://opensource.facebook.com/legal/cookie-policy/',
            },
          ],
        },
      ],
      logo: {
        alt: 'Facebook Open Source Logo',
        src: 'img/oss_logo.png',
        href: 'https://opensource.facebook.com',
      },
      // Please do not remove the credits, help to publicize Docusaurus :)
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/facebookresearch/pytouch/edit/master/website/',
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/facebookresearch/pytouch/edit/master/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
