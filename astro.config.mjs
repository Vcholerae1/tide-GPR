import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

const guideItems = [
  'api-orientation',
  'modeling',
  'inversion',
  'workflow-optim',
  'configuration',
  'sources-receivers',
  'boundaries',
  'storage',
  'callbacks',
  'performance',
  'validation',
  'limitations',
  'verification',
].map((slug) => ({ slug: `guides/${slug}` }));

const exampleItems = [
  'multiscale-inversion',
  'source-wavelet-inversion',
  'simultaneous-sources',
].map((slug) => ({ slug: `examples/${slug}` }));

const apiItems = [
  'index',
  'tide',
  'core',
  'maxwell',
  'workflow',
  'wavelets',
  'callbacks',
  'resampling',
  'cfl',
  'padding',
  'validation',
  'staggered',
  'utils',
  'storage',
  'backend_utils',
  'csrc',
].map((slug) => ({ slug: slug === 'index' ? 'api' : `api/${slug}` }));

const developerItems = [
  'architecture',
  'build',
  'cuda',
  'feature-lifecycle',
  'feature-matrix',
].map((slug) => ({ slug: `dev/${slug}` }));

export default defineConfig({
  site: 'https://vcholerae1.github.io',
  base: '/tide-GPR',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [
    mermaid({
      theme: 'base',
      autoTheme: false,
      enableLog: false,
      mermaidConfig: {
        flowchart: {
          curve: 'linear',
          useMaxWidth: true,
        },
        themeVariables: {
          background: '#f8f4ea',
          primaryColor: '#eee5d6',
          primaryTextColor: '#173f39',
          primaryBorderColor: '#9f4933',
          lineColor: '#536d66',
          secondaryColor: '#e5eee9',
          tertiaryColor: '#f8f4ea',
          clusterBkg: '#f8f4ea',
          clusterBorder: '#b8a991',
        },
        securityLevel: 'strict',
      },
    }),
    starlight({
      title: 'TIDE',
      description:
        'Differentiable electromagnetic modeling and inversion in PyTorch.',
      favicon: '/tide-GPR/favicon.svg',
      logo: {
        src: './src/assets/tide-mark.svg',
        replacesTitle: false,
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Vcholerae1/tide-GPR',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/Vcholerae1/tide-GPR/edit/main/',
      },
      customCss: ['./src/styles/custom.css'],
      head: [
        {
          tag: 'meta',
          attrs: {
            name: 'theme-color',
            content: '#f3efe5',
          },
        },
      ],
      sidebar: [
        {
          label: 'Start here',
          items: [
            { slug: 'overview' },
            { slug: 'getting-started' },
          ],
        },
        { label: 'Research workflows', items: guideItems },
        { label: 'Worked examples', items: exampleItems },
        { label: 'API reference', items: apiItems },
        { label: 'Developer internals', items: developerItems },
      ],
    }),
  ],
});
