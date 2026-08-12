import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

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
  integrations: [
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
        { label: 'API reference', items: apiItems },
        { label: 'Developer internals', items: developerItems },
      ],
    }),
  ],
});
