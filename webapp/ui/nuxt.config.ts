// The UI runs on your computer; all GPU work happens on the Colab server behind the Nitro proxy
// (server/api/colab/[...path].ts), so the tunnel URL and token never reach the browser.
export default defineNuxtConfig({
  compatibilityDate: '2026-11-01',
  ssr: false,
  modules: ['@nuxt/ui'],
  css: ['~/assets/css/main.css'],
  devtools: { enabled: true },
  runtimeConfig: {
    // Optional defaults, overridden by the Settings page: NUXT_COLAB_URL, NUXT_COLAB_TOKEN
    colabUrl: '',
    colabToken: '',
  },
  app: {
    head: {
      title: 'Lao OCR Trainer',
      link: [
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;600&display=swap' },
      ],
    },
  },
})
