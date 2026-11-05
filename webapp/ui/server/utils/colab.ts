export interface ColabSettings {
  url: string
  token: string
}

const KEY = 'colab-settings'

/** Settings saved from the UI (in .data/, never sent to the browser) win over NUXT_COLAB_* env vars. */
export async function getColabSettings(): Promise<ColabSettings> {
  const stored = await useStorage('data').getItem<ColabSettings>(KEY)
  const cfg = useRuntimeConfig()
  return {
    url: stored?.url || (cfg.colabUrl as string) || '',
    token: stored?.token || (cfg.colabToken as string) || '',
  }
}

export async function saveColabSettings(s: ColabSettings) {
  await useStorage('data').setItem(KEY, s)
}
