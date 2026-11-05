export default defineEventHandler(async (event) => {
  const body = await readBody<{ url?: string, token?: string }>(event)
  const url = (body?.url || '').trim().replace(/\/+$/, '')
  if (!/^https?:\/\/[^\s/]+/.test(url)) {
    throw createError({ statusCode: 400, statusMessage: 'URL must start with http:// or https://' })
  }
  const current = await getColabSettings()
  // An empty token field keeps the saved one, so you only paste the new tunnel URL each session.
  const token = (body?.token || '').trim() || current.token
  await saveColabSettings({ url, token })
  return { url, hasToken: Boolean(token) }
})
