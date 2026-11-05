export default defineEventHandler(async () => {
  const { url, token } = await getColabSettings()
  return { url, hasToken: Boolean(token) }
})
