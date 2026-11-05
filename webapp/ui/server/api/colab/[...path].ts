// Forwards /api/colab/<path> to the Colab server, adding the bearer token. Streams responses,
// so the SSE log endpoint works through it.
export default defineEventHandler(async (event) => {
  const { url, token } = await getColabSettings()
  if (!url || !token) {
    throw createError({ statusCode: 503, statusMessage: 'Colab server not configured. Open Settings.' })
  }
  const path = getRouterParam(event, 'path') || ''
  const { search } = getRequestURL(event)
  return proxyRequest(event, `${url}/${path}${search}`, {
    headers: { authorization: `Bearer ${token}` },
  })
})
