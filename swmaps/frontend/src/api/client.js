const BASE = import.meta.env.DEV ? '/api' : ''

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function fetchScenes({ bbox, sensor, dateFrom, dateTo }) {
  const params = new URLSearchParams()
  if (bbox) params.set('bbox', bbox)
  if (sensor) params.set('sensor', sensor)
  if (dateFrom) params.set('date_from', dateFrom)
  if (dateTo) params.set('date_to', dateTo)
  return get(`/scenes?${params}`)
}

export async function fetchScene(sceneId) {
  return get(`/scenes/${sceneId}`)
}

export async function fetchSceneProducts(sceneId) {
  return get(`/scenes/${sceneId}/products`)
}

export async function fetchSalinityProfiles({ bbox, dateFrom, dateTo }) {
  const params = new URLSearchParams()
  if (bbox) params.set('bbox', bbox)
  if (dateFrom) params.set('date_from', dateFrom)
  if (dateTo) params.set('date_to', dateTo)
  return get(`/salinity/profiles?${params}`)
}

export async function fetchRuns({ task, status } = {}) {
  const params = new URLSearchParams()
  if (task) params.set('task', task)
  if (status) params.set('status', status)
  return get(`/runs?${params}`)
}

export function productImageUrl(outputPath) {
  // Convert a file path to a URL served by FastAPI's /preview endpoint
  return `${BASE}/preview?path=${encodeURIComponent(outputPath)}`
}

export async function fetchTasks() {
  return get('/tasks')
}

export async function fetchSensors() {
  return get('/sensors')
}