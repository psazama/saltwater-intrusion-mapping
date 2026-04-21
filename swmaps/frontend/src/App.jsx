import { useState, useEffect, useCallback } from 'react'
import SceneMap from './components/SceneMap'
import SceneList from './components/SceneList'
import ProductPanel from './components/ProductPanel'
import { fetchScenes, fetchSceneProducts, fetchTasks, fetchSensors, fetchConfig } from './api/client'

export default function App() {
  const [scenes, setScenes] = useState([])
  const [products, setProducts] = useState({})
  const [selectedSceneId, setSelectedSceneId] = useState(null)
  const [compareMode, setCompareMode] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [tasks, setTasks] = useState([])
  const [sensors, setSensors] = useState([])
  const [titilerUrl, setTitilerUrl] = useState('http://localhost:8001')

  // Filters
  const [bbox, setBbox] = useState('-76.5,37.5,-74.5,39.5')
  const [sensor, setSensor] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  const loadScenes = useCallback(async () => {
    if (!bbox) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchScenes({ bbox, sensor, dateFrom, dateTo })
      setScenes(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [bbox, sensor, dateFrom, dateTo])

  useEffect(() => {
    loadScenes()
    fetchConfig().then((data) => setTitilerUrl(data.titiler_url)).catch(console.error)
    fetchTasks().then((data) => setTasks(data.tasks)).catch(console.error)
    fetchSensors().then((data) => setSensors(['', ...data.sensors])).catch(console.error)
  }, [])

  async function handleSelectScene(sceneId) {
    setSelectedSceneId(sceneId)
    setSelectedProduct(null)
    if (!products[sceneId]) {
      try {
        const data = await fetchSceneProducts(sceneId)
        setProducts((prev) => ({ ...prev, [sceneId]: data }))
      } catch (err) {
        console.error('Failed to load products for', sceneId, err)
      }
    }
  }

  return (
    <div className="app">
      <div className="header">
        <h1>swmaps</h1>
        <div className="header-filters">
          <input
            placeholder="bbox: minLon,minLat,maxLon,maxLat"
            value={bbox}
            onChange={(e) => setBbox(e.target.value)}
            style={{ width: 260 }}
          />
          <select value={sensor} onChange={(e) => setSensor(e.target.value)}>
            {sensors.map((s) => (
              <option key={s} value={s}>{s || 'all sensors'}</option>
            ))}
          </select>
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            placeholder="from"
          />
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            placeholder="to"
          />
          <button
            onClick={loadScenes}
            style={{
              padding: '4px 12px',
              background: '#4a90e2',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
            }}
          >
            search
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="main">
        <div className="map-pane">
          <SceneMap
            scenes={scenes}
            selectedSceneId={selectedSceneId}
            onSelectScene={handleSelectScene}
            selectedProduct={selectedProduct}
            titilerUrl={titilerUrl}
          />
          {loading && (
            <div style={{
              position: 'absolute', top: 8, left: 8,
              background: 'white', padding: '4px 8px',
              borderRadius: 4, fontSize: 12, boxShadow: '0 1px 4px rgba(0,0,0,0.2)',
            }}>
              loading...
            </div>
          )}
        </div>

        <SceneList
          scenes={scenes}
          products={products}
          selectedSceneId={selectedSceneId}
          onSelectScene={handleSelectScene}
          compareMode={compareMode}
          onToggleCompare={() => setCompareMode((m) => !m)}
          tasks={tasks}
        />
      </div>

      <ProductPanel
        sceneId={selectedSceneId}
        products={products[selectedSceneId]}
        compareMode={compareMode}
        tasks={tasks}
        onSelectProduct={setSelectedProduct}
      />
    </div>
  )
}