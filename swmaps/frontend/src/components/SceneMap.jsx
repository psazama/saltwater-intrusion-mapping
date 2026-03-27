import { MapContainer, TileLayer, Rectangle, Tooltip, useMap } from 'react-leaflet'
import { useEffect } from 'react'

const SENSOR_COLORS = {
  'sentinel-2': '#4a90e2',
  'landsat-5': '#e2844a',
  'landsat-7': '#4ae2a0',
}

function FitBounds({ scenes }) {
  const map = useMap()
  useEffect(() => {
    if (!scenes.length) return
    // Fit map to scene extents on first load
  }, [scenes])
  return null
}

function parseBbox(wkt) {
  // Parse WKT POLYGON to [[minLat, minLon], [maxLat, maxLon]] for Leaflet
  const nums = wkt.match(/-?[\d.]+/g)?.map(Number)
  if (!nums || nums.length < 8) return null
  const lons = [nums[0], nums[2], nums[4], nums[6]]
  const lats = [nums[1], nums[3], nums[5], nums[7]]
  return [
    [Math.min(...lats), Math.min(...lons)],
    [Math.max(...lats), Math.max(...lons)],
  ]
}

export default function SceneMap({ scenes, selectedSceneId, onSelectScene }) {
  return (
    <MapContainer
      center={[38.5, -76.0]}
      zoom={8}
      style={{ height: '100%', width: '100%' }}
    >
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {scenes.map((scene) => {
        const bounds = parseBbox(scene.location_wkt || '')
        if (!bounds) return null
        const color = SENSOR_COLORS[scene.sensor] || '#999'
        const isSelected = scene.scene_id === selectedSceneId
        return (
          <Rectangle
            key={scene.scene_id}
            bounds={bounds}
            pathOptions={{
              color,
              weight: isSelected ? 3 : 1,
              fillOpacity: isSelected ? 0.3 : 0.1,
            }}
            eventHandlers={{
              click: () => onSelectScene(scene.scene_id),
            }}
          >
            <Tooltip>
              <div style={{ fontSize: 11 }}>
                <div>{scene.scene_id}</div>
                <div>{scene.sensor} · {scene.acquisition_date}</div>
              </div>
            </Tooltip>
          </Rectangle>
        )
      })}
    </MapContainer>
  )
}