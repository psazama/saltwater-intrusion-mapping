import { MapContainer, TileLayer, Rectangle, useMap } from 'react-leaflet'
import { useEffect, useRef } from 'react'
import L from 'leaflet'

const SENSOR_COLORS = {
  'sentinel-2': '#4a90e2',
  'landsat-5': '#e2844a',
  'landsat-7': '#4ae2a0',
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

// Compute the union bounding box across all scenes
function unionBounds(scenes) {
  const allBounds = scenes
    .map((s) => parseBbox(s.location_wkt || ''))
    .filter(Boolean)
  if (!allBounds.length) return null

  let minLat = Infinity, minLon = Infinity
  let maxLat = -Infinity, maxLon = -Infinity
  for (const [[sLat, sLon], [nLat, nLon]] of allBounds) {
    minLat = Math.min(minLat, sLat)
    minLon = Math.min(minLon, sLon)
    maxLat = Math.max(maxLat, nLat)
    maxLon = Math.max(maxLon, nLon)
  }
  return [[minLat, minLon], [maxLat, maxLon]]
}

function ProductTileLayer({ selectedProduct, titilerUrl }) {
  const map = useMap()
  const layerRef = useRef(null)

  useEffect(() => {
    // Remove existing layer
    if (layerRef.current) {
      map.removeLayer(layerRef.current)
      layerRef.current = null
    }

    if (!selectedProduct) {
      return
    }

    const tifPath = selectedProduct.output_paths?.find((p) => p.endsWith('.tif'))
    if (!tifPath) {
      return
    }

    const encodedPath = encodeURIComponent(`/data/${tifPath}`)
    const tilesUrl = `/tiles/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=${encodedPath}&rescale=0,1&colormap_name=blues`

    layerRef.current = L.tileLayer(tilesUrl, { opacity: 0.2 })
    layerRef.current.addTo(map)

    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current)
        layerRef.current = null
      }
    }
  }, [selectedProduct, titilerUrl, map])

  return null
}

export default function SceneMap({ 
  scenes, 
  selectedSceneId, 
  hoveredSceneId,
  selectedProduct, 
  titilerUrl }) {

  const aggregate = unionBounds(scenes)
  const hoveredScene = scenes.find((s) => s.scene_id === hoveredSceneId)
  const selectedScene = scenes.find((s) => s.scene_id === selectedSceneId)

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

      {/* 1. Aggregate extent — light dashed outline, always shown */}
      {aggregate && (
        <Rectangle
          bounds={aggregate}
          pathOptions={{
            color: '#888',
            weight: 1,
            dashArray: '4',
            fill: false,
          }}
          interactive={false}
        />
      )}

      {/* 2. Hovered scene — medium highlight */}
      {hoveredScene && hoveredScene.scene_id !== selectedSceneId && (() => {
        const bounds = parseBbox(hoveredScene.location_wkt || '')
        if (!bounds) return null
        return (
          <Rectangle
            bounds={bounds}
            pathOptions={{
              color: SENSOR_COLORS[hoveredScene.sensor] || '#999',
              weight: 2,
              fillOpacity: 0.1,
            }}
            interactive={false}
          />
        )
      })()}

      {/* 3. Selected scene — prominent */}
      {selectedScene && (() => {
        const bounds = parseBbox(selectedScene.location_wkt || '')
        if (!bounds) return null
        return (
          <Rectangle
            bounds={bounds}
            pathOptions={{
              color: SENSOR_COLORS[selectedScene.sensor] || '#999',
              weight: 3,
              fillOpacity: 0.05,
            }}
          />
        )
      })()}

      <ProductTileLayer selectedProduct={selectedProduct} titilerUrl={titilerUrl} />
    </MapContainer>
  )
}