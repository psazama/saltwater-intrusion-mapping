import { MapContainer, TileLayer, Rectangle, Tooltip, useMap } from 'react-leaflet'
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
      console.log('[TileLayer] no product selected')
      return
    }

    const tifPath = selectedProduct.output_paths?.find((p) => p.endsWith('.tif'))
    console.log('[TileLayer] selectedProduct:', selectedProduct)
    console.log('[TileLayer] tifPath:', tifPath)
    if (!tifPath) {
      console.log('[TileLayer] no tif path found in', selectedProduct.output_paths)
      return
    }

    const encodedPath = encodeURIComponent(`/data/${tifPath}`)
    const tilesUrl = `/tiles/cog/tiles/{z}/{x}/{y}.png?url=${encodedPath}&rescale=0,1&colormap_name=blues`
    console.log('[TileLayer] tilesUrl:', tilesUrl)

    layerRef.current = L.tileLayer(tilesUrl, { opacity: 0.7 })
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

export default function SceneMap({ scenes, selectedSceneId, onSelectScene, selectedProduct, titilerUrl }) {
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
      <ProductTileLayer selectedProduct={selectedProduct} titilerUrl={titilerUrl} />
    </MapContainer>
  )
}